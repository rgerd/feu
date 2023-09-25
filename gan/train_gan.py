from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os.path
import sys

from discriminator import MNISTDiscriminator
from generator import MNISTGenerator
from util.torch_utils import gan_weights_init, select_device


class MNISTGanTrainer:
    def train(
        self,
        epoch_count: int,
        data_loader: DataLoader,
        models: (torch.nn.Embedding, MNISTDiscriminator, MNISTGenerator),
        optimizers: (optim.Adam, optim.SGD, optim.Adam),
        device: torch.device,
    ):
        self.device = device
        embedding, discriminator, generator = models
        (
            self.e_optimizer,
            self.d_optimizer,
            self.g_optimizer,
        ) = optimizers

        self.layer_grads = []
        self.layer_grads_to_data = []
        self.layer_data = []
        # for _ in generator.layers():
        #     self.layer_grads.append([])
        #     self.layer_grads_to_data.append([])
        #     self.layer_data.append([])
        self.gen_sat = [0.0]
        self.disc_acc = [0.0]

        for epoch in range(1, epoch_count + 1):
            print(f"Epoch {epoch}...")
            self._train_gan(discriminator, generator, data_loader)
            self._train_emb(embedding, generator, discriminator, data_loader)

    def _train_gan(
        self,
        discriminator: MNISTDiscriminator,
        generator: MNISTGenerator,
        data_loader: DataLoader,
    ):
        timer_tick = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            discriminator.zero_grad()
            loss_real, loss_fake = self._loss_discriminator(
                discriminator, generator, data, target
            )
            loss_real.backward()
            loss_fake.backward()
            self.d_optimizer.step()

            discriminator.zero_grad()
            generator.zero_grad()
            loss_g = self._loss_generator(discriminator, generator)
            loss_g.backward()
            # generator.print_grads()
            self.g_optimizer.step()

            if batch_idx % 128 == 0:
                print(
                    "[{}/{} ({:.0f}%)] ({} seconds)".format(
                        batch_idx * len(data_loader),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        time.time() - timer_tick,
                    )
                )
                print(
                    f"D: [Loss_Real: {loss_real.item()}], [Loss_Fake: {loss_fake.item()}]"
                )
                print(f"G: [Loss: {loss_g.item()}]")
                print(f"G: [Saturation: {self.gen_sat[-1] * 100.0}]")
                print(f"G: [D_acc: {self.disc_acc[-1] * 100.0}]")
                timer_tick = time.time()

    def _train_emb(
        self,
        embedding: nn.Embedding,
        generator: MNISTGenerator,
        discriminator: MNISTDiscriminator,
        data_loader: DataLoader,
    ):
        embedding.train()
        generator.eval()
        discriminator.eval()
        for batch_idx, (_, target) in enumerate(data_loader):
            target = target.to(self.device)

            self.e_optimizer.zero_grad()
            loss_e = self._loss_embedding(embedding, generator, discriminator, target)
            loss_e.backward()
            self.e_optimizer.step()

            if batch_idx % 128 == 0:
                print(
                    "E [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        batch_idx * len(data_loader),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        loss_e.item(),
                    )
                )

    def _loss_discriminator(
        self,
        discriminator: MNISTDiscriminator,
        generator: MNISTGenerator,
        data: torch.Tensor,
        target: torch.Tensor,
    ):
        (B, C, H, W) = data.shape
        rand_g_inputs_d = torch.randn((B, 32, 1, 1)).to(self.device)
        d_real_targets = torch.ones((B,)).to(self.device)
        d_fake_targets = torch.zeros((B,)).to(self.device)

        discriminator.train()
        generator.eval()
        fake_data = generator(rand_g_inputs_d)

        d_output_reals = discriminator(data)
        d_output_fakes = discriminator(fake_data.detach())

        loss_real = discriminator.calculate_loss(
            d_output_reals, (target, d_real_targets)
        )
        loss_fake = discriminator.calculate_loss(d_output_fakes, (None, d_fake_targets))
        return loss_real, loss_fake

    def _loss_generator(
        self, discriminator: MNISTDiscriminator, generator: MNISTGenerator
    ):
        rand_g_inputs = torch.randn((128, 32, 1, 1)).to(self.device)

        discriminator.train()  # Really feels like this should be eval()
        generator.train()

        g_output = generator(rand_g_inputs)
        d_class_output, d_disc_output = discriminator(g_output)
        d_accuracy = (1.0 - F.sigmoid(d_disc_output).mean()).data
        loss_g = discriminator.calculate_generator_loss((d_class_output, d_disc_output))

        with torch.no_grad():
            self.gen_sat[0] = (
                (torch.sum(torch.abs(g_output) > 0.97) / g_output.nelement()).cpu().data
            )
            self.disc_acc[0] = d_accuracy.cpu()
        return loss_g

    def _loss_embedding(
        self,
        embedding: nn.Embedding,
        generator: MNISTGenerator,
        discriminator: MNISTDiscriminator,
        target: torch.Tensor,
    ):
        d_output = discriminator(generator(embedding(target).reshape(-1, 32, 1, 1)))
        loss_e = discriminator.calculate_loss(d_output, (target, None))
        return loss_e


def main():
    parser = argparse.ArgumentParser(
        prog="MNIST Gan", description="Trains a GAN to generate MNIST numbers"
    )
    parser.add_argument("--nosave", action="store_true", required=False, default=False)
    parser.add_argument("--noload", action="store_true", required=False, default=False)
    parser.add_argument("--display", action="store_true", required=False, default=False)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # torch.manual_seed(1)
    device = select_device()

    num_epochs = 32

    train_kwargs = {"batch_size": 128}
    test_kwargs = {"batch_size": 1000}
    if torch.cuda.is_available():
        cuda_kwargs = {"shuffle": True, "num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    elif torch.backends.mps.is_available():
        mps_kwargs = {"pin_memory": True, "shuffle": True}
        train_kwargs.update(mps_kwargs)
        test_kwargs.update(mps_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (1.0,)
            ),  # Usually (0.3081,)), but allowing the std to stay 0.3 we have ~ (-1.0-1.0), which is good for tanh training.
        ]
    )
    datasetTrain = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    trainTransformed = []
    for data, target in datasetTrain:
        trainTransformed.append((data, target))
    datasetTest = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(trainTransformed, **train_kwargs)
    # TODO: Validation on discriminator classifier and discrimination
    test_loader = DataLoader(datasetTest, **test_kwargs)

    print(
        f"Train size: {len(datasetTrain)} x {train_kwargs['batch_size']} = {len(datasetTrain) * train_kwargs['batch_size']}"
    )

    embedding = torch.nn.Embedding(10, 32).to(device)
    if not args.noload and os.path.exists("./saved/embedding.pt"):
        print("Loading embedding...")
        e_saved = torch.load("./saved/embedding.pt", map_location=device)
        embedding.load_state_dict(e_saved["state_dict"])
    e_optimizer = optim.Adam(embedding.parameters(), lr=1e-1)

    discriminator = MNISTDiscriminator().to(device)
    if not args.noload and os.path.exists("./saved/discriminator.pt"):
        print("Loading discriminator...")
        d_saved = torch.load("./saved/discriminator.pt", map_location=device)
        discriminator.load_state_dict(d_saved["state_dict"])
    else:
        discriminator.apply(gan_weights_init)
    d_optimizer = optim.SGD(discriminator.parameters(), lr=2e-2)

    generator = MNISTGenerator().to(device)
    if not args.noload and os.path.exists("./saved/generator.pt"):
        print("Loading generator...")
        g_saved = torch.load("./saved/generator.pt", map_location=device)
        generator.load_state_dict(g_saved["state_dict"])
    else:
        generator.apply(gan_weights_init)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    trainer = MNISTGanTrainer()
    try:
        trainer.train(
            num_epochs,
            train_loader,
            (embedding, discriminator, generator),
            (e_optimizer, d_optimizer, g_optimizer),
            device,
        )
    except KeyboardInterrupt:
        print("Done.")
        pass
    except Exception as e:
        print(f"Exception: {e}")
        print("Done.")

    if args.display:
        _, axs = plt.subplots(3, 10, layout="constrained")
        for r in range(3):
            g_outputs = generator(
                embedding(torch.arange(10).to(device)).reshape(10, 32, 1, 1)
            )
            for c in range(10):
                axs[r, c].imshow(g_outputs[c][0].cpu().data)
        plt.show()

        plt.show()

    embedding.eval()
    generator.eval()
    discriminator.eval()

    # Evaluate
    with torch.no_grad():
        rand_g_inputs = torch.randn((1000, 32, 1, 1)).to(device)
        g_output = generator(rand_g_inputs)
        d_output = discriminator(g_output)
        d_accuracy = torch.mean(1.0 - F.sigmoid(d_output[1]))
        print(f"Discriminator accuracy: {d_accuracy}")

        if not args.nosave:
            print("Saving discriminator...")
            torch.save(
                {"state_dict": discriminator.state_dict()},
                "./saved/discriminator.pt",
            )
            print("Saving generator...")
            torch.save(
                {"state_dict": generator.state_dict()},
                "./saved/generator.pt",
            )

    print("Saving embedding...")
    torch.save(
        {"state_dict": embedding.state_dict()},
        "./saved/embedding.pt",
    )


def plot_activations(trainer: MNISTGanTrainer, generator: MNISTGenerator):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(torch.arange(len(trainer.gen_sat)).data, trainer.gen_sat)
    for layer_idx, (layer, grads, grads_to_data, data) in enumerate(
        zip(
            generator.layers(),
            trainer.layer_grads,
            trainer.layer_grads_to_data,
            trainer.layer_data,
        )
    ):
        if len(grads) > 0:
            ax2.plot(
                torch.arange(len(grads)).data,
                grads,
                label=f"({layer_idx}) {layer.__class__.__name__}",
            )
            ax3.plot(
                torch.arange(len(grads_to_data)).data,
                grads_to_data,
                label=f"({layer_idx}) {layer.__class__.__name__}",
            )
            ax4.plot(
                torch.arange(len(data)).data,
                data,
                label=f"({layer_idx}) {layer.__class__.__name__}",
            )
    ax2.set_title("Grads")
    ax3.set_title("Grads 2 Data")
    ax4.set_title("Data")
    ax2.legend()


if __name__ == "__main__":
    main()
