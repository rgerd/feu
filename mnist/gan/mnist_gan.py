from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import time
import os.path
import sys


from classifier import MNISTClassifier
from discriminator import MNISTDiscriminator
from generator import MNISTGenerator
from util import select_device

class MNISTGanTrainer:
    def train(
        self,
        epoch_count,
        data_loader,
        models,
        optimizers,
        # schedulers,
        device,
    ):
        embedding, classifier, discriminator, generator = models
        e_optimizer, c_optimizer, d_optimizer, g_optimizer = optimizers
        # e_scheduler, c_scheduler, d_scheduler, g_scheduler = schedulers

        self.class_loss = nn.NLLLoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

        self.layer_grads = []
        self.layer_grads_to_data = []
        self.layer_data = []
        # for _ in generator.layers():
        #     self.layer_grads.append([])
        #     self.layer_grads_to_data.append([])
        #     self.layer_data.append([])
        self.gen_sat = []

        for epoch in range(1, epoch_count + 1):
            print(f"Epoch {epoch}...")
            # self._train_classifier(classifier, data_loader)
            self._train_gan(discriminator, generator, classifier, data_loader)

    def _train_classifier(self, classifier, data_loader):
        classifier.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = classifier(data)

            c_optimizer.zero_grad()
            loss = self.class_loss(output, target)
            loss.backward()
            # classifier.print_grads()
            c_optimizer.step()

            if batch_idx % 128 == 0:
                print(
                    "C [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        # c_scheduler.step()

    def _train_gan(self, discriminator, generator: MNISTGenerator, classifier, data_loader):
        timer_tick = time.time()

        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)

            discriminator.zero_grad()
            loss_real, loss_fake = self._loss_discriminator(discriminator, generator, data)
            loss_real.backward()
            loss_fake.backward()
            d_optimizer.step()

            '''
            discriminator.zero_grad()
            generator.zero_grad()
            loss_g = self._loss_generator(discriminator, generator)
            loss_g.backward()
            # generator.print_grads()
            g_optimizer.step()
            '''

            # for layer_idx, layer in enumerate(generator.layers()):
            #     for n, p in layer.named_parameters():
            #         if n in ["weight"] and p.requires_grad:
            #             self.layer_grads[layer_idx].append(
            #                 torch.mean(torch.abs(p.grad)).cpu().data
            #             )
            #             self.layer_data[layer_idx].append(torch.std(p.data).cpu().data)
            #             self.layer_grads_to_data[layer_idx].append(
            #                 (torch.std(p.grad) / torch.std(p.data) + 1e-10)
            #                 .log10()
            #                 .cpu()
            #                 .data
            #             )
            # self.gen_sat.append(
            #     (torch.sum(torch.abs(g_output) > 0.9) / g_output.nelement()).cpu().data
            # )

            if batch_idx % 128 == 0:
                print(
                    "[{}/{} ({:.0f}%)] ({} seconds)".format(
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        time.time() - timer_tick,
                    )
                )
                print(
                    f"D: [Loss_Real: {loss_real.item()}], [Loss_Fake: {loss_fake.item()}]"
                )
                # print(f"G: [Loss: {loss_g.item()}]")
                # print(f"G: [Saturation: {self.gen_sat[-1] * 100.0}]")
                timer_tick = time.time()
        # d_scheduler.step()
        # g_scheduler.step()

    def _loss_discriminator(self, discriminator, generator, data):
        (B, C, H, W) = data.shape
        rand_g_inputs_d = torch.randn((B, 32, 1, 1)).to(device)
        d_real_targets = torch.ones((B, 1, 1, 1)).to(device)
        d_fake_targets = torch.zeros((B, 1, 1, 1)).to(device)

        discriminator.train()
        generator.eval()
        # fake_data = generator(rand_g_inputs_d)
        fake_data = torch.randn((B, 1, 28, 28)).to(device)
        d_output_reals = discriminator(data)
        d_output_fakes = discriminator(fake_data.detach())
        loss_real = self.disc_loss(d_output_reals, d_real_targets)
        loss_fake = self.disc_loss(d_output_fakes, d_fake_targets)
        return loss_real, loss_fake

    def _loss_generator(self, discriminator, generator):
            rand_g_inputs = torch.randn((128, 32, 1, 1)).to(device)
            d_real_targets = torch.ones((128, 1, 1, 1)).to(device)

            discriminator.eval()
            classifier.eval()
            generator.train()

            # c_output = classifier(generator(rand_g_inputs_c))

            # print(torch.mean(data.detach()), torch.mean(g_output.detach()), torch.var(data.detach()), torch.var(g_output.detach()))
            g_output = generator(rand_g_inputs)
            d_output = discriminator(g_output)
            loss_g = self.disc_loss(d_output, d_real_targets)
            # g_loss = F.nll_loss(c_output, torch.zeros_like(target))
            # g_loss = -torch.mean(torch.max(c_output, dim=1).values) * 0.1    

            self.gen_sat.append(
                (torch.sum(torch.abs(g_output) > 0.97) / g_output.nelement()).cpu().data
            )
            return loss_g

    def _train_emb(embedding, generator, classifier):
        generator.eval()
        classifier.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            (B,) = target.shape
            target_oh = (
                F.one_hot(target, num_classes=10)
                .to(torch.float32)
                .reshape((B, 10, 1, 1))
            )
            output = classifier(generator(target_oh))

            g_optimizer.zero_grad()
            loss = F.nll_loss(output, target)
            loss.backward()
            g_optimizer.step()

            if batch_idx % 64 == 0:
                # print(f"Output: {output[0]}")
                # print(f"Target: {target[0]}")
                print(
                    "G [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def gan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.001)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.001)


if __name__ == "__main__":
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

    num_epochs = 16

    train_kwargs = {"batch_size": 128}
    test_kwargs = {"batch_size": 1000}
    if torch.cuda.is_available():
        cuda_kwargs = { "shuffle": True, "num_workers": 1, "pin_memory": True }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    elif torch.backends.mps.is_available():
        mps_kwargs = {"pin_memory": True, "shuffle": True}
        train_kwargs.update(mps_kwargs)
        test_kwargs.update(mps_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (1.0,)), # (0.3081,)),
        ]
    )
    datasetTrain = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    trainTransformed = []
    for data, target in datasetTrain:
        trainTransformed.append((data, target))
    datasetTest = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainTransformed, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(datasetTest, **test_kwargs)

    print(
        f"Train size: {len(datasetTrain)} x {train_kwargs['batch_size']} = {len(datasetTrain) * train_kwargs['batch_size']}"
    )

    embedding = torch.nn.Embedding(10, 32)
    e_optimizer = optim.Adam(embedding.parameters(), lr=1e-1)

    classifier = MNISTClassifier().to(device)
    c_optimizer = optim.Adadelta(classifier.parameters(), lr=1.0)
    c_saved_acc = 0.0
    if not args.noload and os.path.exists("./saved/classifier.pt"):
        print("Loading classifier...")
        c_saved = torch.load("./saved/classifier.pt", map_location=device)
        c_saved_acc = c_saved["acc"]
        classifier.load_state_dict(c_saved["state_dict"])

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

    # e_scheduler = StepLR(e_optimizer, step_size=1, gamma=0.7)
    # c_scheduler = StepLR(c_optimizer, step_size=1, gamma=0.7)
    # d_scheduler = StepLR(d_optimizer, step_size=1, gamma=0.7)
    # g_scheduler = StepLR(g_optimizer, step_size=1, gamma=0.7)

    trainer = MNISTGanTrainer()
    try:
        trainer.train(
            num_epochs,
            train_loader,
            [embedding, classifier, discriminator, generator],
            [e_optimizer, c_optimizer, d_optimizer, g_optimizer],
            # [e_scheduler, c_scheduler, d_scheduler, g_scheduler],
            device,
        )
    except KeyboardInterrupt:
        print("Done.")

    if args.display:
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
        plt.show()

    embedding.eval()
    classifier.eval()
    generator.eval()
    discriminator.eval()

    # Evaluate
    with torch.no_grad():
        c_correct_count = 0.0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            selections = torch.argmax(output, dim=1)
            c_correct_count += torch.sum(selections == target)
        target_count = len(test_loader) * test_kwargs["batch_size"]
        c_acc = c_correct_count / target_count
        print(f"Classifier: {c_correct_count} / {target_count} = {(c_acc * 100):.2f}%")

        if (
            not args.nosave
            and c_correct_count / target_count > 0.98
            and c_acc > c_saved_acc
        ):
            print("Saving classifier...")
            torch.save(
                {"acc": c_acc, "state_dict": classifier.state_dict()},
                "./saved/classifier.pt",
            )

    with torch.no_grad():
        rand_g_inputs = torch.randn((1000, 32, 1, 1)).to(device)
        g_output = generator(rand_g_inputs)
        d_output = discriminator(g_output)
        d_accuracy = torch.mean(1.0 - F.sigmoid(d_output))
        print(f"Discriminator: {d_accuracy}")

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
