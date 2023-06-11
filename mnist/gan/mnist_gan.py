from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os.path
import sys

from classifier import MNISTClassifier
from discriminator import MNISTDiscriminator
from generator import MNISTGenerator


class MNISTGanTrainer:
    def train(
        self,
        epoch_count,
        data_loader,
        models,
        optimizers,
        schedulers,
        device,
    ):
        embedding, classifier, discriminator, generator = models
        e_optimizer, c_optimizer, d_optimizer, g_optimizer = optimizers
        e_scheduler, c_scheduler, d_scheduler, g_scheduler = schedulers

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
            loss = F.nll_loss(output, target)
            loss.backward()
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
        c_scheduler.step()

    def _train_gan(self, discriminator, generator, classifier, data_loader):
        timer_tick = time.time()
        bce_loss = nn.BCELoss()
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            (batch_size, _, _, _) = data.shape
            rand_g_inputs_d = torch.randn((batch_size, 32, 1, 1)).to(device)
            # rand_g_inputs_c = torch.randn((batch_size, 32, 1, 1)).to(device)
            d_real_targets = torch.ones((batch_size,)).to(device)
            d_fake_targets = torch.zeros((batch_size,)).to(device)

            discriminator.train()
            generator.eval()

            discriminator.zero_grad()
            g_output = generator(rand_g_inputs_d)
            d_output_reals = discriminator(data)[:, 0, 0, 0]
            d_output_fakes = discriminator(g_output.detach())[:, 0, 0, 0]
            loss_real = bce_loss(d_output_reals, d_real_targets)
            loss_real.backward()
            loss_fake = bce_loss(d_output_fakes, d_fake_targets)
            loss_fake.backward()
            d_optimizer.step()

            discriminator.eval()
            classifier.eval()
            generator.train()

            # c_output = classifier(generator(rand_g_inputs_c))

            generator.zero_grad()
            d_output_fakes_g = discriminator(g_output)[:, 0, 0, 0]
            g_loss = bce_loss(d_output_fakes_g, d_real_targets)
            # g_loss = F.nll_loss(c_output, torch.zeros_like(target))
            # g_loss = -torch.mean(torch.max(c_output, dim=1).values) * 0.1
            g_loss.backward()
            g_optimizer.step()

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
                print(f"G: [Loss: {g_loss.item()}]")
                timer_tick = time.time()

        d_scheduler.step()
        g_scheduler.step()

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
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MNIST Gan", description="Trains a GAN to generate MNIST numbers"
    )
    parser.add_argument("--nosave", action="store_true", required=False, default=False)
    parser.add_argument("--noload", action="store_true", required=False, default=False)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    # torch.manual_seed(1)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    num_epochs = 32

    train_kwargs = {"batch_size": 128}
    test_kwargs = {"batch_size": 1000}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    elif use_mps:
        mps_kwargs = {"pin_memory": True, "shuffle": True}
        train_kwargs.update(mps_kwargs)
        test_kwargs.update(mps_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

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
    d_optimizer = optim.SGD(discriminator.parameters(), lr=2e-4)
    if not args.noload and os.path.exists("./saved/discriminator.pt"):
        print("Loading discriminator...")
        d_saved = torch.load("./saved/discriminator.pt", map_location=device)
        discriminator.load_state_dict(d_saved["state_dict"])
    else:
        discriminator.apply(gan_weights_init)

    generator = MNISTGenerator().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    if not args.noload and os.path.exists("./saved/generator.pt"):
        print("Loading generator...")
        g_saved = torch.load("./saved/generator.pt", map_location=device)
        generator.load_state_dict(g_saved["state_dict"])
    else:
        generator.apply(gan_weights_init)

    e_scheduler = StepLR(e_optimizer, step_size=1, gamma=0.7)
    c_scheduler = StepLR(c_optimizer, step_size=1, gamma=0.7)
    d_scheduler = StepLR(d_optimizer, step_size=1, gamma=0.7)
    g_scheduler = StepLR(g_optimizer, step_size=1, gamma=0.7)

    try:
        MNISTGanTrainer().train(
            num_epochs,
            train_loader,
            [embedding, classifier, discriminator, generator],
            [e_optimizer, c_optimizer, d_optimizer, g_optimizer],
            [e_scheduler, c_scheduler, d_scheduler, g_scheduler],
            device,
        )
    except KeyboardInterrupt:
        print("Done.")

    # for epoch in range(1, num_epochs + 1):
    #     print(f"Epoch {epoch} / {num_epochs}")

    #     generator_file_name = (
    #         f"generator-{epoch}-{datetime.now().strftime('%H%M%S')}.pt"
    #     )
    #     print(f"Saving {generator_file_name}")
    #     torch.save(
    #         generator.state_dict(),
    #         f"./saved/{generator_file_name}",
    #     )

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
        rand_g_inputs = torch.randn((test_kwargs["batch_size"], 32, 1, 1)).to(device)
        g_output = generator(rand_g_inputs)
        d_output = discriminator(g_output)
        d_correct_count = torch.sum(torch.isclose(d_output, torch.zeros_like(d_output)))
        print(f"{torch.mean(d_output)}, {torch.var(d_output)}")
        print(f"Discriminator: {d_correct_count}")

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
