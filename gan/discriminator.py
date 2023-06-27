import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms
from debug import DebuggableSequential
from util import select_device

class MNISTDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTDiscriminator, self).__init__()
        self.main = DebuggableSequential(
            nn.Conv2d(1, 256, 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(256, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, 4, 2, bias=False),

            print=debug,
            show=debug
        )

    def forward(self, x):
        return self.main(x)

    def print_grads(self):
        self.main.print_grads()

if __name__ == "__main__":
    device = select_device()

    debug_layers = True
    disc = MNISTDiscriminator(debug_layers).to(device)
    disc.load_state_dict(
        torch.load("./saved/discriminator.pt", map_location=device)["state_dict"]
    )
    disc.eval()

    train_kwargs = {"batch_size": 1000, "shuffle": True}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    datasetTrain = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    trainTransformed = []
    for data, target in datasetTrain:
        trainTransformed.append((data, target))
    train_loader = torch.utils.data.DataLoader(trainTransformed, **train_kwargs)

    for data, _ in train_loader:
        print(torch.min(data), torch.max(data), torch.mean(data), torch.std(data))
        data = data.to(device)
        disc(data)
        break
