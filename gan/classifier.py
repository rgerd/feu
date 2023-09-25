from __future__ import print_function
import torch
from torch import Tensor
import torch.nn as nn
from debug import DebuggableSequential


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.main = DebuggableSequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.BatchNorm1d(9216),
            nn.Linear(9216, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)

    def print_grads(self):
        self.main.print_grads()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    classifier = MNISTClassifier()
    classifier.load_state_dict(torch.load("./saved/classifier.pt")["state_dict"])
    classifier.eval()

    examples_to_show = 5
    with torch.no_grad():
        for data, target in train_loader:
            print(torch.argmax(classifier(data)), target)
            examples_to_show -= 1
            if examples_to_show == 0:
                break
