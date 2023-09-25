import torch
from torch import Tensor
import torch.nn as nn
from torchvision import datasets, transforms
from debug import DebuggableSequential
from util.layers.squeeze import Squeeze2dLayer
from util.torch_utils import select_device


class MNISTDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTDiscriminator, self).__init__()
        self.classifier_loss = nn.NLLLoss()
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
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
            nn.Conv2d(64, 32, 4, 2, bias=False),
            Squeeze2dLayer(),
            # Output classifier logits for 0-9, plus a discriminator real/fake output
            nn.Linear(32, 11, bias=False),
            print=debug,
            show=debug,
        )

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        full_output = self.main(x)
        return (
            self.log_softmax(full_output[:, :10]).squeeze(),
            full_output[:, 10].squeeze(),
        )

    def calculate_loss(
        self, output: (Tensor, Tensor), target: (Tensor | None, Tensor | None)
    ) -> Tensor:
        class_logits, discrimination = output
        target_classes, target_discrimination = target
        class_loss = (
            torch.zeros((1)).to(class_logits.device)
            if target_classes is None
            else self.classifier_loss(class_logits, target_classes)
        )
        disc_loss = (
            torch.zeros((1)).to(discrimination.device)
            if target_discrimination is None
            else self.discrimination_loss(discrimination, target_discrimination)
        )
        return class_loss + disc_loss

    def calculate_generator_loss(self, output: (Tensor, Tensor)) -> Tensor:
        class_logits, discrimination = output
        real_disc_targets = torch.ones_like(discrimination).to(discrimination.device)
        # Want to maximize the max logit, minimize negative max logit
        class_loss = -torch.mean(torch.max(class_logits, dim=1).values)
        disc_loss = self.discrimination_loss(discrimination, real_disc_targets)
        return class_loss + disc_loss

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
