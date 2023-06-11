import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms

from rescale import Rescale


class MNISTDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTDiscriminator, self).__init__()
        self.debug = debug
        self.main = nn.Sequential(
            nn.Conv2d(1, 256, 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv2d(256, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 2, bias=False),
            nn.BatchNorm2d(1, affine=False),
            nn.Sigmoid(),
        )

    def layers(self):
        return self.main.children()

    def forward(self, x):
        if self.debug:
            fig, axs = plt.subplots(1, 4, layout="constrained")
            out_idx = 0
            print(
                torch.mean(x).cpu().data,
                "\t",
                torch.std(x).cpu().data,
                "\t",
                x.shape,
            )
            for layer in self.layers():
                x = layer(x)
                print(
                    layer.__class__.__name__,
                    " " * (32 - len(layer.__class__.__name__)),
                    torch.mean(x).cpu().data,
                    "\t",
                    torch.std(x).cpu().data,
                    "\t",
                    x.shape,
                )
                for n, p in layer.named_parameters():
                    if n in ["weight"] and p.requires_grad:
                        print("\t", p.data.mean().cpu().data, p.data.std().cpu().data)
                if (
                    layer.__class__.__name__ == "LeakyReLU"
                    or layer.__class__.__name__ == "Sigmoid"
                ):
                    repr = x.cpu().data.std(dim=0)
                    if repr.shape[0] == 1:
                        repr = repr[0]
                    else:
                        repr = repr.std(dim=0)
                    axs[out_idx].imshow(repr)
                    out_idx += 1
            print(x.mean())
            plt.show()
            return x
        else:
            return self.main(x)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    debug_layers = True
    disc = MNISTDiscriminator(debug_layers).to(device)
    disc.load_state_dict(
        torch.load("./saved/discriminator.pt", map_location=device)["state_dict"]
    )
    disc.eval()

    train_kwargs = {"batch_size": 100000, "shuffle": True}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (1.0,)) #(0.3081,)),
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
        # data = data.to(device)
        # disc(data)
        break
