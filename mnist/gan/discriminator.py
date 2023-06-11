import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MNISTDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTDiscriminator, self).__init__()
        self.debug = debug
        ndf = 64
        self.main = nn.Sequential(
            # input is ``(nc) x 28 x 28``
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 14 x 14``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 7 x 7``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 3 x 3``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 1 x 1``
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if self.debug:
            fig, ax = plt.subplots(figsize=(2, 2), layout='constrained')
            module_idx = 0
            for module in self.main:
                module_idx += 1
                x = module(x)
                print(module.__class__.__name__, ":", x.shape, torch.mean(x), torch.std(x))
                if module.__class__.__name__ == "LeakyReLU" or module.__class__.__name__ == "Sigmoid":
                    hy, hx = torch.histogram(torch.flatten(x).cpu(), density=True)
                    ax.plot(hx[:-1].detach(), hy.detach(), label=f"{module.__class__.__name__} ({module_idx})")
            ax.legend()
            plt.show()
            return x
        else:
            return self.main(x)
    
if __name__ == '__main__':
    disc = MNISTDiscriminator(debug=True)
    disc.load_state_dict(torch.load("./saved/discriminator.pt")["state_dict"])
    disc.eval()
    with torch.no_grad():
        disc(torch.randn(1000, 1, 28, 28))