import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MNISTGenerator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTGenerator, self).__init__()
        self.debug = debug
        nz = 32
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.Conv2d( ngf, 1, 5, 1, bias=False),
            nn.BatchNorm2d(1, affine=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
    
    def forward(self, x):
        if self.debug:
            fig, ax = plt.subplots()
            module_idx = 0
            for module in self.main:
                module_idx += 1
                x = module(x)
                print(module.__class__.__name__, ":", x.shape, torch.mean(x), torch.std(x))
                # if isinstance(module, nn.BatchNorm2d):
                    # print(module.weight)
                if module.__class__.__name__ == "ReLU" or module.__class__.__name__ == "Tanh":
                    hy, hx = torch.histogram(torch.flatten(x).cpu(), density=True)
                    ax.plot(hx[:-1].detach(), hy.detach(), label=f"{module.__class__.__name__} ({module_idx})")
            ax.legend()
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

    gen = MNISTGenerator(debug=True).to(device)
    gen.load_state_dict(torch.load("./saved/generator.pt", map_location=device)["state_dict"])
    gen.eval()

    with torch.no_grad():
        feat_vec = torch.randn((1000, 32, 1, 1)).to(device)
        gen(feat_vec)

    gen.debug = False

    fig, ax = plt.subplots()
    imgs = []
    feat_vec = torch.randn((1, 32, 1, 1)).to(device)
    with torch.no_grad():
        for i in range(0, 128):
            g_output = gen(feat_vec).cpu().data[0][0]
            feat_vec = torch.randn((1, 32, 1, 1)).to(device)
            output = g_output
            # output = torch.heaviside(g_output, torch.ones((1)))
            imgs.append([ax.imshow(output, animated=True)])
            if i == 0:
                ax.imshow(output)
            # noise_gain = torch.scalar_tensor(32 if (i % 128) < 60 else 4, dtype=torch.float32)
            # feat_vec += (torch.randn((1, 32, 1, 1)) * noise_gain).to(device)

    ani = animation.ArtistAnimation(fig, imgs, interval=50, repeat_delay=50, blit=True)

    plt.show()