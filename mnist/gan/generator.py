import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from debug import DebuggableSequential
from util import select_device

class MNISTGenerator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTGenerator, self).__init__()
        self.debug = debug
        self.main = DebuggableSequential(
            nn.ConvTranspose2d(32, 128, 4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 128, 4, 2, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, 4),
            nn.Tanh(),
            print=debug,
            show=debug
        )

    def forward(self, x):
        return self.main(x)
    
    def print_grads(self):
        self.main.print_grads()

if __name__ == "__main__":
    device = select_device()

    debug_layers = False
    gen = MNISTGenerator(debug_layers).to(device)
    gen.load_state_dict(
        torch.load("./saved/generator.pt", map_location=device)["state_dict"]
    )
    gen.eval()

    if debug_layers:
        out = gen(torch.randn((1000, 32, 1, 1)).to(device))
        print(out.mean().cpu().data, out.std().cpu().data, out.min().cpu().data, out.max().cpu().data)
        exit()

    fig, ax = plt.subplots()
    imgs = []
    with torch.no_grad():
        for i in range(0, 128):
            feat_vec = torch.randn((1, 32, 1, 1)).to(device)
            g_output = gen(feat_vec).cpu().data[0][0]
            output = g_output
            # output = torch.heaviside(g_output, torch.ones((1)))
            imgs.append([ax.imshow(output, animated=True)])
            # if i == 0:
            #     ax.imshow(output)
            # noise_gain = torch.scalar_tensor(32 if (i % 128) < 60 else 4, dtype=torch.float32)
            # feat_vec += (torch.randn((1, 32, 1, 1)) * noise_gain).to(device)

    ani = animation.ArtistAnimation(fig, imgs, interval=50, repeat_delay=50, blit=True)

    plt.show()
