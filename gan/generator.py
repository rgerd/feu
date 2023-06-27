import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from debug import DebuggableSequential
from util import select_device, gan_weights_init

class MNISTGenerator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTGenerator, self).__init__()
        self.debug = debug
        self.main = DebuggableSequential(
            nn.ConvTranspose2d(32, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.ConvTranspose2d(128, 128, 4, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 128, 4, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.ConvTranspose2d(128, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 1, 4),
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
    gen.apply(gan_weights_init)
    gen.load_state_dict(
        torch.load("./saved/generator.pt", map_location=device)["state_dict"]
    )
    gen.eval()

    with torch.no_grad():
        if debug_layers:
            out = gen(torch.randn((1000, 32, 1, 1)).to(device))
            print(out.mean().cpu().data, out.std().cpu().data, out.min().cpu().data, out.max().cpu().data)
        else:
            fig, ax = plt.subplots()
            imgs = []
            for i in range(0, 128):
                feat_vec = torch.randn((1, 32, 1, 1)).to(device)
                g_output = gen(feat_vec).cpu().data[0][0]
                output = g_output
                imgs.append([ax.imshow(output, animated=True)])

            ani = animation.ArtistAnimation(fig, imgs, interval=500, repeat_delay=500, blit=True)
            plt.show()
