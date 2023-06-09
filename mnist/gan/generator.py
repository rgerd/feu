import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MNISTGenerator(nn.Module):
    def __init__(self):
        super(MNISTGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(32, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 1, 5, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
    
    def forward(self, x):
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

    gen = MNISTGenerator().to(device)
    gen.load_state_dict(torch.load("./saved/generator.pt", map_location=device)["state_dict"])
    gen.eval()

    fig, ax = plt.subplots()
    imgs = []
    feat_vec = torch.randn((1, 32, 1, 1)).to(device)
    with torch.no_grad():
        for i in range(0, 1024):
            g_output = gen(feat_vec).cpu().data.reshape((28, 28))
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