import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MNISTGenerator(nn.Module):
    def __init__(self, debug=False):
        super(MNISTGenerator, self).__init__()
        self.debug = debug
        self.main = nn.Sequential(
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
        )

    def layers(self):
        return self.main.children()

    def forward(self, x):
        if self.debug:
            fig, axs = plt.subplots(1, 5, layout="constrained")
            out_idx = 0
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
                if layer.__class__.__name__ == "LeakyReLU" or layer.__class__.__name__ == "Tanh":
                    repr = x.cpu().data.std(dim=0)
                    if repr.shape[0] == 1:
                        repr = repr[0]
                    else:
                        repr = repr.std(dim=0)
                    axs[out_idx].imshow(repr)
                    out_idx += 1
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
