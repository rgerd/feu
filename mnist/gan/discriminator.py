import torch
import torch.nn as nn

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv2d(256, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 1, 4, 2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)