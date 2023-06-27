import torch
import torch.nn as nn

class Rescale(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.range = max - min

    def forward(self, img):
        img_min, img_max = torch.min(img), torch.max(img)
        img_range = img_max - img_min
        rescaled = ((img - img_min) / img_range) * self.range + self.min
        return rescaled