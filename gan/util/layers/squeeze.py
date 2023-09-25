from typing import Sequence
import torch.nn as nn
import torch

# Copied from http://pytorch.org/rl/_modules/torchrl/modules/models/utils.html#Squeeze2dLayer

class SqueezeLayer(nn.Module):
    """Squeezing layer.

    Squeezes some given singleton dimensions of an input tensor.

    Args:
         dims (iterable): dimensions to be squeezed
            default: (-1,)

    """

    def __init__(self, dims: Sequence[int] = (-1,)):
        super().__init__()
        for dim in dims:
            if dim >= 0:
                raise RuntimeError("dims must all be < 0")
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for dim in self.dims:
            if input.shape[dim] != 1:
                raise RuntimeError(
                    f"Tried to squeeze an input over dims {self.dims} with shape {input.shape}"
                )
            input = input.squeeze(dim)
        return input


class Squeeze2dLayer(SqueezeLayer):
    """Squeezing layer for convolutional neural networks.

    Squeezes the last two singleton dimensions of an input tensor.

    """

    def __init__(self):
        super().__init__((-2, -1))
