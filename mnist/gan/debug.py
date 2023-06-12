import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def layer_is_activation(layer):
    return layer.__class__.__name__ in ["LeakyReLU", "ReLU", "Sigmoid", "Tanh"]

class DebuggableSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(DebuggableSequential, self).__init__(*args)
        self.print = kwargs['print'] if 'print' in kwargs else False
        self.show = kwargs['show'] if 'show' in kwargs else False
    
    def layers(self):
        return self.children()

    def forward(self, x):
        if self.print or self.show:
            return self.print_forward(x)
        else:
            return super().forward(x)
    
    def print_forward(self, x):
        if self.show:
            fig, axs = plt.subplots(1, sum([1 if layer_is_activation(layer) else 0 for layer in self.layers()]), layout="constrained")
        print(
            torch.mean(x).cpu().data,
            "\t",
            torch.std(x).cpu().data,
            "\t",
            x.shape,
        )
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
                    print("\t (W):", p.data.mean().cpu().data, p.data.std().cpu().data)
            if self.show:
                if layer_is_activation(layer):
                    repr = x.cpu().data.std(dim=0)
                    if repr.shape[0] == 1:
                        repr = repr[0]
                    else:
                        repr = repr.std(dim=0)
                    axs[out_idx].imshow(repr)
                    out_idx += 1
        print(x.mean())
        if self.show:
            plt.show()
        return x
    
    def print_grads(self):
        for layer_idx, layer in enumerate(self.layers()):
            for n, p in layer.named_parameters():
                if n in ["weight"] and p.requires_grad:
                    print(f"({layer_idx}) {layer.__class__.__name__}: {torch.mean(torch.abs(p.grad)).cpu().data} {torch.std(p.grad).cpu().data} {(p.grad.std() / p.std()).cpu().data}")
                    # self.layer_data[layer_idx].append(torch.std(p.data).cpu().data)
                    # self.layer_grads_to_data[layer_idx].append(
                    #     (torch.std(p.grad) / torch.std(p.data) + 1e-10)
                    #     .log10()
                    #     .cpu()
                    #     .data
                    # )

# class StatisticsLogger():
#     def __init__():
