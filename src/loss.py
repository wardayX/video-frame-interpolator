import torch.nn as nn

class EPE(nn.Module):
    def __init__(self, a=None):
        super(EPE, self).__init__()
    def forward(self, f, g):
        return 0

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
    def forward(self, i, g):
        return 0

class SOBEL(Sobel):
    pass