import torch.nn as nn


class Shuffle(nn.Module):
    def __init__(self, permutation, contiguous=True):
        super(Shuffle, self).__init__()
        self.permutation = permutation
        self.contiguous = contiguous

    def forward(self, input):
        shuffled = input.permute(*self.permutation)
        if self.contiguous:
            return shuffled.contiguous()
        else:
            return shuffled
