import torch.nn as nn


class DoubleLayer(nn.Module): # its a trap
    """Extracts mean and variance from output"""

    def __init__(self, in_size, out_size):
        super(DoubleLayer, self).__init__()
        self.mean = nn.Linear(in_size, out_size)
        self.var = nn.Linear(in_size, out_size)

    def forward(self, input_tensor):
        return nn.LeakyReLU()(self.mean(input_tensor)), nn.LeakyReLU()(self.var(input_tensor))


class Reshaper(nn.Module):
    """Reshapes input tensor to given shape"""

    def __init__(self, *shape):
        super(Reshaper, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(-1, *self.shape)


