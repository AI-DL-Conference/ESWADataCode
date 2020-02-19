import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer):
        super(Net, self).__init__()
        self.W1 = nn.GRU(n_input, n_hidden, n_layer)
        self.W2 = nn.Linear(n_hidden, 2)

    def forward(self, x):
        o1, c1 = self.W1(x)
        output = o1[-1]
        # output = self.W2(output)
        return output
