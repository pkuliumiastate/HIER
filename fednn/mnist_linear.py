import torch.nn as nn

class mnist_linear(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(mnist_linear, self).__init__()
        self.fc2 = nn.Linear(input_channels * 28 * 28, output_channels)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc2(x)
        return  x