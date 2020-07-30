import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP for the elevation data and raw CNN output
    Parameters:
    -----------
    in_channels: Int
        Number of input channels
    out_channels: Int
        Number of output channels
    hidden_channels: Int
        Number of hidden nodes
    hidden_layers: Int
        Number of hidden layers
    """

    def __init__(self, in_channels, 
                out_channels, 
                hidden_channels,
                hidden_layers):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = hidden_layers
        self.relu = nn.ReLU()

        self.in_to_hidden = nn.Linear(self.in_channels, self.hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels) 
            for _ in range(hidden_layers)]
            )
        self.hidden_to_out = nn.Linear(self.hidden_channels, self.out_channels) 

    def forward(self, h):
        # in -> hidden
        h = self.relu(self.in_to_hidden(h))
        # hidden
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
        # hidden -> out
        h = self.hidden_to_out(h)

        return h