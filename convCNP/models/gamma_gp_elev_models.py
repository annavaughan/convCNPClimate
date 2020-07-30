"""
Module incorporating elevation correction into the 
precipitation (Bernoulli-Gamma-Generalised Pareto)
model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .mlp import MLP
from .gamma_gp_final_layers import GammaGPFinalLayer
from .cnn import CNN, ResConvBlock
from .utils import force_positive

class GammaGPBiasConvCNPElev(nn.Module):
    """
    Model to downscale precipitation 
    (Bernoulli-Gamma-Generalised Pareto)
    with elevation correction

    Parameters:
    ----------
    decoder: convolutional architecture
    in_channels: Int
        Total number of context variables
    ls: float
        Initial length scale for the RBF kernel
    """

    def __init__(self, 
                 decoder,
                 in_channels = 1, 
                 ls = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.activation = torch.relu

        self.encoder = Encoder(in_channels = in_channels)
        self.mlp = MLP(in_channels = 128,
            out_channels = 7,
            hidden_channels = 64,
            hidden_layers = 4)
        self.decoder = decoder
        self.out_layer = GammaGPFinalLayer(
            init_ls = ls,
            n_params = 7
        )

        self.elev_mlp = MLP(in_channels = 10,
            out_channels = 7,
            hidden_channels = 64,
            hidden_layers = 4)
        self.sigmoid = nn.Sigmoid()
        
    def make_positive(self, params):
        """
        Fix overflow errors in log likelihood calculation
        """
        params[...,0] = torch.clamp(self.sigmoid(params[...,0]),
                                    min = 1e-5,
                                    max = 1-1e-5)
        params[...,2] = torch.clamp(self.sigmoid(params[...,2]),
                                    min = 1e-5,
                                    max = 1-1e-5)

        # other parameters
        params[...,1] = torch.clamp(force_positive(params[...,1]),
                                    min = 1e-5,
                                    max = 1e5)
        params[...,3:] = torch.clamp(force_positive(params[...,3:]),
                                    min = 1e-5,
                                    max = 1e5)
        return params

    def forward(self, h, mask, dists, elev):
        
        # Encode with set convolution
        h = self.activation(self.encoder(h, mask))

        # Decode with CNN
        h = self.activation(self.decoder(h))

        # MLP 
        h = self.mlp(h)

        # out layer
        h = self.out_layer(h, dists)
        
        # elevation
        elev = elev.repeat(h.shape[0], 1, 1)
        h = torch.cat([h, elev], dim = 2)
        params = self.elev_mlp(h)
        params = self.make_positive(params)

        return params