"""
Downscaling convCNPs including MLP for elevation data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .mlp import MLP
from .final_layers import GaussianFinalLayer, GammaFinalLayer, GammaGPFinalLayer
from .cnn import CNN, ResConvBlock
from .utils import force_positive


class TmaxBiasConvCNPElev(nn.Module):
    """
    Bias correction for temperature, including MLP for elevation
    Parameters:
    ----------
    decoder: convolutional architecture
    """
    
    def __init__(self, decoder, in_channels=1, ls = 0.1):
        
        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder= decoder

        self.mlp = MLP(decoder.out_channels, 2,
            hidden_channels = 64,
            hidden_layers = 4)

        self.elev_mlp = MLP(5, 2,
            hidden_channels = 64,
            hidden_layers = 4)

        self.activation_function = torch.relu
        self.final_layer = GaussianFinalLayer(0.1, 2)
 

    def forward(self, x, mask, dists, elev):

        # Encode with set convolution
        x = self.encoder(x, mask)
        x = self.activation_function(x)
        # Decode with CNN
        x = self.decoder(x)
        x = self.activation_function(x)
        # MLP 
        x = self.mlp(x)

        mu, sigma = self.final_layer(x, dists)

        out = torch.cat([mu.view(*mu.shape, 1),
            sigma.view(*sigma.shape, 1)], dim = 2)

        # Do elevation
        elev = elev.repeat(out.shape[0], 1, 1)
        out = torch.cat([out, elev], dim = 2)
        out = self.elev_mlp(out)
        out[...,1] = force_positive(out[...,1])

        return out

class GammaBiasConvCNPElev(nn.Module):
    """
    Bias correction for precipitation, including MLP for elevation
    Parameters:
    ----------
    decoder: convolutional architecture
    """

    def __init__(self, 
                 decoder,
                 in_channels = 1, 
                 ls = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()

        self.encoder = Encoder(in_channels = in_channels)
        self.mlp = MLP(in_channels = 128,
            out_channels = 3,
            hidden_channels = 64,
            hidden_layers = 4)
        self.decoder = decoder
        self.out_layer = GammaFinalLayer(
            init_ls = ls,
            n_params = 3
        )

        self.elev_mlp = MLP(6,
            out_channels = 3,
            hidden_channels = 64,
            hidden_layers = 4)

    def forward(self, h, mask, dists, elev):

        # Encode with set convolution
        h = self.activation(self.encoder(h, mask))
        # Decode with CNN
        h = self.activation(self.decoder(h))
        # MLP 
        h = self.mlp(h)
        # out layer
        rho, alpha, beta = self.out_layer(h, dists)
        out = torch.cat([rho.view(*rho.shape, 1),
            alpha.view(*alpha.shape, 1),
            beta.view(*beta.shape, 1)], dim = 2)

        # Do elevation
        elev = elev.repeat(out.shape[0], 1, 1)
        out = torch.cat([out[...,0], elev], dim = 2)
        out = self.elev_mlp(out)
        out[...,0] = self.sigmoid(out[...,0])
        out[...,1:] = force_positive(out[...,1:])
        
        return out
