"""
ConvCNP models for downscaling temperature, precipitation (Bernoulli-
Gamma distribution) and precipitation (Bernoulli-Gamma-Generalised Pareto
distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .mlp import MLP
from .final_layers import GaussianFinalLayer, GammaFinalLayer, GammaGPFinalLayer
from .cnn import CNN, ResConvBlock

class TmaxBiasConvCNP(nn.Module):
    """
    Downscaling for temperature 
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
                 in_channels=1, 
                 ls = 0.1):
        
        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder= decoder

        self.mlp = MLP(decoder.out_channels, 2,
            hidden_channels = 64,
            hidden_layers = 4)

        self.activation_function = torch.relu
        self.final_layer = GaussianFinalLayer(0.1, 2)
 

    def forward(self, x, mask, dists):

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

        return out

class GammaBiasConvCNP(nn.Module):
    """
    Bias correction for precipitation (Bernoulli-
    Gamma) 
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
            out_channels = 3,
            hidden_channels = 64,
            hidden_layers = 4)
        self.decoder = decoder
        self.out_layer = GammaFinalLayer(
            init_ls = ls,
            n_params = 3
        )

    def forward(self, h, mask, dists):

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
                
        return out

class GammaGPBiasConvCNP(nn.Module):
    """
    Bias correction for precipitation (Bernoulli-
    Gamma-Generalised Pareto) 
    Parameters:
    ----------
    decoder: convolutional architecture
    in_channels: Int
        Total number of context variables
    ls: float
        Initial length scale for the RBF kernel
    """

    def __init__(self, 
                 x_context,
                 x_target,
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
            target_x = x_target,
            grid_x = x_context[:, :, 0],
            grid_y = x_context[:, :, 1],
            init_ls = ls,
            n_params = 7
        )

    def forward(self, h, mask):

        # Encode with set convolution
        h = self.activation(self.encoder(h, mask))
        # Decode with CNN
        h = self.activation(self.decoder(h))
        # MLP 
        h = self.mlp(h)
        # out layer
        params = self.out_layer(h)
        
        return params