"""
Module defining convCNP models to downscale temperature and precipitation
with a two layer approach where the first convCNP downscales given weather 
context data and the second refines this prediction based on local landform 
and elevation data. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .mlp import MLP
from .land_final_layers import LandGaussianFinalLayer,LandGammaFinalLayer
from .cnn import CNN, ResConvBlock
from .models import TmaxBiasConvCNP
from .utils import generate_context_mask

class LandConvCNP(nn.Module):
    """
    Second convCNP module for topography context data
    Downscales maximum temperature 
    """
    def __init__(self, 
                 decoder, 
                 n_target,
                 in_channels=1, 
                 ls = 0.1):

        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder= decoder

        self.mlp = MLP(decoder.out_channels, 2,
            hidden_channels = 64,
            hidden_layers = 4)

        self.activation_function = torch.relu
        self.final_layer = LandGaussianFinalLayer(
                 n_target,
                 init_ls = 0.1,
                 n_params = 2)

    def forward(self, x, mask):

        # Encode with set convolution
        x = self.encoder(x, mask)
        x = self.activation_function(x)
        # Decode with CNN
        x = self.decoder(x)
        x = self.activation_function(x)
        # MLP 
        x = self.mlp(x)

        mu, sigma = self.final_layer(x)

        out = torch.cat([mu.view(*mu.shape, 1),
            sigma.view(*sigma.shape, 1)], dim = 1)

        return out

class LandConvCNPPrecip(nn.Module):
    """
    Second convCNP module for topography context data
    downscales precipitation
    """
    def __init__(self, 
                 decoder, 
                 n_target,
                 in_channels=1, 
                 ls = 0.1):

        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder= decoder

        self.mlp = MLP(decoder.out_channels, 3,
            hidden_channels = 64,
            hidden_layers = 4)

        self.activation_function = torch.relu
        self.final_layer = LandGammaFinalLayer(
                 n_target,
                 init_ls = 0.1,
                 n_params = 3)

    def forward(self, x, mask):

        # Encode with set convolution
        x = self.encoder(x, mask)
        x = self.activation_function(x)
        # Decode with CNN
        x = self.decoder(x)
        x = self.activation_function(x)
        # MLP 
        x = self.mlp(x)

        rho, alpha, beta = self.final_layer(x)

        out = torch.cat([rho.view(*rho.shape, 1),
            alpha.view(*alpha.shape, 1), 
            beta.view(*beta.shape, 1)], dim = 1)

        return out

    
class StackConvCNP(nn.Module):
    """
    Double convCNP for two-stage downscaling
    """

    def __init__(self, 
                 weather_convCNP, 
                 land_convCNP):
        super().__init__()
        self.weather = weather_convCNP
        self.land = land_convCNP
        self.x_target = weather_convCNP.x_target

    def forward(self, x_weather, x_land, weather_mask):

        l1_preds = self.weather(x_weather, weather_mask)
        b, t, p = l1_preds.shape
        weather_context = l1_preds.view(b, t, 1, 1, p).repeat(1, 1, 49, 49, 1)
        # New shape: batch, points, lon, lat, param
        # Make new batch
        weather_context = weather_context.view(b*t, 49, 49, 2)
        # Now do x land (shape 86, 21, 21, channels)
        x_land = x_land.view(1, *x_land.shape)
        x_land = x_land.repeat(b, 1, 1, 1, 1)
        # Final reshape
        x_land = x_land.view(b*t, 49, 49, 2)

        # Make full context
        l2_context = torch.cat([weather_context, x_land], dim = 3)

        # Make full context
        l2_context = l2_context.permute(0, 3, 1, 2)
        land_mask = torch.ones(*l2_context.shape).cuda()

        l2_preds = self.land(l2_context, land_mask).view(b, t, 2)

        return l2_preds