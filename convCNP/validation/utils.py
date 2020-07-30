from collections import OrderedDict
import torch
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from convCNP.models.elev_models import TmaxBiasConvCNPElev, GammaBiasConvCNPElev
from convCNP.models.cnn import CNN, ResConvBlock, UNet

def get_dists(target_x, grid_x, grid_y):
    """
    Get the distances between the grid points and true points
    """
    # dimensions
    x_dim, y_dim = grid_x.shape

    # Init large scale grid
    total_grid = torch.zeros(target_x.shape[0], x_dim, y_dim)
    count = 0

    for point in target_x:
        # Calculate distance from point to each grid
        dists = (grid_x - point[0])**2+(grid_y - point[1])**2
        total_grid[count, :, :] = dists

        count += 1

    return total_grid

def load_model_precip(PATH, N_CHANNELS, x_context, x_target, device):
    """
    Load a pretrained model
    """
    print("ok")
    decoder = CNN(n_channels = 128, 
        ConvBlock=ResConvBlock,n_blocks=6,
        Conv=nn.Conv2d,
        Normalization=nn.Identity,
        kernel_size=5)
    model = GammaBiasConvCNPElev(
        decoder, 
        in_channels=25, 
        ls = 0.1)
    model.to(device)
    checkpoint = torch.load(PATH, map_location=device)
    weights = OrderedDict([(k[7:], v) for k, v in checkpoint['model_state_dict'].items()])
    model.load_state_dict(weights)
  
    return model

def load_model(PATH, N_CHANNELS, x_context, x_target, device):
    """
    Load a pretrained model
    """
    decoder = CNN(n_channels = 128, 
        ConvBlock=ResConvBlock,n_blocks=6,
        Conv=nn.Conv2d,
        Normalization=nn.Identity,
        kernel_size=5)
    model = TmaxBiasConvCNPElev(
        decoder, 
        in_channels=25, 
        ls = 0.1)
    model.to(device)
    checkpoint = torch.load(PATH, map_location=device)
    weights = OrderedDict([(k[7:], v) for k, v in checkpoint['model_state_dict'].items()])
    model.load_state_dict(weights)
  
    return model

def get_output(model, context, dists_to_target, elev_at_target):
    """
    Use model to make predictions on held out dataset
    """
    
    # Predict
    with torch.no_grad():
        batch_size, channels, x, y = context.shape
        mask = generate_context_mask(batch_size, channels, x, y)
        out = model(context, mask, dists_to_target, elev_at_target)
    
    return out

def generate_context_mask(batch_size, n_channels, x, y):
    """
    Generate a context mask - in this simple case this will be one 
    for all grid points
    """
    return torch.ones(batch_size, n_channels, x, y)