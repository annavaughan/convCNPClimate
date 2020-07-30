"""
Functions to calculate NLL for various distributions
"""
import numpy as np 
import torch 
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from .utils import make_r_mask, log_exp

def gll(target_vals, v):
    """
    Calculate mean Gaussian log likelihood over the batch
    """
    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 2)
    
    # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]
    
    dist = Normal(loc=v[:,0], scale=v[:,1])
    logp = dist.log_prob(target_vals).view(-1)
    return torch.mean(logp)

def gamma_ll(target_vals, v):
    """
    Evaluate gamma-bernoulli mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch,86,channels)
        parameters from model [rho, alpha, beta]
    target_vals: torch.Tensor(batch,86)
        target vals to eval at
    """

    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 3)
    
    # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]

    # Make r mask
    r, target_vals = make_r_mask(target_vals)

    gamma = Gamma(concentration = v[:,1], rate = v[:,2])
    logp = gamma.log_prob(target_vals)

    total = r*(torch.log(v[:,0])+logp)+(1-r)*torch.log(1-v[:,0])
    
    return torch.mean(total)


def tbi_func(x, v):
    """
    Evaluate gamma-GP-Bernoulli mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch*86, channels)
        parameters from model
    x: torch.Tensor(batch*86)
        target vals to eval at
    """
    # Gamma distribution
    g = Gamma(concentration = v[:,2], rate = v[:,3])
    gamma = torch.exp(torch.clamp(g.log_prob(x), min=-1e5, max=1e5))

    # Weight term
    weight_term = (1/2)+(1/np.pi)*torch.atan((x-v[:,5])/v[:,6])

    # GP distribution
    gp = (1/v[:,4])*(1+(v[:,1]*x/v[:,4]))**((-1/v[:,1])-1)

    # total
    tbi = gamma*(1-weight_term)+gp*weight_term
    return torch.clamp(tbi, min = 1e-5)


def gamma_gp_ll(target_vals, v):
    """
    Calculate mixture distribution log likelihood over the batch
    Parameters:
    -----------
    target_vals: torch.Tensor(16, 86)
        True precipitation values
    v: torch.Tensor(16, 86, 7)
        Output [rho, xi, alpha, beta, sigma, m, tau]
    """
    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 7)

    # Drop nans
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]

    r, target_vals = make_r_mask(target_vals)

    # Get normalisation coefficients
    x = torch.linspace(0.2, 150, 1499)
    x = x.view(-1, 1).repeat(1, v.shape[0]).cuda()
    
    vals = tbi_func(x, v)
    norms = torch.sum(vals, dim = 0)

    # Calculate log likelihood
    total = r*(torch.log(v[:,0])+(1/norms)*tbi_func(target_vals, v))+(1-r)*torch.log(1-v[:,0])

    return torch.mean(total)

