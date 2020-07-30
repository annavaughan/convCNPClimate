"""
On-grid to off-grid layer for the precipitation
(Bernoulli-Gamma-Generalised Pareto) model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParamLayer(nn.Module):
    """
    Calculate predicted of a parameter from gridded output
    Parameters:
    -----------
    init_ls: float
        initial length scale for the RBF kernel
    """

    def __init__(self, init_ls):
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.init_ls.requires_grad = True

    def forward(self, wt, dists):
        # Calculate rbf kernel
        kernel = torch.exp(-0.5 * dists / self.init_ls ** 2) 
        vals = torch.einsum('bijk,pij->bpijk', wt, kernel)
        return torch.sum(vals, (2, 3))

class FinalLayer(nn.Module):
    """
    Final layer for converting gridded parameter 
    predictions to off the grid points
    Parameters:
    ----------
    init_ls: Int
        Initial length scale for the RBF kernel
    n_params:
        Total number of parameters in the target 
        distribution
    """

    def __init__(self, 
                 init_ls, 
                 n_params):
        super(FinalLayer, self).__init__()
        self.param_layer = ParamLayer(init_ls)
        self.sigmoid = nn.Sigmoid()

    def _log_exp(self, x):
        """
        Fix overflow
        """
        lt = torch.where(torch.exp(x)<1000)
        if lt[0].shape[0] > 0:
            x[lt] = torch.log(1+torch.exp(x[lt]))
        return x

    def _force_positive(self, x):
        return 0.01+ (1-0.1)*self._log_exp(x)

    def forward(self, x, dists):
        pass

class GammaGPFinalLayer(FinalLayer):
    """
    Final layer for a Bernoulli-Gamma-Generalised Pareto distribution
    """
    def __init__(self,  
                 init_ls,
                 n_params):

        FinalLayer.__init__(self, 
                 init_ls, 
                 n_params)

    def forward(self, h, dists):
        #[rho, xi, alpha, beta, sigma, m, tau]

        params = self.param_layer(h, dists)
        # Do rho, alpha, beta
        params[...,0] = torch.clamp(self.sigmoid(params[...,0]),
                                    min = 1e-5,
                                    max = 1-1e-5)
        params[...,2] = torch.clamp(self.sigmoid(params[...,2]),
                                    min = 1e-5,
                                    max = 1-1e-5)

        # other parameters
        params[...,1] = torch.clamp(self._force_positive(params[...,1]),
                                    min = 1e-5,
                                    max = 1e5)
        params[...,3:] = torch.clamp(self._force_positive(params[...,3:]),
                                    min = 1e-5,
                                    max = 1e5)

        return params

