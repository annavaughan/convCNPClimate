import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

class LandParamLayer(nn.Module):

    def __init__(self, init_ls):
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.init_ls.requires_grad = True
        self.dists = self.get_dists()

    def get_dists(self):
        """
        Get the distances between the center of the patch
        and the other points. This is constant.
        """
        # dimensions
        grid_x, grid_y = np.meshgrid(np.linspace(-2.5, 2.5, 49), 
            np.linspace(-2.5, 2.5, 49))
        dists = np.sqrt(grid_x**2+grid_y**2)
        d = torch.from_numpy(dists)
        return d.cuda()

    def forward(self, wt):
        # Calculate rbf kernel
        kernel = torch.exp(-0.5 * self.dists / self.init_ls ** 2)
        vals = torch.einsum('bij,ij->bij', wt, kernel)
        return torch.sum(vals, (1, 2))

class LandFinalLayer(nn.Module):

    def __init__(self, 
                 init_ls, 
                 n_params):
        super(LandFinalLayer, self).__init__()
        self.param_layers = nn.ModuleList(
            [LandParamLayer(init_ls)
             for _ in range(n_params)]
        )
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

    def forward(self, x):
        pass

class LandGaussianFinalLayer(LandFinalLayer):

    def __init__(self,
                 n_targets,
                 init_ls,
                 n_params):

        super().__init__(init_ls, n_params)

    def forward(self, h):
        mu_h = h[...,0]
        sigma_h = h[..., 1]
        mu = self.param_layers[0](mu_h)
        sigma = self._force_positive(self.param_layers[1](sigma_h))

        return mu, sigma

class LandGammaFinalLayer(LandFinalLayer):

    def __init__(self,  
                 n_targets,
                 init_ls,
                 n_params):

        super().__init__(init_ls, n_params)

    def forward(self, h):

        rho = self.param_layers[0](h[..., 0])
        alpha = self.param_layers[1](h[..., 1])
        beta = self.param_layers[2](h[..., 2])

        rho = self.sigmoid(rho).view(*rho.shape, 1)
        alpha =self._force_positive(alpha).view(*rho.shape)
        beta = self._force_positive(beta).view(*rho.shape)

        # clamp values
        rho = torch.clamp(rho, min = 1e-5, max=1-1e-5)
        alpha = torch.clamp(alpha, min = 1e-5, max=1e5)
        beta = torch.clamp(beta, min = 1e-5, max=1e5)

        return rho, alpha, beta
