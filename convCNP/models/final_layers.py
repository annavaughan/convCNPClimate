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
        vals = torch.einsum('bij,pij->bpij', wt, kernel)
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
        self.param_layers = nn.ModuleList(
            [ParamLayer(init_ls)
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
        """
        Make values greater than zero
        """
        return 0.01+ (1-0.1)*self._log_exp(x)

    def forward(self, x, dists):
        pass

class GaussianFinalLayer(FinalLayer):
    """
    On-grid -> off-grid layer for Gaussian distribution
    """

    def __init__(self,
                 init_ls,
                 n_params):

        FinalLayer.__init__(self, 
                 init_ls, 
                 n_params)

    def forward(self, h, dists):
        mu_h = h[...,0]
        sigma_h = h[..., 1]

        mu = self.param_layers[0](mu_h, dists)
        sigma = self._force_positive(self.param_layers[1](sigma_h, dists))

        return mu, sigma

class GammaFinalLayer(FinalLayer):
    """
    On-grid -> off-grid layer for Bernoulli-Gamma 
    mixture distribution
    """

    def __init__(self,  
                 init_ls,
                 n_params):

        FinalLayer.__init__(self, 
                 init_ls, 
                 n_params)

    def forward(self, h, dists):

        rho = self.param_layers[0](h[..., 0], dists)
        alpha = self.param_layers[1](h[..., 1], dists)
        beta = self.param_layers[2](h[..., 2], dists)

        rho = self.sigmoid(rho).view(*rho.shape, 1)
        alpha =self._force_positive(alpha).view(*rho.shape)
        beta = self._force_positive(beta).view(*rho.shape)

        # clamp values
        rho = torch.clamp(rho, min = 1e-5, max=1-1e-5)
        alpha = torch.clamp(alpha, min = 1e-5, max=1e5)
        beta = torch.clamp(beta, min = 1e-5, max=1e5)

        return rho, alpha, beta

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

        params = [self.param_layers[i](h[..., i], dists) 
            for i in range(7)]
        # Do rho
        params[0] = self.sigmoid(params[0]).view(*params[0].shape, 1)
        params_rho = torch.clamp(params[0], min = 1e-5, max=1-1e-5)
        
        # Other parameters
        params = [self._force_positive(params[i]).view(*(params[i].shape), 1) 
                     for i in range(1,7)]
        params = [torch.clamp(params[i], min = 1e-5, max=1e5) 
                     for i in range(6)]
        params.insert(0, params_rho)
        params[-2] = params[-2]+2.78

        return torch.cat(params, dim = 2)