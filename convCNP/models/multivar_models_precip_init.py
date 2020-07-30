"""
Models for multivariate downscaling - predicting P(Tmax|precipitation)
"""

from .models.encoder import Encoder
from .models.mlp import MLP
from .models.final_layers import GaussianFinalLayer, GammaFinalLayer, GammaGPFinalLayer
from .models.cnn import CNN, ResConvBlock
from .models.utils import force_positive

class GammaElevStacked(nn.Module):
    """
    Multivariate downscaling P(Tmax|precip)
    - Precip
    Parameters:
    ----------
    decoder: convolutional architecture
    in_channels: Int
        Number of context variables
    ls: float
        Initial length scale for the RBF kernel
    """

    def __init__(self, 
                 decoder,
                 tmax_model,
                 in_channels = 1, 
                 ls = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()

        self.encoder = Encoder(in_channels = in_channels+2)
        self.mlp = MLP(in_channels = 128,
            out_channels = 3,
            hidden_channels = 64,
            hidden_layers = 4)
        self.decoder = decoder
        self.out_layer = GammaFinalLayer(
            init_ls = ls,
            n_params = 3
        )

        self.elev_mlp = MLP(8,
            out_channels = 3,
            hidden_channels = 64,
            hidden_layers = 4)
        
        self.tmax_model = tmax_model
        
    def forward(self, h, mask, dists, elev):

        self.tmax_model.eval()
        
        # Predict tmax at the target points
        tmax_target, tmax_context  = tmax_model(h,
                                       mask,
                                       dists,
                                       elev)
        
        # Make new context and mask 
        mask = torch.cat([mask, torch.ones(mask.shape[0], 2, 87, 50).cuda()], dim = 1)
        h = torch.cat([h, tmax_context.permute(0, 3, 1, 2)], axis = 1)

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
        out = torch.cat([tmax_target, out[...,0], elev], dim = 2)
        out = self.elev_mlp(out)
        out[...,0] = self.sigmoid(out[...,0])
        out[...,1:] = force_positive(out[...,1:])
        
        return out
        
class TmaxElevStacked(nn.Module):
    """
    Multivariate downscaling P(Tmax|precip)
    - Tmax
    Parameters:
    ----------
    decoder: convolutional architecture
    
    in_channels: Int
        Number of context variables
    ls: float
        Initial length scale for the RBF kernel
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
        
        # Return target refinements and coarse scale predictions
        return out, x

