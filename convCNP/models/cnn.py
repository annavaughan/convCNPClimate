"""
UNet and ResNet architectures for the decoder
"""

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    UNet with identical input and output sizes
    Parameters:
    -----------
    x_dim: Int
        number of longitude points in context grid
    y_dim: Int
        number of latitude points in context grid
    """

    def __init__(self, x_dim, y_dim):
    
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = 128
        self.out_channels = 256
        self.num_halving_layers = 6

        # Up-down sampling layers
        self.upsample = nn.Linear(x_dim*y_dim, 128*128)
        self.downsample = nn.Linear(128*128, x_dim*y_dim)

        # Work out layer sizes
        ls = self.get_layer_sizes(128, 128)

        # Downwards layers
        self.l1 = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        self.l2 = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        self.l3 = nn.Conv2d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        self.l4 = nn.Conv2d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        self.l5 = nn.Conv2d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        self.l6 = nn.Conv2d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=3, stride=2, padding=2)
        
        # Upwards layers
        self.l7 = self.init_deconv(ls[1], ls[0], 
                                   8*self.in_channels,4*self.in_channels)
        self.l8 = self.init_deconv(ls[2], ls[1], 
                                   8*self.in_channels,4*self.in_channels)
        self.l9 = self.init_deconv(ls[3], ls[2], 
                                   8*self.in_channels,2*self.in_channels)
        self.l10 = self.init_deconv(ls[4], ls[3], 
                                   4*self.in_channels,2*self.in_channels)
        self.l11 = self.init_deconv(ls[5], ls[4], 
                                   4*self.in_channels,self.in_channels)
        self.l12 = self.init_deconv((128, 128), ls[5], 
                                   2*self.in_channels,self.in_channels)

        # Initialise layers
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            self.init_weights(layer)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            self.init_weights(layer)

    def init_weights(self, layer):
        """
        Initialise weights for the convolutional layers
        """
        nn.init.xavier_normal_(layer.weight, gain=1)
        nn.init.constant_(layer.bias, 1e-3)


    def skip(self, h1, h2):
        """
        Implement skip connection for ResNet
        """
        return torch.cat([h1, h2], dim=1)

    def get_layer_sizes(self, x_dim, y_dim):
        """
        Calculate the x, y sizes of each layer
        """
        s = []
        cx = x_dim
        cy = y_dim
        for _ in range(self.num_halving_layers):
            nex = np.floor((cx+1)/2+1)
            ney = np.floor((cy+1)/2+1)
            s.append((int(nex), int(ney)))
            cx = nex
            cy = ney  
        return s[::-1]

    def init_deconv(self, out, inn, in_c, out_c):
        x_out, y_out = out
        x_in, y_in = inn

        pad_x = x_out+3-2*x_in
        pad_y = y_out+3-2*y_in
        ll = nn.ConvTranspose2d(in_channels=in_c,
                                  out_channels=out_c,
                                  kernel_size=3, stride=2, padding=2,
                                  output_padding=(pad_x, pad_y))
        return ll

    def forward(self, x):
        """
        Forward pass through the convolutional structure.
        """
        x = channels_to_2nd_dim(x)

        batch, channels, lon, lat = x.shape

        # Upsample to a suitable size
        h0 = self.upsample(x.view(batch, channels, -1)).view(batch, channels, 128, 128)
        h1 = self.activation(self.l1(h0))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = self.skip(h7, h5)
        h8 = self.activation(self.l8(h7))
        h8 = self.skip(h8, h4)
        h9 = self.activation(self.l9(h8))
        h9 = self.skip(h9, h3)
        h10 = self.activation(self.l10(h9))
        h10 = self.skip(h10, h2)
        h11 = self.activation(self.l11(h10))
        h11 = self.skip(h11, h1)
        h12 = self.activation(self.l12(h11))
        out = self.skip(h12,h0)
        final = self.downsample(out.view(batch, channels*2, -1))
        final = final.view(batch, channels*2, lon, lat)
        final = channels_to_last_dim(final)
        return final

def make_depth_sep_conv(Conv):

    class DepthSepConv(nn.Module):
        """
        Make a depth separable conv. 
        """

        def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            confidence=False, 
            bias=True, **kwargs):

            super().__init__()
            self.depthwise = Conv(in_channels, 
                in_channels, 
                kernel_size, 
                groups=in_channels, 
                bias=bias, **kwargs)

            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

    return DepthSepConv

def channels_to_2nd_dim(h):
    return h.permute(*([0, h.dim() - 1] + list(range(1, h.dim() - 1))))


def channels_to_last_dim(h):
    return h.permute(*([0] + list(range(2, h.dim())) + [1]))

class ResConvBlock(nn.Module):
    """
    Residual block for Resnet CNN
    Adapted from https://github.com/YannDubs
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        Padder=None,
    ):
        super().__init__()
        self.activation = activation

        padding = kernel_size // 2

        if Padder is not None:
            self.padder = Padder(padding)
            padding = 0
        else:
            self.padder = nn.Identity()

        self.norm1 = Normalization(in_chan)
        self.conv1 = make_depth_sep_conv(Conv)(
            in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
        )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

    def forward(self, X):
        out = self.padder(X)
        out = self.conv1(self.activation(self.norm1(X)))
        out = self.padder(out)
        out = self.conv2_depthwise(self.activation(self.norm2(X)))
        out = out + X
        out = self.conv2_pointwise(out)
        return out


class CNN(nn.Module):
    """
    Resnet CNN
    Adapted from https://github.com/YannDubs
    """

    def __init__(self, n_channels, ConvBlock, n_blocks=3, **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = n_channels
        self.out_channels = n_channels
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(in_chan, out_chan, **kwargs) for in_chan, out_chan in self.in_out_channels]
        )

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, h):

        h = channels_to_2nd_dim(h)

        for conv_block in self.conv_blocks:
            h = conv_block(h)

        h = channels_to_last_dim(h)

        return h