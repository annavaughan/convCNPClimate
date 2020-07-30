import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import ProbabilityConverter


class Encoder(nn.Module):
    """
    ConvCNP encoder
    Elements of this class based on
    https://github.com/YannDubs
    Parameters:
    ----------
    in_channels: Int
        Total number of context variables 
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        Conv=lambda in_channels: self._make_abs_conv(nn.Conv2d)(
            in_channels, 
            in_channels, 
            groups=in_channels, 
            kernel_size=5, 
            padding=5 // 2, 
            bias=False
        )
        self.conv = Conv(in_channels)

        self.transform_to_cnn = nn.Linear(
            self.in_channels*2, 128
        )

        self.density_to_confidence = ProbabilityConverter(
            trainable_dim=self.in_channels
        )

    def _make_abs_conv(self, Conv):

        class AbsConv(Conv):
            def forward(self, input):
                return F.conv2d(
                    input,
                    self.weight.abs(),
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        return AbsConv

    def forward(self, x, mask):

        batch_size, n_channels, x_grid, y_grid = x.shape

        num = self.conv(x*mask)
        denom = self.conv(mask)

        h = num/torch.clamp(denom, min=1e-5)
        confidence = self.density_to_confidence(denom.view(-1, n_channels) * 0.1).view(
                batch_size, n_channels, x_grid, y_grid
            )

        h = torch.cat([h, confidence], dim=1)
        h = self.transform_to_cnn(h.permute(0, 2, 3, 1))

        return h