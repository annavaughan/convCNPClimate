import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilityConverter(nn.Module):
    """
    Convert from densities to probabilities
    From https://github.com/YannDubs
    """

    def __init__(
        self,
        trainable_dim=1,):

        super().__init__()
        self.min_p = 0.0
        self.trainable_dim = trainable_dim
        self.initial_temperature = 1.0
        self.initial_probability = 0.5
        self.initial_x = 0.0
        self.temperature_transformer = F.softplus

        self.reset_parameters()

    def reset_parameters(self):
        self.temperature = torch.tensor([self.initial_temperature] * self.trainable_dim)
        self.temperature = nn.Parameter(self.temperature)

        initial_bias = self._probability_to_bias(
            self.initial_probability, initial_x=self.initial_x
        )

        self.bias = torch.tensor([initial_bias] * self.trainable_dim)
        self.bias = nn.Parameter(self.bias)

    def _probability_to_bias(self, p, initial_x=0):
        """
        Compute the bias to use to satisfy the constraints.
        """
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)

        bias = -(torch.log((1 - p) / p) / self.initial_temperature + initial_x)
        return bias

    def _rescale_range(self, p, init_range, final_range):
        """
        Rescale vec to be in new range
        """
        init_min, final_min = init_range[0], final_range[0]
        init_delta = init_range[1] - init_range[0]
        final_delta = final_range[1] - final_range[0]

        return (((p - init_min)*final_delta) / init_delta) + final_min

    def forward(self, x):
        self.temperature.to(x.device)
        self.bias.to(x.device)

        temperature = self.temperature_transformer(self.temperature)
        full_p = torch.sigmoid((x + self.bias) * temperature)
        p = self._rescale_range(full_p, (0, 1), (self.min_p, 1 - self.min_p))

        return p