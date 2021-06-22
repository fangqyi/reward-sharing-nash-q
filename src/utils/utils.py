"""
Code adapted from https://github.com/TonghanWang/ROMA
"""

import numpy as np
import torch
from torch import nn


def identity(x):
    return x


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Tensor shape must have dimensions >= 2")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class ConfigDict(dict):
    """Configuration container class."""

    def __init__(self, initial_dictionary=None):
        """Creates an instance of ConfigDict.
        Args:
          initial_dictionary: Optional dictionary or ConfigDict containing initial
          parameters.
        """
        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _convert_sub_configs(value)
            super().__init__(initial_dictionary)
        else:
            super().__init__()

    def __setattr__(self, attribute, value):
        self[attribute] = _convert_sub_configs(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super().__setitem__(key, _convert_sub_configs(value))
