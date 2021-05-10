#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse Auto-Regressive Flows

"""
from torch import nn
import torch
import torch.nn.functional as F
from .autoregressive import SequentialMasked, LinearMasked

class MADE(nn.Module):
    
    """
    See Also:
        Germain et al. MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509
    """

    # Don't use ReLU, so that neurons don't get nullified.
    # This makes sure that the autoregressive test can verified
    def __init__(self, in_features, hidden_features):

        super().__init__()
        self.layers = SequentialMasked(LinearMasked(in_features, hidden_features, in_features), 
            nn.ELU(),
            LinearMasked(hidden_features, hidden_features, in_features),
            nn.ELU(),
            LinearMasked(hidden_features, in_features, in_features),
            nn.Sigmoid(),
        )
        self.layers.set_mask_last_layer()

    def forward(self, x):
        return self.layers(x)
    
class AutoRegressiveNN(MADE):
    def __init__(self, in_features, hidden_features, context_features):
        super().__init__(in_features, hidden_features)
        # remove MADE output layer
        del self.layers[len(self.layers) - 1]

    def forward(self, z, h):
        return self.layers(z) + h

class KL_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        

def get_kl_layers(model):
    return [
        module
        for module in model.modules()
        if type(module) != nn.Sequential and hasattr(module, "_kl_divergence_")
    ]


def accumulate_kl_div(model):
    return sum(map(lambda module: module._kl_divergence_, get_kl_layers(model)))


def reset_kl_div(model):
    for l in get_kl_layers(model):
        l._kl_divergence_ = 0