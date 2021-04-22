#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likelihood class for missing data

"""

import pyro
import torch
import gpytorch
import pyro.contrib.gp as gp
import pyro.distributions as dist
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import base_distributions

# class MaskedGaussianLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, f_loc, f_var, y=None):
#         y_var = f_var + self.variance
#         y_dist = dist.Normal(f_loc, y_var.sqrt())

#         if y is not None:
#             if y.isnan().any():
#                 y_dist = dist.MaskedDistribution(y_dist, ~y.isnan())
#                 y = torch.masked_fill(y, y.isnan(), -999.)
#             y_dist = base_distributions.Normal(f,noise.sqrt())
    
    
class MaskedGaussian(gpytorch.likelihoods.GaussianLikelihood):
    
    def forward(self, f_loc, f_var, y=None):
        y_var = f_var + self.variance
        y_dist = dist.Normal(f_loc, y_var.sqrt())

        if y is not None:
            if y.isnan().any():
                y_dist = dist.MaskedDistribution(y_dist, ~y.isnan())
                y = torch.masked_fill(y, y.isnan(), -999.)

            y_dist =\
                y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)