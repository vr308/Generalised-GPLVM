#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likelihood class for missing data

"""

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import base_distributions

class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, missing_indices, **kwargs):
        super().__init__(**kwargs)
        self.missing_indices = missing_indices

    def expected_log_prob(self, *args, **kwargs):
        res = super().expected_log_prob(*args, **kwargs)
        res[self.missing_indices] = 0.0
        return res

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        marginal = self.marginal(function_dist, *params, **kwargs)
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        
        observations[self.missing_indices] = -999.0
        res = indep_dist.log_prob(observations)
        res[self.missing_indices] = 0.0

        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res