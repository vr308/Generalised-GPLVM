
from torch import masked_fill
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions import Normal

class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_masked_obs(x):
        missing_idx = x.isnan()
        x_masked = masked_fill(x, missing_idx, -999.)
        return missing_idx, x_masked

    def expected_log_prob(self, target, input, *params, **kwargs):
        missing_idx, target = self._get_masked_obs(target)
        res = super().expected_log_prob(target, input, *params, **kwargs)
        res[missing_idx] = 0.0
        return res

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        missing_idx, observations = self._get_masked_obs(observations)
        res = super().expected_log_prob(observations, function_dist, *params, **kwargs)
        res[missing_idx] = 0.0
        return res

if __name__ == '__main__':

    import torch
    import numpy as np
    from gpytorch.distributions import MultivariateNormal
    torch.manual_seed(42)

    mu = torch.zeros(5)
    sigma = torch.ones(5)
    mvn = Normal(mu, sigma)
    x = mvn.sample()
    x[0] = np.nan
    
    lik = GaussianLikelihoodWithMissingObs(batch_shape=(5,))
    lik.expected_log_prob(x, mvn) == lik.log_marginal(x, mvn)

