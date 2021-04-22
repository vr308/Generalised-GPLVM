
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal, base_distributions

class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def expected_log_prob(self, *args, **kwargs):
        res = super().expected_log_prob(*args, **kwargs)
        res[res.isnan()] = 0.0
        return res

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        marginal = self.marginal(function_dist, *params, **kwargs)
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        
        missing_indices = observations.isnan()
        observations[missing_indices] = -999.0
        res = indep_dist.log_prob(observations)
        res[missing_indices] = 0.0

        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

if __name__ == '__main__':

    import torch as t
    import numpy as np

    mvn = MultivariateNormal(t.zeros(5, 2), t.cat([t.eye(2)[None, ...]]*5, axis=0))
    x = mvn.sample()
    x[0, 1] = np.nan
    
    lik = GaussianLikelihoodWithMissingObs(batch_shape=(5,))
    lik.expected_log_prob(x, mvn)
    lik.log_marginal(x, mvn)
