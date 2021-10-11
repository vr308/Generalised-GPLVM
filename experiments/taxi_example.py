
import numpy as np
import torch, gpytorch
from tqdm import trange

import pandas as pd

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, PeriodicKernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.models import ApproximateGP
from gpytorch.constraints import Interval

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

class PoissonLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, **kwargs):
        return base_distributions.Poisson(rate=function_samples.exp())

class PointLatentVariable(gpytorch.Module):
    def __init__(self, n, latent_dim):
        super().__init__()
        self.register_parameter('X_unt', torch.nn.Parameter(torch.ones(n, latent_dim)))

    def forward(self):
        return torch.sigmoid(self.X_unt) * 2*np.pi

class GPLVM(ApproximateGP):
    def __init__(self, n, data_dim, n_inducing):
        latent_dim = 1
        self.inducing_inputs = torch.tensor(np.linspace(0, 2*np.pi, n_inducing))[None, ..., None].repeat((data_dim, 1, latent_dim)).float()
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=(data_dim,)) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=False)

        super(GPLVM, self).__init__(q_f)

        self.intercept = ConstantMean(batch_shape=(data_dim,))
        self.covar_module = \
            ScaleKernel(PeriodicKernel(ard_num_dims=1, period_length_constraint=Interval(2*np.pi - 0.001, 2*np.pi)))

    def forward(self, X):
        return MultivariateNormal(self.intercept(X), self.covar_module(X))

def train(gplvm, likelihood, X_latent, Y, steps=1000):

    elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
    optimizer = torch.optim.Adam([
        dict(params=gplvm.parameters(), lr=0.01),
        dict(params=likelihood.parameters(), lr=0.01),
        dict(params=X_latent.parameters(), lr=0.01)
    ])

    losses = []; iterator = trange(steps, leave=False)
    for i in iterator:
        optimizer.zero_grad()
        loss = -elbo_func(gplvm(X_latent()), Y.T).sum()
        losses.append(loss.item())
        iterator.set_description('-elbo: ' + str(np.round(loss.item(), 2)) + '. Step: ' + str(i))
        loss.backward()
        optimizer.step()

    return losses

if __name__ == '__main__':

    data = pd.read_csv('../taxi_count_data.csv')
    Y = np.array(data[['yellow', 'green', 'fhv']], dtype=int)

    n = len(Y); d = len(Y.T); m = 10

    gplvm = GPLVM(n, d, m)
    likelihood = PoissonLikelihood()
    X_latent = PointLatentVariable(n, 1)

    losses = train(gplvm=gplvm, X_latent=X_latent, likelihood=likelihood, Y=torch.tensor(Y), steps=10000)

    plt.plot(losses)
