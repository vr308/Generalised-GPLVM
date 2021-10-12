
import numpy as np
import torch, gpytorch
from tqdm import trange

import pandas as pd

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, PeriodicKernel, RBFKernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.models import ApproximateGP
from gpytorch.constraints import Interval
from gpytorch.priors import NormalPrior

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
        self.register_parameter('X', torch.nn.Parameter(torch.ones(n, latent_dim)))

    def forward(self):
        return self.X

class GPLVM(ApproximateGP):
    def __init__(self, n, data_dim, latent_dim, n_inducing):
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=(data_dim,)) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=False)

        super(GPLVM, self).__init__(q_f)

        self.intercept = ConstantMean(batch_shape=(data_dim,))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

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

    data = pd.read_csv('taxi_count_data.csv')
    Y = np.array(data[['yellow', 'green', 'fhv']], dtype=int)

    n = len(Y); d = len(Y.T); m = 36; q = 3
    n_train = 500; n_test = n - n_train
    Y_train = Y[:n_train, :]
    Y_test  = Y[n_train:, :]
    latent_var = 'point'

    # X_latent = PointLatentVariable(n, q)

    train_zeros = torch.zeros(n_train, q)
    test_zeros  = torch.zeros(n_test,  q)
    param = lambda x: torch.nn.Parameter(x)

    from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable

    if latent_var == 'point':
        X = PointLatentVariable(param(train_zeros))

    elif latent_var == 'map':
        prior_x = NormalPrior(train_zeros, torch.ones_like(train_zeros))
        prior_x_test = NormalPrior(test_zeros, torch.ones_like(test_zeros))
        X = MAPLatentVariable(param(train_zeros), prior_x)

    elif latent_var == 'gauss':
        prior_x = NormalPrior(train_zeros, torch.ones_like(train_zeros))
        prior_x_test = NormalPrior(test_zeros, torch.ones_like(test_zeros))
        X = VariationalLatentVariable(param(train_zeros), prior_x, q)

    gplvm = GPLVM(n_train, d, q, m)
    likelihood = PoissonLikelihood()

    losses = train(gplvm=gplvm, X_latent=X, likelihood=likelihood, Y=torch.tensor(Y_train), steps=10000)

