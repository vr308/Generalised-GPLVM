
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

from utils.data import generate_synthetic_data
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import IAFEncoder
from models.likelihoods import GaussianLikelihoodWithMissingObs
from utils.visualisation import plot_y_reconstruction

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import LinearKernel
from gpytorch.distributions import MultivariateNormal

class GPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, nn_layers=None, context_size=2, n_flows=10):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)

        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
        X = IAFEncoder(n, latent_dim, context_size, prior_x, data_dim, nn_layers, n_flows)

        super(GPLVM, self).__init__(X, q_f)

        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.covar_module = LinearKernel(ard_num_dims=latent_dim)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):        
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

def train(model, likelihood, Y, steps=1000, batch_size=100):

    elbo = VariationalELBO(likelihood, model, num_data=len(Y))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.001)

    losses = []
    iterator = trange(steps)
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        loss= 0.0
        for _ in range(5):
            sample = model.sample_latent_variable(Y)
            sample_batch = sample[batch_index]
            output_batch = model(sample_batch)
            loss += -elbo(output_batch, Y[batch_index].T).sum()/5
        losses.append(loss.item())
        iterator.set_description(
            '-elbo: ' + str(np.round(loss.item(), 2)) +\
            ". Step: " + str(i))
        loss.backward()
        optimizer.step()

    return losses

if __name__ == '__main__':

    torch.manual_seed(42)

    np.random.seed(42)
    n = 2; q = 2; d = 10

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, d))
    Y = torch.tensor(X @ W).float()

    model = GPLVM(n, d, q, n_inducing=2, n_flows=30, nn_layers=(4, 4))
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    losses = train(model, likelihood, Y, steps=10000, batch_size=n)
