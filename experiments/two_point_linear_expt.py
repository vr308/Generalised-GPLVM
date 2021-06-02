
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

from utils.data import generate_synthetic_data
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import LatentVariable, flow_det_loss_term, kl_gaussian_loss_term, IAF
from models.likelihoods import GaussianLikelihoodWithMissingObs
from utils.visualisation import plot_y_reconstruction

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import LinearKernel
from gpytorch.distributions import MultivariateNormal

class LongFlow(LatentVariable):
    def __init__(self, n_flows=30):
        self.n = 2
        self.d = 10
        self.q = 2
        self.flow_len = self.n * self.q
        super().__init__(self.n, self.q)

        self.flows = [IAF(self.flow_len, self.flow_len) for _ in range(n_flows)]
        for i in range(n_flows): self.add_module(f'iaf{i}', self.flows[i])

        self.prior_x = NormalPrior(torch.zeros(1, self.flow_len), torch.ones(1, self.flow_len))

        self.mu_and_h = torch.nn.Parameter(torch.zeros(1, self.flow_len * 2))
        self.log_sigma = torch.nn.Parameter(torch.zeros(1, self.flow_len))

        self.register_added_loss_term("x_kl")
        self.register_added_loss_term("x_det_jacobian")
    
    def get_latent_flow_samples(self):    
        mu, h = self.mu_and_h[:, :-self.flow_len], self.mu_and_h[:, -self.flow_len:]
        sg = self.log_sigma.exp()
        q_x = torch.distributions.Normal(mu, sg)
        flow_samples = q_x.rsample(sample_shape=torch.Size([500])) # shape 500 x N x Q 
        
        for flow in self.flows:
           flow_samples = flow.forward(flow_samples, h)
         
        return flow_samples

    def forward(self):
        mu, h = self.mu_and_h[:, :-self.flow_len], self.mu_and_h[:, -self.flow_len:]
        sg = self.log_sigma.exp()

        q_x = torch.distributions.Normal(mu, sg)
        sample = q_x.rsample()

        for flow in self.flows:
            sample = flow.forward(sample, h)

        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, 1, self.d)        
        sum_log_det_jac = flow_det_loss_term(self.flows, 1, self.d)

        self.update_added_loss_term('x_kl', x_kl)
        self.update_added_loss_term('x_det_jacobian', sum_log_det_jac)
        
        return sample[0].reshape(self.n, self.q)

class GPLVM(BayesianGPLVM):
    def __init__(self, n_inducing=2, n_flows=30):
        self.n = 2
        self.d = 10
        self.q = 2
        self.inducing_inputs = torch.randn(d, n_inducing, q)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=torch.Size([self.d]))
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        X = LongFlow(n_flows)

        super(GPLVM, self).__init__(X, q_f)

        self.mean_module = ConstantMean(ard_num_dims=q)
        self.covar_module = LinearKernel(ard_num_dims=q)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

def train(model, likelihood, Y, steps=1000):

    elbo = VariationalELBO(likelihood, model, num_data=len(Y))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    losses = []
    iterator = trange(steps)
    for i in iterator: 
        optimizer.zero_grad()
        loss= 0.0
        for _ in range(5):
            sample = model.sample_latent_variable()
            output = model(sample)
            loss += -elbo(output, Y.T).sum()/5
        losses.append(loss.item())
        iterator.set_description('-elbo: ' + str(np.round(loss.item(), 2)) + ". Step: " + str(i))
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

    model = GPLVM()
    likelihood = GaussianLikelihood(batch_shape=[model.d])
    losses = train(model, likelihood, Y, steps=10000)
