
import pyro, torch
import numpy as np
import pyro.contrib.gp as gp
import pyro.distributions as dist
from tqdm import trange
from uuid import uuid4
import matplotlib.pyplot as plt
from utils.data import float_tensor

plt.ion(); plt.style.use('ggplot')

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mu = torch.nn.Parameter(float_tensor(np.random.normal(size=(4,))))
        self.sigma = torch.nn.Parameter(float_tensor(np.random.normal(size=(4,))))
        self.uuid = str(uuid4())
        self.nfs = [dist.transforms.Radial(4) for f in range(20)]
        self.update()

    def update(self):
        self.base_dist = dist.Normal(torch.tanh(self.mu), torch.tanh(self.sigma) + 1)
        self.nf_dist = dist.TransformedDistribution(self.base_dist, self.nfs)

    def _register(self):
        pyro.module(self.uuid + '_' + str('self'), self)
        for f in range(len(self.nfs)):
            pyro.module(self.uuid + '_' + str(f), self.nfs[f])

    def model(self, x = None, p_z = None):
        self._register()
        X_unreshaped = pyro.sample('X_unreshaped', self.nf_dist)
        pyro.sample('X', dist.Normal(X_unreshaped.reshape(2, 2), 0.0001))

    def forward(self, X):
        for i in range(len(self.nfs)):
            X = self.nfs[i](X)
        return X

def mcmc(model):
    nuts = pyro.infer.NUTS(model)
    MCMC = pyro.infer.MCMC(
            kernel=nuts,
            num_samples=300,
            warmup_steps=300,
            num_chains=1)
    MCMC.run()

    X_mcmc = MCMC.get_samples()['X']
    plt.scatter(X_mcmc[:, 0, 0], X_mcmc[:, 0, 1], alpha=0.1)
    return X_mcmc

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 2; q = 2; m = 1000

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = float_tensor(X @ W)

    def model():
        zeros = torch.zeros((n, q))
        ones = torch.ones((n, q))
        X = pyro.sample('X', dist.Normal(zeros, ones))

        zeros = float_tensor([0])
        sigma = X @ X.T + torch.eye(n)*1e-3
        sigma = torch.cat([sigma[None, ...]]*m, axis=0)
        pyro.sample('Y', dist.MultivariateNormal(zeros, sigma), obs=Y.T)

    mcmc(model)

    '''
    MCMC with pyro's GP implementation doesn't even generate the right
    latent distribution, let alone VI.
    '''

    # X_init = float_tensor(np.random.normal(size = (n, q)))

    # kernel = gp.kernels.Linear(q)
    # kernel.variance = pyro.sample('variance', dist.Delta(torch.ones(2)))

    # gpmodule = gp.models.GPRegression(X_init, Y.T, kernel, noise=torch.tensor(1e-6))
    # gplvm = gp.models.GPLVM(gpmodule)

    # mcmc(gplvm.model)

    #########################################
    # Variational Inferece

    '''
    Currently, VI doesn't work; doesn't recover the ring latent dists.
    '''

    enc = Encoder()

    adam = pyro.optim.Adam({"lr": 0.05})
    svi  = pyro.infer.SVI(model=model, guide=enc.model, optim=adam,
                loss=pyro.infer.Trace_ELBO(5, retain_graph=True))

    steps = 10000; losses = np.zeros(steps)
    bar = trange(steps, leave=False)
    for step in bar:
        enc.update()
        losses[step] = svi.step()
        bar.set_description(str(int(losses[step])))
