
'''
Nnet + long flows example; Doesn't work
'''

import numpy as np
from uuid import uuid4
from tqdm import trange
import pyro, torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
import matplotlib.pyplot as plt

plt.ion(); plt.style.use('ggplot')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = 2
        self.d = 2
        self.nf = 30
        self.uuid = str(uuid4())

        dim = self.d * self.q
        self.mu = nn.Parameter(torch.ones(dim).normal_().float())
        self.sigma = nn.Parameter(torch.ones(dim).normal_().float())
        self.flows = [dist.transforms.Radial(dim) for _ in range(self.nf)]

        self.nnet = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](10, 10),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Linear](10, 10),
            PyroModule[nn.ReLU](),
            PyroModule[nn.Linear](10, self.d),
            PyroModule[nn.Tanh]()
        )
        self.update()

    def update(self):
        mu_flat = 3 * torch.tanh(self.mu)
        sg_flat = 1 * torch.tanh(self.sigma) + 1

        self.base_dist = dist.Normal(mu_flat, sg_flat)
        self.flow_dist = dist.TransformedDistribution(
            self.base_dist, self.flows)

    def _register(self):
        pyro.module(self.uuid + '_' + str('self'), self)
        for f in range(self.nf):
            pyro.module(self.uuid + '_' + str(f), self.flows[f])

    def model(self):
        self._register()
        Z = pyro.sample('Z', self.flow_dist)
        Z = Z.reshape(self.d, self.q)
        pyro.sample('X', dist.Normal(self.nnet(Y) @ Z, 1e-2))
        

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 250; q = 2; m = 10

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = torch.tensor(X @ W).float()

    # def model():
    #     zeros = torch.zeros((n, q))
    #     ones = torch.ones((n, q))
    #     X = pyro.sample('X', dist.Normal(zeros, ones))

    #     zeros = torch.tensor([0])
    #     sigma = X @ X.T + torch.eye(n)*1e-3
    #     sigma = torch.cat([sigma[None, ...]]*m, axis=0)
    #     pyro.sample('Y', dist.MultivariateNormal(zeros, sigma), obs=Y.T)

    X_init = torch.tensor(np.random.normal(size = (n, q)))
    kernel = gp.kernels.Linear(q)
    kernel.variance = torch.ones(1)

    gpmodule = gp.models.GPRegression(
        X=X_init, y=Y.T,
        kernel=kernel,
        noise=torch.tensor(1e-3))
    gplvm = gp.models.GPLVM(gpmodule)

    enc = Encoder()
    svi = pyro.infer.SVI(
        model=gplvm.model,
        guide=enc.model,
        optim=pyro.optim.Adam({"lr": 0.05}),
        loss=pyro.infer.Trace_ELBO(5, retain_graph=True))

    steps = 10000; losses = np.zeros(steps); minimum = 10000
    bar = trange(steps, leave=False)
    for step in bar:
        enc.update()
        losses[step] = svi.step()
        bar.set_description(str(int(losses[step])))
        if losses[step] < -2000:
            x = enc.flow_dist.sample_n(1)[0]
            plt.scatter(x[0], x[1], c='blue', alpha=0.05)
            plt.scatter(x[2], x[3], c='red', alpha=0.05)

    # x = enc.flow_dist.sample_n(1000)
    # plt.scatter(x[:, 1], x[:, 0])

