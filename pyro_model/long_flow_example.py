
'''
long flows example (matches mcmc)
'''

import numpy as np
from uuid import uuid4
from tqdm import trange
import pyro, torch
import pyro.distributions as dist
from torch import nn
# from pyro.nn import PyroModule
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm

plt.ion(); plt.style.use('ggplot')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = 2
        self.d = 2
        self.nf = 40
        self.uuid = str(uuid4())

        dim = self.d * self.q
        self.mu_p = nn.Parameter(torch.ones(dim).normal_().float())
        self.sigma_p = nn.Parameter(torch.ones(dim).normal_().float())
        self.flows = [dist.transforms.Planar(dim) for _ in range(self.nf)]

        self.nnet = lambda x: torch.eye(self.q)
        # PyroModule[nn.Sequential](
        #     PyroModule[nn.Linear](10, 10),
        #     PyroModule[nn.ReLU](),
        #     PyroModule[nn.Linear](10, 10),
        #     PyroModule[nn.ReLU](),
        #     PyroModule[nn.Linear](10, self.d),
        #     PyroModule[nn.Tanh]()
        # )
        self.update()

    def update(self):
        self.mu = torch.tanh(self.mu_p)*5
        self.sigma = torch.tanh(self.sigma_p)*2 + 2

        self.base_dist = dist.Normal(self.mu, self.sigma)
        self.flow_dist = dist.TransformedDistribution(
            self.base_dist, self.flows)

    # model(): self.nnet(Y) @ Z

    def forward(self, X):
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 2; q = 2; m = 10
    assert n == 2 # the nnet = lambda ... requires this

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = torch.tensor(X @ W).float()

    pyro.set_rng_seed(42)
    std = torch.zeros(750, n*q).normal_().float()

    def neg_elbo():
        base_samp = std*enc.sigma + enc.mu
        flow_samp = enc.forward(base_samp)
        log_q = enc.flow_dist.log_prob(flow_samp)

        log_p = torch.distributions.Normal(0, 1).log_prob(flow_samp).sum(axis=1)

        loc = torch.zeros(1)
        scale = flow_samp.reshape(-1, 2, 2)
        scale = torch.einsum('aij,akj->aik', scale, scale)
        jitter = torch.eye(2)[None, ...].repeat([len(flow_samp), 1, 1])*1e-4
        log_p_given_x = torch.distributions.MultivariateNormal(
            loc, (scale + jitter).repeat([m, 1, 1])).log_prob(Y.T.repeat([len(flow_samp), 1]))

        return -log_p_given_x.mean()*m - (log_p - log_q).mean()

    # for trial in range(100):
    if True:
        trial = 42
        pyro.set_rng_seed(trial)

        enc = Encoder()

        params = list(enc.parameters())
        for flow in enc.flows: params += list(flow.parameters())

        optimizer = torch.optim.Adam(params, lr=0.001)

        steps = 60001; losses = np.zeros(steps)
        bar = trange(steps, leave=False)
        for step in bar:
            enc.update()
            optimizer.zero_grad()
            try:
                loss = neg_elbo()
                loss.backward(retain_graph=True)
                optimizer.step()
            except:
                break

            losses[step] = loss.data
            bar.set_description(str(losses[step]))

            if step % 1000 == 0:
                x = enc.flow_dist.sample_n(10000)
                plt.ylim(-5, 5)
                plt.xlim(-5, 5)
                plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
                plt.scatter(x[:, 2], x[:, 3], alpha=0.1)
                plt.savefig(str(step) + '_' + str(int(loss.data)) + '.png')
                plt.clf()

        print(str(trial) + ': ' + str(losses[:(step - 1)].min()))
        np.save('flow_samples.npy', enc.flow_dist.sample_n(10000))
