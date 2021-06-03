
'''
long flows example (matches mcmc)
'''

import torch
import numpy as np
from uuid import uuid4
from tqdm import trange
from pyro import set_rng_seed
from pyro.distributions.transforms import Planar
from pyro.distributions import Normal, TransformedDistribution
import matplotlib.pyplot as plt

plt.style.use('ggplot')

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.uuid = str(uuid4())
        self.mu_p = torch.nn.Parameter(torch.ones(n*q).normal_().float())
        self.sigma_p = torch.nn.Parameter(torch.ones(n*q).normal_().float())
        self.flows = [Planar(n*q) for _ in range(40)]
        self.update()

    def update(self):
        self.mu = torch.tanh(self.mu_p)*5
        self.sigma = torch.tanh(self.sigma_p)*2 + 2

        self.base_dist = Normal(self.mu, self.sigma)
        self.flow_dist = TransformedDistribution(
            self.base_dist, self.flows)

    def forward(self, X):
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 100; q = 2; m = 10

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = torch.tensor(X @ W).float()

    set_rng_seed(42)
    std = torch.zeros(75, n*q).normal_().float()

    def neg_elbo():
        base_samp = std*enc.sigma + enc.mu
        flow_samp = enc.forward(base_samp)
        log_q = enc.flow_dist.log_prob(flow_samp)

        log_p = torch.distributions.Normal(0, 1).log_prob(flow_samp).sum(axis=1)

        loc = torch.zeros(n)
        scale = flow_samp.reshape(-1, n, q)
        scale = torch.einsum('aij,akj->aik', scale, scale)
        jitter = torch.eye(n)[None, ...].repeat([len(flow_samp), 1, 1])*1e-2
        log_p_given_x = torch.distributions.MultivariateNormal(
            loc, (scale + jitter).repeat_interleave(m, 0)).log_prob(Y.T.repeat([len(flow_samp), 1]))

        return -log_p_given_x.mean()*m - (log_p - log_q).mean()

    set_rng_seed(420)

    enc = Encoder()

    params = list(enc.parameters())
    for flow in enc.flows: params += list(flow.parameters())

    optimizer = torch.optim.Adam(params, lr=0.001)

    steps = 60001; losses = np.zeros(steps)
    bar = trange(steps, leave=False)
    for step in bar:
        enc.update()
        optimizer.zero_grad()
        loss = neg_elbo()
        loss.backward(retain_graph=True)
        optimizer.step()

        losses[step] = loss.data
        bar.set_description(str(losses[step]))

        if step % 1000 == 0:
            x = enc.flow_dist.sample_n(100)
            plt.ylim(-5, 5)
            plt.xlim(-5, 5)
            plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
            plt.scatter(x[:, 2], x[:, 3], alpha=0.1)
            plt.savefig(str(step) + '__' + str(int(loss.data)) + '.png')
            plt.clf()
            plt.close()
