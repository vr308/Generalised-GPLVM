''' long flow - matches mcmc '''

import numpy as np
from uuid import uuid4
from tqdm import trange
import pyro, torch
import pyro.distributions as dist
from torch import nn
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm

from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive

plt.ion(); plt.style.use('ggplot')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.uuid = str(uuid4())

        self.mu_p = nn.Parameter(torch.ones(n*q).normal_().float())
        self.sigma_p = nn.Parameter(torch.ones(n*q).normal_().float())
        # self.flows = [dist.transforms.Planar(n*q) for _ in range(40)]
        self.flows = [AffineAutoregressive(AutoRegressiveNN(n*q, [8, 8, 8, 8])) for _ in range(10)]
        self.update()

    def update(self):
        self.mu = torch.tanh(self.mu_p)*5  # mu in (-5, 5)
        self.sigma = torch.tanh(self.sigma_p)*2 + 2  # sg in (0, 4)

        self.base_dist = dist.Normal(self.mu, self.sigma)
        self.flow_dist = \
            dist.TransformedDistribution(self.base_dist, self.flows)

    def forward(self, X):
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 2; q = 2; m = 10

    X = np.random.normal(size=(n, q))
    W = np.random.normal(size=(q, m))
    Y = torch.tensor(X @ W).float()

    #########################################
    # Prepare loss function

    pyro.set_rng_seed(42)
    std = torch.zeros(50, n*q).normal_().float()  # samples for kl means
    
    def neg_elbo():
        base_samp = std*enc.sigma + enc.mu
        flow_samp = enc.forward(base_samp)
        log_q = enc.flow_dist.log_prob(flow_samp)

        log_p = torch.distributions.Normal(0, 1).\
                log_prob(flow_samp).sum(axis=1)

        loc = torch.zeros(2)
        scale = flow_samp.reshape(-1, 2, 2)
        scale = torch.einsum('aij,akj->aik', scale, scale)
        jitter = torch.eye(2)[None, ...].repeat([len(flow_samp), 1, 1])*1e-4
        log_p_given_x = torch.distributions.MultivariateNormal(
            loc, (scale + jitter).repeat_interleave(m, 0)).log_prob(Y.T.repeat([len(flow_samp), 1]))

        return -log_p_given_x.mean()*m - (log_p - log_q).mean()

    if True:
        trial = 42
        pyro.set_rng_seed(trial)

        enc = Encoder()

        params = list(enc.parameters())
        for flow in enc.flows: params += list(flow.parameters())

        optimizer = torch.optim.Adam(params, lr=0.01)

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
                x = enc.flow_dist.sample_n(1000)
                plt.ylim(-5, 5)
                plt.xlim(-5, 5)
                plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
                plt.scatter(x[:, 2], x[:, 3], alpha=0.1)
                plt.savefig(str(step) + '_' + str(int(loss.data)) + '.png')
                plt.clf()

        print(str(trial) + ': ' + str(losses[:(step - 1)].min()))
        np.save('flow_samples.npy', enc.flow_dist.sample_n(10000))
