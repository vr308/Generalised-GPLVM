
'''
long flows example with nnet idea
refer to the older commit for the 2-point example
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
    def __init__(self, latent_dim=2, lib_n=2, data_dim=10, n_flows=40,
                 hidden_layers=[10, 10]):
        super().__init__()

        self.q = latent_dim
        self.d = lib_n
        self.nf = n_flows
        self.uuid = str(uuid4())

        dim = self.d * self.q
        self.mu_p = nn.Parameter(torch.ones(dim).normal_().float())
        self.sigma_p = nn.Parameter(torch.ones(dim).normal_().float())
        self.flows = [dist.transforms.Planar(dim) for _ in range(self.nf)]

        hidden_layers.append(lib_n)
        layers = [nn.Linear(data_dim, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        layers.append(nn.Softmax(1))

        self.nnet = nn.Sequential(*layers)
        self.update()

    def update(self):
        self.mu = torch.tanh(self.mu_p)*10
        self.sigma = torch.tanh(self.sigma_p)*5 + 5

        self.base_dist = dist.Normal(self.mu, self.sigma)
        self.flow_dist = dist.TransformedDistribution(
            self.base_dist, self.flows)

    def forward_nnet(self, Y, Z):
        return self.nnet(Y) @ Z

    def forward_flow(self, X):
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 10; q = 2; m = 20

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = torch.tensor(X @ W).float()

    def neg_elbo():
        Z_0 = std*enc.sigma + enc.mu
        Z_f = enc.forward_flow(Z_0)
        X_n = torch.einsum('ij,ajk->aik', enc.nnet(Y), Z_f.reshape(-1, 2, 2))
        log_q = enc.flow_dist.log_prob(Z_f).mean()

        log_p = torch.distributions.Normal(0, 1).log_prob(X_n).mean(axis=0).sum()

        loc = torch.zeros(1)
        scale = torch.einsum('aij,akj->aik', X_n, X_n)
        jitter = torch.eye(n)[None, ...].repeat([len(X_n), 1, 1])*1e-4
        log_p_given_x = torch.distributions.MultivariateNormal(
            loc, (scale + jitter).repeat_interleave(m, 0)).log_prob(Y.T.repeat([len(X_n), 1]))

        return -log_p_given_x.mean()*m - (log_p - log_q)

    # for trial in range(100):
    if True:
        trial = 42
        pyro.set_rng_seed(trial)

        enc = Encoder(data_dim=m, hidden_layers=[7, 7, 7])
        std = torch.zeros(750, enc.d*enc.q).normal_().float()

        params = list(enc.parameters())
        for flow in enc.flows: params += list(flow.parameters())

        optimizer = torch.optim.Adam(params, lr=0.001)

        steps = 100001; losses = np.zeros(steps)
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

        X_samples = np.vstack(list(map(
            lambda i: (enc.nnet(Y[:2, :]).detach() @ \
            enc.flow_dist.sample((1,))[0].reshape(2, 2)).reshape(1, -1),
            range(1000)))

        plt.scatter(X_samples[:, 1], X_samples[:, 3])