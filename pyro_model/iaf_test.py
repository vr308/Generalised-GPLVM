
import numpy as np
import torch, pyro
from tqdm import trange
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive

plt.ion(); plt.style.use('ggplot')

def potential_three(z):
    z = z.T
    w1 = torch.sin(2.*np.pi*z[0]/4.)
    w2 = 3.*torch.exp(-.5*(((z[0]-1.)/.6))**2)
    p = torch.exp(-.5*((z[1]-w1)/.35)**2)
    p = p + torch.exp(-.5*((z[1]-w1+w2)/.35)**2) + 1e-30
    p = -torch.log(p) + 0.1*torch.abs_(z[0])
    return p

def load_2d_weird_latent(n=300):
    np.random.seed(42)
    Z = np.linspace(-5, 5, 500)
    Z = np.vstack([np.repeat(Z, 500), np.tile(Z, 500)]).T
    p = torch.exp(-potential_three(torch.tensor(Z))).numpy()
    p /= p.sum()

    choice_idx = range(len(p))
    sample_idx = np.random.choice(choice_idx, n, True, p)
    X = torch.tensor(Z[sample_idx, :].copy()).float()
    return X

def forward(X):
    for i in range(len(transforms)):
        X = transforms[i](X)
    return X

base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
transforms = [AffineAutoregressive(AutoRegressiveNN(2, [5, 5, 5])) for _ in range(10)]

for i, transform in enumerate(transforms):
    pyro.module("my_transform_" + str(i), transform)

params = []
for t in transforms: params += list(t.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

steps = 10000; losses = np.zeros(steps)
bar = trange(steps, leave=False)
for step in bar:
    optimizer.zero_grad()
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    z = base_dist.sample((1000,))
    x = forward(z)
    loss = (potential_three(x) + flow_dist.log_prob(x)).sum()
    loss.backward(retain_graph=True)
    optimizer.step()
    losses[step] = loss.data
    bar.set_description(str(losses[step]))

sample = flow_dist.sample((1000,))
plt.scatter(sample[:, 0], sample[:, 1], alpha=0.05)
