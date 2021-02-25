
import pyro, torch
import numpy as np
import pyro.contrib.gp as gp
import pyro.distributions as dist
from tqdm import trange
from uuid import uuid4
import matplotlib.pyplot as plt
from utils.data import float_tensor

plt.ion(); plt.style.use('ggplot')

class NormalMix(torch.nn.Module):
    def __init__(self, n_comp=7):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(n_comp).float())
        self.mu = torch.nn.Parameter(torch.ones(n_comp, 4).float())
        self.sigma = torch.nn.Parameter(torch.ones(n_comp, 16).normal_().float())
        self.update()

    @staticmethod
    def reshape_dot(x):
        x = x.reshape(4, 4)
        return (x @ x.T + torch.eye(4)*1e-6)[None, ...]

    def update(self):
        mix = dist.Categorical(logits=self.p)
        sigma = torch.cat([self.reshape_dot(self.sigma[i, :]) for i in range(len(self.p))], axis=0)
        comp = dist.MultivariateNormal(self.mu, sigma)
        self.mix_dist = dist.MixtureSameFamily(mix, comp)

    def _register(self):
        pyro.module('model_normal_mix_0', self)

    def model(self):
        self._register()
        X_unreshaped = pyro.sample('unreshaped', self.mix_dist)
        pyro.sample('X', dist.Normal(X_unreshaped.reshape(2, 2), 0.0001))

def mcmc(model, num_samples=300):
    nuts = pyro.infer.NUTS(model)
    MCMC = pyro.infer.MCMC(
            kernel=nuts,
            num_samples=num_samples,
            warmup_steps=num_samples//2,
            num_chains=1)
    MCMC.run()

    X_mcmc = MCMC.get_samples()['X']
    plt.scatter(X_mcmc[:, 0, 0], X_mcmc[:, 0, 1], alpha=0.1)
    return X_mcmc

if __name__ == '__main__':

    #########################################
    # Create Synthetic Data

    np.random.seed(42)
    n = 2; q = 2; m = 10

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = float_tensor(X @ W)

    # def model():
    #     zeros = torch.zeros((n, q))
    #     ones = torch.ones((n, q))
    #     X = pyro.sample('X', dist.Normal(zeros, ones))

    #     zeros = float_tensor([0])
    #     sigma = X @ X.T + torch.eye(n)*1e-3
    #     sigma = torch.cat([sigma[None, ...]]*m, axis=0)
    #     pyro.sample('Y', dist.MultivariateNormal(zeros, sigma), obs=Y.T)

    X_init = float_tensor(np.random.normal(size = (n, q)))

    kernel = gp.kernels.Linear(q)
    kernel.variance = torch.ones(1) # tensor([-0.3883, -0.3960])

    gpmodule = gp.models.GPRegression(X_init, Y.T, kernel, noise=torch.tensor(1e-3))
    gplvm = gp.models.GPLVM(gpmodule)

    X_mcmc = mcmc(gplvm.model, 1000)

    X_mcmc = X_mcmc.reshape(-1, 4)
    mix = NormalMix()
    optimizer = torch.optim.Adam(list(mix.parameters()), 0.05)

    steps = 10000; losses = np.zeros(steps)
    bar = trange(steps, leave=False)
    for step in bar:
        optimizer.zero_grad()
        mix.update()
        nll = -mix.mix_dist.log_prob(X_mcmc).sum()
        nll.backward(retain_graph=True)
        optimizer.step()
        bar.set_description(str(int(nll.data)))

    xx=mix.mix_dist.sample_n(1000)
    plt.scatter(xx[:, 2], xx[:, 3])

    import pickle as pkl

    with open('mix_dist_state.pkl', 'wb') as file:
        pkl.dump(mix.state_dict(), file)

    mix = NormalMix()
    with open('mix_dist_state.pkl', 'rb') as file:
        state=pkl.load(file)
    mix.load_state_dict(state)
    mix.update()

    # mix.mu.requires_grad_(False)
    # mix.sigma.requires_grad_(False)

    svi = pyro.infer.SVI(
        model=gplvm.model,
        guide=mix.model,
        optim=pyro.optim.Adam({"lr": 0.05}),
        loss=pyro.infer.Trace_ELBO(5, retain_graph=True))

    steps = 10000; losses = np.zeros(steps)
    bar = trange(steps, leave=False)
    minimum = 1000
    for step in bar:
        mix.update()
        losses[step] = svi.step()
        bar.set_description(str(int(losses[step])))
        if losses[step] < minimum:
            minimum = losses[step]
            record = mix.state_dict()

    ''' ----------------- '''

    # def potential(X):
    #     return -torch.distributions.Gamma(1e3, 1e3).\
    #             log_prob((X[:, 0]**2 + X[:, 1]**2).sqrt()) -\
    #             torch.distributions.Gamma(3e3, 1e3).\
    #             log_prob((X[:, 2]**2 + X[:, 3]**2).sqrt())

    # enc = Encoder(4)
    # params = list(enc.parameters())
    # for flow in enc.nfs: params += list(flow.parameters())
    # optimizer = torch.optim.Adam(params, 0.05)

    # steps = 10000; losses = np.zeros(steps)
    # bar = trange(steps, leave=False)
    # for step in bar:
    #     optimizer.zero_grad()
    #     _Z = enc.base_dist.sample_n(1000)
    #     _X = enc.forward(_Z)
    #     neg_elbo = enc.nf_dist.log_prob(_X) +\
    #                potential(_X)
    #     neg_elbo = neg_elbo.sum()
    #     neg_elbo.backward(retain_graph=True)
    #     optimizer.step()
    #     bar.set_description(str(int(neg_elbo.data)))
    #     if int(neg_elbo.data) < -3500:
    #         break

    # x=enc.mix_dist.sample_n(10000)
    # plt.scatter(x[:, 0], x[:, 1], alpha=0.005)
    # plt.scatter(x[:, 0], x[:, 2], alpha=0.005)

