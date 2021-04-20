
import pyro, torch
import numpy as np
import pyro.contrib.gp as gp
import pyro.distributions as dist
from tqdm import trange
from uuid import uuid4
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.data import float_tensor
import torch.distributions as D
from flows import AffineHalfFlow, NormalizingFlow, NormalizingFlowModel, NormalizingFlowModel2

plt.ion(); plt.style.use('ggplot')

class PlanarTransform(nn.Module):
    #2d planar flow
    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, z, normalised_u=True):
         # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(z, tuple):
            z, sum_log_abs_det_jacobians = z
        else:
            z, sum_log_abs_det_jacobians = z, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalised_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return f_z, sum_log_abs_det_jacobians
        

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
    n = 2; q = 2; m = 100

    X = np.random.normal(size = (n, q))
    W = np.random.normal(size = (q, m))
    Y = float_tensor(X @ W)

    def model(X):
        zeros = torch.zeros((n, q))
        ones = torch.ones((n, q))
        prior = D.Normal(zeros, ones).log_prob(X)

        #import pdb; pdb.set_trace()
        zeros = float_tensor([0])
        sigma = X @ X.T + torch.eye(n)*1e-4
        sigma = torch.cat([sigma[None, ...]]*m, axis=0)
        likelihood = D.MultivariateNormal(zeros, sigma).log_prob(Y.T)
        
        return prior.sum() + likelihood.sum()

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
    SVI on a ring-like posterior 
    '''

    flow_length = 32
    n_steps = 5000 
    batch_size=2
    lr = 0.005
    
    # setup base distribution
    mu = torch.nn.Parameter(torch.zeros(n,q))
    log_sigma = torch.nn.Parameter(torch.ones(n, q).normal_())
    
     # set up flow
    #flows = [AffineHalfFlow(dim=2, parity=i%2) for i in range(5)]
    
    #flows = [PlanarTransform() for i in range(10)]
    #flow_model = NormalizingFlowModel2(mu, log_sigma, flows)
    
    #optimizer = torch.optim.RMSprop(flow_model.parameters(), lr=lr, momentum=0.9, alpha=0.90, eps=1e-6, weight_decay=1e-3)
    
    optimizer = torch.optim.Adam([mu, log_sigma], lr=lr)
    #print("number of params: ", sum(p.numel() for p in flow_model.parameters()))
    
    temp = lambda i: min(1, 0.01 + i/10000)

    #flow_model(mu)
    
    for i in range(5000):
                
        #z0 = flow_model.base_dist.sample_n(1)[0]
        optimizer.zero_grad()
        
        # Computing the three terms of the variational free energy
        #zk, base_log_prob, sum_log_abs_det_jacobians = flow_model(z0)
        #p_log_joint = model(zk[-1])
        loss = 0
        base_dist = D.Normal(mu, torch.exp(log_sigma))

        for _ in range(50):
            
            z0 = base_dist.sample_n(1)[0]
            base_log_prob = base_dist.log_prob(z0).sum()
            
            p_log_joint = model(z0)
            #loss = (base_log_prob - sum_log_abs_det_jacobians).sum() - p_log_joint
            loss += -(p_log_joint - base_log_prob)
            
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            # display loss
            print('Loss at step {}: {}'.format(i, loss))
            
    import seaborn as sns
    
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    zk = flow_model.sample(10000)
    fs = zk[-1].detach()
    sns.kdeplot(fs[:,0], fs[:,1], fill=True, color='green')
    plt.scatter(fs[:,0], fs[:,1], color='b', s=2)
    #plt.scatter(x[:,0], x[:,1], color='r', s=2)
    
