#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Variable class with sub-classes that determine type of inference for the latent variable

"""
import gpytorch
import torch
from torch import nn
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm

class LatentVariable(gpytorch.Module):
    
    """
    :param n (int): Size of the latent space.
    :param latent_dim (int): Dimensionality of latent space.

    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim
        
    def forward(self, x):
        raise NotImplementedError
        
class PointLatentVariable(LatentVariable):
    def __init__(self, n, latent_dim, X_init):
        super().__init__(n, latent_dim)
        self.register_parameter('X', X_init)

    def forward(self):
        return self.X
        
class MAPLatentVariable(LatentVariable):
    
    def __init__(self, n, latent_dim, X_init, prior_x):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.register_parameter('X', X_init)
        self.register_prior('prior_x', prior_x, 'X')

    def forward(self):
        return self.X

class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, prior_x, data_dim, layers):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.data_dim = data_dim

        n_sg_out = latent_dim**2
        n_sg_nodes = (data_dim + n_sg_out)//2
        mu_layers = (data_dim,) + layers + (latent_dim,)
        sg_layers = (data_dim,) + (n_sg_nodes,)*len(layers) + (n_sg_out,)
        n_layers = len(mu_layers)

        self.mu_layers = nn.ModuleList([ \
            nn.Linear(mu_layers[i], mu_layers[i + 1]) \
            for i in range(n_layers - 1)])

        self.sg_layers = nn.ModuleList([ \
            nn.Linear(sg_layers[i], sg_layers[i + 1]) \
            for i in range(n_layers - 1)])

        self.register_added_loss_term("x_kl")

    def forward(self, Y):
        mu = torch.relu(self.mu_layers[0](Y))
        for i in range(1, len(self.mu_layers)):
            mu = torch.tanh(self.mu_layers[i](mu))
            if i == (len(self.mu_layers) - 1): mu = mu * 5

        sg = torch.relu(self.sg_layers[0](Y))
        for i in range(1, len(self.sg_layers)):
            sg = torch.relu(self.sg_layers[i](sg))

        sg = sg.reshape(len(sg), self.latent_dim, self.latent_dim)
        sg = torch.einsum('aij,akj->aik', sg, sg)

        jitter = torch.eye(self.latent_dim).unsqueeze(0)*1e-6
        jitter = torch.cat([jitter for i in range(len(sg))], axis=0)
        sg += jitter

        q_x = torch.distributions.MultivariateNormal(mu, sg)
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()


class VariationalLatentVariable(LatentVariable):
    
    def __init__(self, n, data_dim, latent_dim, X_init, prior_x):
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))     
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")

    def forward(self):
        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma))
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)  # Update the KL term
        return q_x.rsample()
    
class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_x, p_x, n, data_dim):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        # G 
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0) # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return (kl_per_point/self.data_dim)
