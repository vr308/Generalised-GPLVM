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
from models.iaf import AutoRegressiveNN
import torch.nn.functional as F
import numpy as np
from models.partial_gaussian import PointNet

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
        
    def reset(self):
         raise NotImplementedError
        
class PointLatentVariable(LatentVariable):
    def __init__(self, X_init):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.register_parameter('X', X_init)

    def forward(self):
        return self.X
    
    def reset(self, X_init_test):
        self.__init__(X_init_test)
        
class MAPLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.register_parameter('X', X_init)
        self.register_prior('prior_x', prior_x, 'X')

    def forward(self):
        return self.X
    
    def reset(self, X_init_test, prior_x_test):
        self.__init__(X_init_test, prior_x_test)

class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, prior_x, data_dim, layers):
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self._init_mu_nnet(layers)
        self._init_sg_nnet(len(layers))
        self.register_added_loss_term("x_kl")

    def _get_mu_layers(self, layers):
        return (self.data_dim,) + layers + (self.latent_dim,)

    def _init_mu_nnet(self, layers):
        layers = self._get_mu_layers(layers)
        n_layers = len(layers)

        self.mu_layers = nn.ModuleList([ \
            nn.Linear(layers[i], layers[i + 1]) \
            for i in range(n_layers - 1)])

    def _get_sg_layers(self, n_layers):
        n_sg_out = self.latent_dim**2
        n_sg_nodes = (self.data_dim + n_sg_out)//2
        sg_layers = (self.data_dim,) + (n_sg_nodes,)*n_layers + (n_sg_out,)
        return sg_layers

    def _init_sg_nnet(self, n_layers):
        layers = self._get_sg_layers(n_layers)
        n_layers = len(layers)

        self.sg_layers = nn.ModuleList([ \
            nn.Linear(layers[i], layers[i + 1]) \
            for i in range(n_layers - 1)])

    def mu(self, Y):
        mu = torch.tanh(self.mu_layers[0](Y))
        for i in range(1, len(self.mu_layers)):
            mu = torch.tanh(self.mu_layers[i](mu))
            if i == (len(self.mu_layers) - 1): mu = mu * 5
        return mu        

    def sigma(self, Y):
        sg = torch.tanh(self.sg_layers[0](Y))
        for i in range(1, len(self.sg_layers)):
            sg = torch.tanh(self.sg_layers[i](sg))
            if i == (len(self.sg_layers) - 1): sg = sg * 5

        sg = sg.reshape(len(sg), self.latent_dim, self.latent_dim)
        sg = torch.einsum('aij,akj->aik', sg, sg)

        jitter = torch.eye(self.latent_dim).unsqueeze(0)*1e-5
        jitter = torch.cat([jitter for i in range(len(sg))], axis=0)
        sg += jitter
        return sg

    def forward(self, Y):
        mu = self.mu(Y)
        sg = self.sigma(Y)
        q_x = torch.distributions.MultivariateNormal(mu, sg)
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()

class IAFEncoder(NNEncoder):
    def __init__(self, n, latent_dim, context_size, prior_x, data_dim, layers, n_flows):
        self.context_size = context_size
        super().__init__(n, latent_dim, prior_x, data_dim, layers)
        self.prior_x = prior_x
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n = n
        self.flows = [IAF(latent_dim, context_size) for _ in range(n_flows)]
        
        for i in range(n_flows):
            self.add_module(f'flows{i}', self.flows[i])
        
        self.register_added_loss_term("x_kl")
        self.register_added_loss_term("x_det_jacobian")

    def _get_mu_layers(self, layers):
        return (self.data_dim,) + layers + (self.latent_dim + self.context_size,)

    def get_latent_flow_means(self, Y):    
        flow_mu, h = self.get_mu_and_h(Y)
        for flow in self.flows:
            flow_mu = flow.forward(flow_mu, h)
        return flow_mu
    
    def get_latent_flow_samples(self, Y):    
        mu, h = self.get_mu_and_h(Y, self.context_size)
        sg = self.sigma(Y)
        q_x = torch.distributions.MultivariateNormal(mu, sg)
        gauss_base_samples = q_x.rsample(sample_shape=torch.Size([500])) # shape 500 x N x Q 
        
        for flow in self.flows:
           flow_samples = flow.forward(gauss_base_samples, h)
         
        return flow_samples
        
    def get_mu_and_h(self, Y, context_size):
        
        mu_and_h = self.mu(Y)
        mu = mu_and_h[:, :-self.context_size]
        h = mu_and_h[:, -self.context_size:]
        return mu, h
    
    def forward(self, Y):
        mu, h = self.get_mu_and_h(Y, self.context_size)
        sg = self.sigma(Y)
        q_x = torch.distributions.MultivariateNormal(mu, sg)
        sample = q_x.rsample()

        for flow in self.flows:
            sample = flow.forward(sample, h)

        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)  # Update the KL term with the seed gaussian
        
        # add further loss term accounting for jacobian determinant
        sum_log_det_jac = flow_det_loss_term(self.flows, self.n, self.data_dim)
        self.update_added_loss_term('x_det_jacobian', sum_log_det_jac)
        
        return sample

class IAF(gpytorch.Module):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    This implementation is from ritchie46/vi-torch
    """

    def __init__(self, latent_dim, context_size=1, auto_regressive_hidden=1):
        super().__init__()
        self._kl_divergence_ = 0.0
        self.context_size = context_size
        self.s_t = AutoRegressiveNN(
            in_features=latent_dim,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        self.m_t = AutoRegressiveNN(
            in_features=latent_dim,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        
        num_layers = len(self.s_t.layers)
        for i in range(num_layers):
            self.add_module('flows', self.m_t.layers[i])
            self.add_module('flows', self.s_t.layers[i])
        
    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, z, h=None):
        if h is None:
            h = torch.zeros(self.context_size)

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = self.s_t(z, h) + 1.5
        sigma_t = F.sigmoid(s_t)
        m_t = self.m_t(z, h)

        # log |det Jac|
        self._kl_divergence_ = self.determine_log_det_jac(sigma_t)

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t

class VariationalLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x, data_dim):
        n, latent_dim = X_init.shape
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
    
    def reset(self, X_init, prior_x, data_dim):
        self.__init__(X_init, prior_x, data_dim)

class PointNetEncoder(LatentVariable):
    def __init__(self, n, data_dim, latent_dim, prior_x, inter_dim=5, h_dims=(5, 5), rho_dims=(5, 5)):
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        self.pointnet = PointNet(latent_dim, inter_dim, h_dims=h_dims, rho_dims=rho_dims,
                 min_sigma=1e-6, init_sigma=None, nonlinearity=torch.tanh)
        self.register_added_loss_term("x_kl")

    def forward(self, Y):
        q_x = self.pointnet(Y)
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)  # Update the KL term
        return q_x.rsample()
    
##### WIP

# def PointNetIAF(IAFEncoder):
#     def __init__(self, n, latent_dim, context_size, prior_x, data_dim, layers, n_flows):
#         self.context_size = context_size
#         super().__init__(n, latent_dim, prior_x, data_dim, layers)
#         self.prior_x = prior_x
#         self.latent_dim = latent_dim
#         self.data_dim = data_dim
#         self.n = n
#         self.flows = [IAF(latent_dim, context_size) for _ in range(n_flows)]
#         self.pointnet = PointNet(latent_dim, inter_dim, h_dims=h_dims, rho_dims=rho_dims,
#                  min_sigma=1e-6, init_sigma=None, nonlinearity=torch.tanh)
        
#         for i in range(n_flows):
#             self.add_module(f'flows{i}', self.flows[i])
        
#         self.register_added_loss_term("x_kl")
#         self.register_added_loss_term("x_det_jacobian")

#         ### Notes from A
#         # Just re-write mu and sigma methods 
            
#     def forward(self, Y):
#         mu, h = self.get_mu_and_h(Y, self.context_size)
#         sg = self.sigma(Y)
#         q_x = torch.distributions.MultivariateNormal(mu, sg)
#         sample = q_x.rsample()

#         for flow in self.flows:
#             sample = flow.forward(sample, h)

#         x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
#         self.update_added_loss_term('x_kl', x_kl)  # Update the KL term with the seed gaussian
        
#         # add further loss term accounting for jacobian determinant
#         sum_log_det_jac = flow_det_loss_term(self.flows, self.n, self.data_dim)
#         self.update_added_loss_term('x_det_jacobian', sum_log_det_jac)
        
#         return sample

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

class flow_det_loss_term(AddedLossTerm):
    
    def __init__(self, flow_list, n, data_dim):
        self.flow_list = flow_list
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        # G 
        det_loss_per_latent_dim = np.array([x._kl_divergence_ for x in self.flow_list]).sum(axis=0)
        det_loss_per_point = det_loss_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return -1*(det_loss_per_point/self.data_dim)
