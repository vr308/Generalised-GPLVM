#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EncoderFlow implements the flow transformed q_(X)

"""

import torch
import numpy as np
import pyro.distributions as dist
import matplotlib.pyplot as plt
from utils.data import float_tensor
import scipy

class EncoderFlow(torch.nn.Module):
    ''' '''
    def __init__(self, num_latent, num_flows, base_params=None,
                 flow_type='planar', flows=None, activate_flow=True):
        '''Initiates the flow.

        You can manually provide a list of flows to use (each one from
        pyro.distributions.transforms). This is useful if you want to
        input tranforms like the exponential.

        If this isn't given, n_flows of flow_type are initialized.

        Parameters
        ----------
        num_latent : int
            Number of dimensions for the flow to act on.
        num_flows : int
            Number of flows.
        base_params : EncoderBase instance
        flow_type : str
            One of 'sylvester', 'planar' or 'radial'.
        flows : list
            List of relevant pyro.distributions.transforms.
        activate_flow : bool
            If False, the flow is set to the identity transform (not
            implemented for custom flows via the 'flows' argument).

        '''
        super().__init__()

        if flows is None:
            if flow_type == 'sylvester':
                sylvester = dist.transforms.Sylvester
                q = num_latent
                self.flows = [sylvester(q, q) for i in range(num_flows)]
            elif flow_type == 'planar':
                planar = dist.transforms.Planar
                self.flows = [planar(num_latent) for i in range(num_flows)]
            elif flow_type == 'radial':
                radial = dist.transforms.Radial
                self.flows = [radial(num_latent) for i in range(num_flows)]
            else:
                raise NotImplementedError('Flow type not implemented.')
        else:
            self.flows = flows

        if not activate_flow:
            self.deactivate_flow(num_latent)

        for i, flow in enumerate(self.flows):
            if hasattr(flow, 'parameters'):
                super().add_module('flow_module_' + str(i), flow)

        if base_params is not None:
            self.generate_base_dist(
                base_params.model, base_params.mu, base_params.sigma)
        else:
            print('Remember to call `self.generate_base_dist(base_params)`.')

    def generate_base_dist(self, model, mu, sigma, mask_idx=None):
        ''' Generate the transformed distribution.
        'base_params` must have the `model`, `mu` and `sigma` attributes.'''

        if mask_idx is not None:
            mu_train = mu[:mask_idx].clone().detach()
            mu_test = mu[mask_idx:]
            mu = torch.cat([mu_train, mu_test])
            
            sg_train = sigma[:mask_idx].clone().detach()
            sg_test = sigma[mask_idx:]
            sigma = torch.cat([sg_train, sg_test])

        if model in ('mf', 'pca', 'gp'):
            self.base_dist = dist.Normal(mu, sigma)

        # elif base_params.model == 'gp':

            # Non FITC GP Approx:
            # mu = base_params.mu.T
            # sigma = base_params.sigma
            # MVN = dist.MultivariateNormal
            # X_latent = pyro.sample('X_latent', MVN(mu, scale_tril=sigma))
            # self.base_dist = dist.Normal(X_latent.T, 1e-4)

            # the proper way of doing this would be something like:
            # (from my expts and advice from the pyro forum)
            # mvn = dist.MultivariateNormal(mu, sigma).to_event(1)
            # _flows = [_Transpose(), dist.transforms.Planar(d), _Transpose()]
            # flow = dist.TransformedDistribution(mvn, _flows)

        elif model == 'nn':
            self.base_dist = dist.MultivariateNormal(mu, scale_tril=sigma)

        base = self.base_dist
        self.flow_dist = dist.TransformedDistribution(base, self.flows)

    def deactivate_flow(self, q):
        ''' Set flow to the identity function.'''
        loc = torch.zeros(q).float()
        scale = torch.ones(q).float()
        self.flows = [dist.transforms.AffineTransform(loc, scale)]

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.parameters():
            param.requires_grad_(trainable)
        if hasattr(self, 'flows'):
            for flow in self.flows:
                if hasattr(flow, 'parameters'):
                    for param in flow.parameters():
                        param.requires_grad_(trainable)

    def forward(self, X):
        '''Compute the flow function X_f = flow(X_b).'''
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

    def X_map(self, n_restarts=3, use_base_mu=False):
        '''Compute mode of the flow. This uses scipy because torch fails.'''

        if use_base_mu:
            Z = float_tensor(self.base_dist.loc.detach().numpy())
            return self.forward(Z).detach().float()

        torch.manual_seed(42)

        def loss(Z):
            X = self.forward(float_tensor(Z.reshape(shape)))
            lp = self.flow_dist.log_prob(X)
            return -lp.sum().detach().numpy()

        def d_loss(Z):
            return scipy.optimize.approx_fprime(Z, loss, 1e-3)

        optimize = scipy.optimize.lbfgsb.fmin_l_bfgs_b

        shape = tuple(self.base_dist.loc.shape)
        best_loss = np.inf

        for _ in range(n_restarts):
            Z_guess = self.base_dist().detach().numpy().reshape(-1)
            Z_opt, loss_cur, _ = optimize(loss, Z_guess, fprime=d_loss)

            if loss_cur < best_loss:
                best_loss = loss_cur
                Z_best = Z_opt

        torch.seed()
        Z = float_tensor(Z_best.reshape(shape))
        return self.forward(Z).detach().float()

    def plot_flow(self, n_grid=500, mu=torch.zeros(2), sg=torch.ones(2),
                  distb='norm'):
        '''Plot a flow transformed normal distribution.

        The viz is basically:
        flow_grid = flow.forward(uniform_grid)
        log_prob_on_flow_grid = flow.log_prob(flow_grid)
        heatmap(flow_grid[0], flow_grid[1], log_prob_on_flow_grid)

        Parameters
        ----------
        n_grid : int
            Resolution of the grid of the base distribution that the flow
            acts on.
        mu : torch.tensor
            Size (2,). Means of the normal distribution the flow acts on.
        sg : torch.tensor
            Size (2,). SD of the normal distribution the flow acts on.
        distb : str
            Base distribution, one of 'norm' or 'unif'.
        '''

        grid_base_dist = np.linspace(-5, 5, n_grid)
        grid_base_dist = np.vstack([np.repeat(grid_base_dist, n_grid),
                                    np.tile(grid_base_dist, n_grid)]).T

        if distb == 'norm':
            distb = dist.Normal(mu, sg)
        elif distb == 'unif':
            distb = dist.Uniform(-5.*sg + mu, 5*sg + mu)

        flow_dist = dist.TransformedDistribution(
            base_distribution=distb,
            transforms=self.flows)

        transformed_grid = self.forward(float_tensor(grid_base_dist))
        log_p = flow_dist.log_prob(transformed_grid).detach()

        # plt.figure()
        plt.hexbin(transformed_grid.detach()[:, 0],
                   transformed_grid.detach()[:, 1],
                   log_p.exp(), gridsize=300)