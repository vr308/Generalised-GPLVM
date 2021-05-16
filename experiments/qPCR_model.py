#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qPCR - Model

"""

from utils.data import load_real_data 
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import *
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from models.likelihoods import GaussianLikelihoodWithMissingObs

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class qPCRModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA 
        else:
             X_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable (X)
        #X = PointLatentVariable(n, latent_dim, X_init)
        #X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
        if nn_layers is not None:
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            X = NNEncoder(n, latent_dim, prior_x, data_dim, layers=nn_layers)
            #context_size = 3
            #n_flows=2
            #X = IAFEncoder(n, latent_dim, context_size, prior_x, data_dim, nn_layers, n_flows)
        else:
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        
        super(OilFlowModel, self).__init__(X, q_f)
        
        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))


     def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
if __name__ == '__main__':
    
    # Setting seed for reproducibility
    
    torch.manual_seed(73)

    # Load some data
    
    # N, d, q, X, Y, labels = load_real_data('movie_lens')
    N, d, q, X, Y, labels = load_real_data('oilflow')
      
    # Setting shapes
    N = len(Y)
    data_dim = Y.shape[1]
    latent_dim = 10
    n_inducing = 25
    pca = False
    
    model = OilFlowModel(N, data_dim, latent_dim, n_inducing, pca=pca, nn_layers=(5, 3, 2))
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(Y))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.001)
    
    model.get_trainable_param_names()
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    iterator = trange(10000, leave=True)
    batch_size = 100
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.sample_latent_variable(Y)  # a full sample returns latent x across all N
        sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -mll(output_batch, Y[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        
    # Plot result
    
    # plt.figure(figsize=(8, 6))
    # colors = ['r', 'b', 'g']
 
    # X = model.X.q_mu.detach().numpy()
    # #X = model.X().detach().numpy()
    # std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    
    # # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
    # for i, label in enumerate(np.unique(labels)):
    #     X_i = X[labels == label]
    #     scale_i = std[labels == label]
    #     plt.scatter(X_i[:, 1], X_i[:, 0], c=[colors[i]], label=label)
    #     plt.errorbar(X_i[:, 1], X_i[:, 0], xerr=scale_i[:,1], yerr=scale_i[:,0], label=label,c=colors[i], fmt='none')
