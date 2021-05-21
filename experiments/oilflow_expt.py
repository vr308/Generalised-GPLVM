#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for experiments with oilflow data

5 different inference modes:
    
   models = ['point','map','gaussian','nn_gaussian','iaf']

"""

# TODO's:
# Snaphot param state and save
# Flexible variational family

from utils.data import load_real_data 
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable, NNEncoder, IAFEncoder
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class OilFlowModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(OilFlowModel, self).__init__(X, q_f)
        
        # Kernel 
        #self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.mean_module = ZeroMean()
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
    
    torch.manual_seed(3)

    # Load some data
    
    N, d, q, X, Y, labels = load_real_data('oilflow')
      
    # Setting shapes
    N = len(Y)
    data_dim = Y.shape[1]
    latent_dim = 10
    n_inducing = 50
    pca = True
    
    # Run all 5 models and store results
    
    models = ['point','map','gaussian','nn_gaussian','iaf']
    
    model_dict = {}
    losses_dict = {}
    
    for model_name in models:
        
        # Define prior for X
        X_prior_mean = torch.zeros(N, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA 
        else:
             X_init = torch.nn.Parameter(torch.zeros(N, latent_dim))
        
        # Each inference model differs in its latent variable configuration / 
        # LatentVariable (X)
        
        nn_layers = None
        if model_name == 'point':
            
            X = PointLatentVariable(N, latent_dim, X_init)
            
        elif model_name == 'map':
            
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = MAPLatentVariable(N, latent_dim, X_init, prior_x)
            
        elif model_name == 'gaussian':
            
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = VariationalLatentVariable(N, data_dim, latent_dim, X_init, prior_x)
        
        elif model_name == 'nn_gaussian':
            
            nn_layers = (5,3,2)
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
        elif model_name == 'iaf':
            
            nn_layers = (5,3,2)
            context_size = 5
            n_flows=2
            X = IAFEncoder(N, latent_dim, context_size, prior_x, data_dim, nn_layers, n_flows)
            
        # Initialise model, likelihood, elbo and optimizer
        
        model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers)
        likelihood = GaussianLikelihood()
        elbo = VariationalELBO(likelihood, model, num_data=len(Y))
    
        optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
        ], lr=0.001)
    
        # Model params
        print(f'Training model params for model {model_name}')
        model.get_trainable_param_names()
    
        # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
        # using the optimizer provided.
        
        loss_list = []
        noise_trace = []
        
        iterator = trange(10000, leave=True)
        batch_size = 100
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gaussian']:
                sample = model.sample_latent_variable()  # a full sample returns latent x across all N
            else:
                sample = model.sample_latent_variable()
            sample_batch = sample[batch_index]
            output_batch = model(sample_batch)
            loss = -elbo(output_batch, Y[batch_index].T).sum()
            loss_list.append(loss.item())
            noise_trace.append(np.round(likelihood.noise_covar.noise.item(),3))
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            
        # Save models 
        print(model.covar_module.base_kernel.lengthscale)
        model_dict[model_name] = model
        losses_dict[model_name] = loss_list
            
    # Plot result
    
    plt.figure(figsize=(8, 6))
    colors = ['r', 'b', 'g']
 
    X = model.X.X.detach().numpy()
    #X = model.X.q_mu.detach().numpy()
    #X = model.X.mu(Y).detach().numpy()
    #std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    
    # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        #scale_i = std[labels == label]
        plt.scatter(X_i[:, 2], X_i[:, 8], c=[colors[i]], label=label)
        #plt.errorbar(X_i[:, 1], X_i[:, 0], xerr=scale_i[:,1], yerr=scale_i[:,0], label=label,c=colors[i], fmt='none')
