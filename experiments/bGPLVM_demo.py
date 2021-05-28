#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for bGPLVM Gaussian with different inference modes. 

"""

# TODO's:
# Snaphot param state and save
# Flexible variational family

from utils.data import load_real_data 
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import *
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import os 
import pickle as pkl
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

class My_GPLVM_Model(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for X
        X_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        if pca == True:
             X_init = _init_pca(Y_train, latent_dim) # Initialise X to PCA 
        else:
             X_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable (X)
        #X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
        if nn_layers is not None:
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            X = NNEncoder(n, latent_dim, prior_x, data_dim, layers=nn_layers)
            #context_size = 10
            #n_flows=3
            #X = IAFEncoder(self.n, latent_dim, context_size, prior_x, data_dim, nn_layers, n_flows)
        else:
            #prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            #X = VariationalLatentVariable(self.n, data_dim, latent_dim, X_init, prior_x)
            X = PointLatentVariable(X_init)

        
        super(My_GPLVM_Model, self).__init__(X, q_f)
        
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
    SEED = 7
    torch.manual_seed(SEED)

    # Load some data
    
    #N, d, q, X, Y, labels = load_real_data('movie_lens')
    N, d, q, X, Y, labels = load_real_data('oilflow')
    
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=100, random_state=SEED)
    lb_train, lb_test = train_test_split(labels, test_size=100, random_state=SEED)
    
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 10
    n_inducing = 25
    pca = False
      
    # Model
    model = My_GPLVM_Model(N, data_dim, latent_dim, n_inducing, pca=pca, nn_layers=(10,5))
    
    # Likelihood
    likelihood = GaussianLikelihood()
    #likelihood = GaussianLikelihoodWithMissingObs()

    # Declaring objective to be optimised along with optimiser
    mll = VariationalELBO(likelihood, model, num_data=len(Y_train))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.001)
    
    model.get_trainable_param_names()
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    # loss_list = []
    # iterator = trange(5000, leave=True)
    # batch_size = 100
    # for i in iterator: 
    #     batch_index = model._get_batch_idx(batch_size)
    #     optimizer.zero_grad()
    #     sample = model.sample_latent_variable(Y_train)  # a full sample returns latent x across all N
    #     sample_batch = sample[batch_index]
    #     output_batch = model(sample_batch)
    #     loss = -mll(output_batch, Y_train[batch_index].T).sum()
    #     loss_list.append(loss.item())
    #     iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
    #     loss.backward()
    #     optimizer.step()
        
    # Save model
    
    if os.path.isfile('pre_trained_models/movie_lens100k_gauss_32.pkl'):
        with open('pre_trained_models/movie_lens100k_gauss_32.pkl', 'rb') as file:
            model_sd = pkl.load(file)
            model.load_state_dict(model_sd)

    filename = f'oilflow_{model_name}_{SEED}.pkl'
    with open(f'pre_trained_models/{filename}', 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    # plt.figure(figsize=(8, 6))
    # colors = ['r', 'b', 'g']
 
    # #X = model.X.q_mu.detach().numpy()
    # #X = model.X().detach().numpy()
    # #X = model.X.mu(Y).detach().numpy()
    # #std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    
    # # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
    # for i, label in enumerate(np.unique(labels)):
    #     X_i = X[labels == label]
    #     #scale_i = std[labels == label]
    #     plt.scatter(X_i[:, 4], X_i[:, 7], c=[colors[i]], label=label, marker='x')
    #     #plt.errorbar(X_i[:, 1], X_i[:, 0], xerr=scale_i[:,1], yerr=scale_i[:,0], label=label,c=colors[i], fmt='none')
