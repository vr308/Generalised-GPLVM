#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for experiments with MovieLens100K / MovieLens1m

5 different inference modes:
    
   models = ['gaussian']

"""

from utils.data import load_real_data 
from models.likelihoods import GaussianLikelihoodWithMissingObs
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import VariationalLatentVariable
from matplotlib import pyplot as plt
import torch
import numpy as np
import os 
import pickle as pkl
from tqdm import trange
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split

class MovieLensModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(MovieLensModel, self).__init__(X, q_f)
        
        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        #self.mean_module = ZeroMean()
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = {}
    noise_trace_dict = {}
    
    TEST = True
    increment = np.random.randint(0,100,1)
    
    SEED = 7 + increment[0]
    torch.manual_seed(SEED)

    # Load some data
    
    N, d, q, X, Y, labels = load_real_data('movie_lens_100k')
    
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=100, random_state=SEED)
    #lb_train, lb_test = train_test_split(labels, test_size=100, random_state=SEED)
    
    Y_train = torch.Tensor(Y_train).cuda()
    Y_test = torch.Tensor(Y_test).cuda()
      
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 15
    n_inducing = 34
    pca = False
    
    # Run all 4 models and store results
    
    # Define prior for X
    X_prior_mean = torch.zeros(N, latent_dim)  # shape: N x Q
    X_prior_mean_test = X_prior_mean[0:len(Y_test),:]

    X_init = torch.nn.Parameter(torch.zeros(N, latent_dim))
      
    nn_layers = None
    model_name = 'gauss'
      
    ae = False
    prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
    prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test))
    X = VariationalLatentVariable(X_init, prior_x, latent_dim)
    
    # Initialise model, likelihood, elbo and optimizer
    
    model = MovieLensModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers)
    likelihood = GaussianLikelihoodWithMissingObs()
    
    model = model.cuda()
    likelihood = likelihood.cuda()
    elbo = VariationalELBO(likelihood, model, num_data=len(Y_train), beta=0.9)

    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.005)

    # Model params
    print(f'Training model params for model {model_name}')
    model.get_trainable_param_names()

    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    noise_trace = []
    
    iterator = trange(8000, leave=True)
    batch_size = 100
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample_batch = model.sample_latent_variable(batch_idx=batch_index)  # a batched sample
        output_batch = model(sample_batch)
        loss = -elbo(output_batch, Y_train[batch_index].T).sum()
        loss_list.append(loss.item())
        #noise_trace.append(np.round(likelihood.noise_covar.noise.item(),3))
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
    model.store(loss_list, likelihood)
    
    # if os.path.isfile('pre_trained_models/movie_lens100k_gauss_93.pkl'):
    #    with open('pre_trained_models/movie_lens100k_gauss_93.pkl', 'rb') as file:
    #        model_sd = pkl.load(file)
    #        model.load_state_dict(model_sd)
        
    # Save models & training info
    
    print(model.covar_module.base_kernel.lengthscale)
    model_dict[model_name + '_' + str(SEED)] = model
    noise_trace_dict[model_name + '_' + str(SEED)] = noise_trace
            
    X_train_mean = model.get_X_mean(Y_train)
    X_train_scales = model.get_X_scales(Y_train)
        
    #### Saving model with seed 
    #print(f'Saving {model_name} {SEED}')
    
    ####################### Testing Framework ################################################
    if TEST:
    #Compute latent test & reconstructions
        with torch.no_grad():
            model.eval()
            likelihood.eval()
        
            losses_test,  X_test = model.predict_latent(Y_train, Y_test, optimizer.defaults['lr'], 
                                  likelihood, SEED, prior_x=prior_x_test, ae=ae, 
                                  model_name=model_name,pca=pca, steps=7000)
            
        X_test_mean = X_test.q_mu.detach().cuda()
        Y_test_recon, Y_test_pred_covar = model.reconstruct_y(X_test_mean, Y_test, ae=ae, model_name=model_name)
        #Y_train_recon, Y_train_pred_covar = model.reconstruct_y(torch.Tensor(X_train_mean), Y_train, ae=ae, model_name=model_name)
        
        # ################################
        # # # Compute the metrics:
        from utils.metrics import *
        
        # 1) Reconstruction error - Train & Test
        
        #mse_train = rmse_missing(Y_train, Y_train_recon.T)
        mse_test = rmse_missing(Y_test.cpu(), Y_test_recon.T.detach().cpu())
        
        #print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
        print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
        