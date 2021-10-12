#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Missing Test framework 

Re-loads pre-saved pkl models to conduct testing

"""

import torch, os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

import pickle as pkl
import gc

from gpytorch.priors import NormalPrior
from experiments.mnist_expt_train import GPLVM
from utils.data import load_real_data
from models.likelihoods import GaussianLikelihoodWithMissingObs

def get_Y_test_missing(Y_full, lb, N_test, percent):
    
    idx = np.random.binomial(n=1, p=percent, size=(N_test, Y_full.shape[1])).astype(bool)
    test_idx = np.random.randint(0, Y_full.shape[0], 1000)
    Y_test_missing = Y_full.clone()[test_idx]
    Y_test_missing[idx] = np.nan
    lb_test = lb[test_idx]
    return Y_test_missing, lb_test

if __name__ == '__main__':

    SEED = 40
    torch.manual_seed(SEED)
    model_name = 'gauss'

    n, d, q, X, Y, lb = load_real_data('mnist')
    q = 6; Y /= 255
    #Y = Y[:5000]; n = len(Y)
    
    Y = Y[np.isin(lb, (1, 7)), :]
    lb = lb[np.isin(lb, (1, 7))]
    n = len(Y)

    # remove some obs from Y
    Y_full = Y.clone()
    
    #### Prepare Y_test data with a % of missing 

    N_test = 1000
    Y_test_missing, lb_test = get_Y_test_missing(Y_full, lb, N_test, percent=0.5)
    
    #### Load pre-saved model
    X_init = torch.nn.Parameter(torch.zeros(n, q))
    
    model = GPLVM(n, d, q, n_inducing=120, X_init=X_init)
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    #### choose one from mnist_full / mnist_10 / mnist_30 / mnist_60
    
    if os.path.isfile('pre_trained_models/mnist_full.pkl'):
      with open('pre_trained_models/mnist_full.pkl', 'rb') as file:
          model_sd, likl_sd = pkl.load(file)
          model.load_state_dict(model_sd)
          likelihood.load_state_dict(likl_sd)
          
    if torch.cuda.is_available():
         device = 'cuda'
         model = model.cuda()
         likelihood = likelihood.cuda()
         Y_test_missing.to(device)
    else:
         device = 'cpu'
         
    #### Testing framework 
    
    with torch.no_grad():
        
             model.eval()
             likelihood.eval()
             optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
             
             X_prior_mean_test = torch.zeros(N_test, q)
             prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test))

             losses_test,  X_test = model.predict_latent(Y.to(device), Y_test_missing.to(device), optimizer.defaults['lr'], 
                         likelihood, SEED, prior_x=prior_x_test.to(device), ae=False, 
                         model_name=model_name,pca=False)
    
    torch.cuda.empty_cache()
    gc.collect()
                             
    #### Compute training and test reconstructions
    
    # if model_name in ('point', 'map'):
    #         X_test_mean = X_test.X
    # elif model_name == 'gauss':
    #        X_test_mean = X_test.q_mu.detach().to(device)
   
    # Y_test_recon, Y_test_pred_covar = model.reconstruct_y(X_test_mean, Y_test_missing, ae=False, model_name=model_name)
    # #Y_train_recon, Y_train_pred_covar = model.reconstruct_y(torch.Tensor(X_train_mean), Y_train, ae=ae, model_name=model_name)
    
    # # ################################
    # # Compute the metrics:
        
    # from utils.metrics import *
    
    # # 1) Reconstruction error - Train & Test
    
    # mse_train = rmse(Y_train, Y_train_recon.T)
    # mse_test = rmse(Y_test, Y_test_recon.T)
    
    # print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
    # print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
    
    # if os.path.isfile('for_paper/mnist_full.pkl'):
    #     with open('for_paper/mnist_full.pkl', 'rb') as file:
    #         model_sd, likl_sd = pkl.load(file)
    #         model.load_state_dict(model_sd)
    #         likelihood.load_state_dict(likl_sd)