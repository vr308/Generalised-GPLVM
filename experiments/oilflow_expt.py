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
from sklearn.model_selection import train_test_split
plt.style.use('seaborn-muted')

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class OilFlowModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(OilFlowModel, self).__init__(X, q_f)
        
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
    
    # Setting seed for reproducibility
    SEED = 7
    torch.manual_seed(SEED)

    # Load some data
    
    N, d, q, X, Y, labels = load_real_data('oilflow')
    
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=50, random_state=SEED)
    lb_train, lb_test = train_test_split(labels, test_size=50, random_state=SEED)
    
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 10
    n_inducing = 25
    pca = True
    
    # Run all 5 models and store results
    
    #models = ['point','map','gauss','nn_gauss','iaf']
    models= ['point']
    model_dict = {}
    losses_dict = {}
    noise_trace_dict = {}
    
    for model_name in models:
        
        # Define prior for X
        X_prior_mean = torch.zeros(N, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        if pca == True:
              X_init = _init_pca(Y_train, latent_dim) # Initialise X to PCA 
        else:
              X_init = torch.nn.Parameter(torch.zeros(N, latent_dim))
        
        # Each inference model differs in its latent variable configuration / 
        # LatentVariable (X)
        
        # defaults - if a model needs them they are internally assigned
        nn_layers = None
        prior_x = None
        
        if model_name == 'point':
            
            ae = False
            X = PointLatentVariable(X_init)
            
        elif model_name == 'map':
                        
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = MAPLatentVariable(X_init, prior_x)
            
        elif model_name == 'gauss':
            
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = VariationalLatentVariable(N, data_dim, latent_dim, X_init, prior_x)
        
        elif model_name == 'nn_gauss':
            
            ae = True
            nn_layers = (5,3,2)
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
        elif model_name == 'iaf':
            
            ae = True
            nn_layers = (5,3,2)
            context_size = 5
            n_flows=2
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            X = IAFEncoder(N, latent_dim, context_size, prior_x, data_dim, nn_layers, n_flows)
            
        # Initialise model, likelihood, elbo and optimizer
        
        model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers)
        likelihood = GaussianLikelihood()
        elbo = VariationalELBO(likelihood, model, num_data=len(Y_train))
    
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
        
        iterator = trange(15000, leave=True)
        batch_size = 100
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gaussian']:
                sample = model.sample_latent_variable()  # a full sample returns latent x across all N
            else:
                sample = model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index]
            output_batch = model(sample_batch)
            loss = -elbo(output_batch, Y_train[batch_index].T).sum()
            loss_list.append(loss.item())
            noise_trace.append(np.round(likelihood.noise_covar.noise.item(),3))
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            
        # Save models & training info
        print(model.covar_module.base_kernel.lengthscale)
        model_dict[model_name] = model
        losses_dict[model_name] = loss_list
        noise_trace_dict[model_name] = noise_trace
        
        #Compute latent test & reconstructions
        model.eval()
        likelihood.eval()
        
        if ae is True:
            if model_name != 'iaf':
                X_test_mean, X_test_covar = model.predict_latent(Y_train, 
                                                                  Y_test, 
                                                                  optimizer.defaults['lr'], 
                                                                  likelihood, 
                                                                  prior_x=prior_x, 
                                                                  ae=True, 
                                                                  model_name='nn_gauss', 
                                                                  pca=pca)
            else:
                X_test_mean, X_flow_samples = model.predict_latent(Y_train, 
                                                                    Y_test, 
                                                                    optimizer.defaults['lr'], 
                                                                    likelihood, 
                                                                    prior_x = prior_x,
                                                                    ae=True, 
                                                                    model_name='iaf',
                                                                    pca=pca)
        
        else: # either point, map or gauss
            losses_test,  X_test = model.predict_latent(Y_train, Y_test, optimizer.defaults['lr'], 
                                      likelihood, prior_x=prior_x, ae=ae, 
                                      model_name=model_name,pca=pca)
                
        
        # # Compute training and test reconstructions
        
        # y_test_pred_mean, y_test_pred_covar = reconstruct_y(self, X_test_mean, Y_test, ae=ae, model_name=model_name)
        # y_train_pred_mean, y_train_pred_covar = reconstruct_y(self, X_train_mean, Y_train, ae=ae, model_name=model_name)

        # ################################
        # # # Compute the metrics:
        
        # # 1) Reconstruction error
        
        # mse_test_gaussian = metrics.mean_reconstruction_error(Y_test, Y_test_recon)
        # mse_test_flow = metrics.mean_reconstruction_error(Y_test, Y_test_flow_recon)
        
        # print(f'Reconstruction error {model_name} = ' + str(mse_test_gaussian))
        # print(f'Reconstruction error ' + model + '(with flows) = ' + str(mse_test_flow))
        
        # # # 2) ELBO Loss
        
        # print('Final -ELBO ' + model + '(no flows) = ' + str(losses[-1]))
        # print('Final -ELBO ' + model + '(with flows) = ' + str(losses_flow[-1]))
        
        # # 3) Negative Test log-likelihood
        
        # metrics.test_log_likelihood(gplvm, Y_test, test_dist)
        # metrics.test_log_likelihood(gplvm_flow, Y_test, test_flow_dist)
    
        # print('Final TLL ' + model + '(no flows) = ' + str(nll))
        # print('Final TLL ' + model + '(with flows) = ' + str(nll_flow))
        
        # metrics_df = pd.DataFrame(columns=['-elbo','mse','tll'], index=['Gaussian','Flows'])
        
        # metrics_df['-elbo'] = [losses[-1], losses_flow[-1]]
        # metrics_df['mse'] = [mse_test_gaussian.item(), mse_test_flow.item()]
        # metrics_df['nll'] = [np.mean(nll), np.mean(nll_flow)]
        # #metrics_df['se_nll'] = [np.std(nll)/np.sqrt(40), np.std(nll_flow)/np.sqrt(40)]
        
        # metrics_df.to_csv('metrics/metrics_' + model + '_' + dataset_name + '.csv')
            
    # # Plot result
    
    # plt.figure(figsize=(8, 6))
    # colors = ['r', 'b', 'g']
 
    # X = X_test.X.detach()
    # X = model.X.X.detach()
    # #X = model.X.q_mu.detach().numpy()
    # #X = model.X.mu(Y).detach().numpy()
    # #std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    
    # # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
    # for i, label in enumerate(np.unique(lb_train)):
    #     X_i = X[lb_train == label]
    #     #scale_i = std[labels == label]
    #     plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)
    #     #plt.errorbar(X_i[:, 1], X_i[:, 0], xerr=scale_i[:,1], yerr=scale_i[:,0], label=label,c=colors[i], fmt='none')
