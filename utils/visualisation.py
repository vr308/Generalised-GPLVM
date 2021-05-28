#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for visualisation 

"""

import matplotlib.pylab as plt
import seaborn as sns
import pathlib
import torch 
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
plt.style.use('ggplot')


def plot_quad_latents(models, Y_trains, labels, colors, plot_order, model_names, titles):
    
    plt.figure(figsize=(13,3))
    
    j = 0
    
    for model, label, Y_train, title, m_name in zip(models, labels, Y_trains, titles, model_names):
    
        inv_lengthscale = 1/model.covar_module.base_kernel.lengthscale
        #print(inv_lengthscale)
                
        values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)
        
        l1 = indices.numpy().flatten()[0]
        l2 = indices.numpy().flatten()[1]
        
        if m_name in ('point', 'map'):
            X_mean = model.X.X.detach().numpy()
            X_scales = None
        elif m_name == 'gauss':
            X_mean = model.X.q_mu.detach().numpy()
            X_scales = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
        elif m_name == 'nn_gauss':
            model.X.jitter = 1e-5
            X_mean = model.X.mu(torch.Tensor(Y_train)).detach().numpy()
            X_scales = np.array([torch.sqrt(x.diag()).numpy() for x in model.X.sigma(torch.Tensor(Y_train)).detach()])
        
        print(m_name)
        print(j)
        plt.subplot(1,4, j+1)
        #plt.title(f'2d latent subspace [{model_name}]',fontsize='small')
        plt.xlabel(r'$x_{1}$', fontsize='medium')
        plt.ylabel(r'$x_{2}$', fontsize='medium')
        # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
        for i, l in enumerate(np.unique(label)):
            X_i = X_mean[label == l]
            plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], s=1, label=label)
            if X_scales is not None:
                scale_i = X_scales[label == l]
                plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:,l1], yerr=scale_i[:,l2], label=label,c=colors[i], fmt='none')
        plt.title(title)
        j = j+1
        
def plot_lengthscales(models, Y_trains, labels, colors, plot_order, model_names, titles):
    
    
    plt.figure(figsize=(13,3))
    j = 0
    for model, label, Y_train, title, m_name in zip(models, labels, Y_trains, titles, model_names):
    
        inv_lengthscale = 1/model.covar_module.base_kernel.lengthscale
        #print(inv_lengthscale)
        
        latent_dim = model.X.latent_dim
                
        values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)
        
        plt.subplot(1,4,j+1)
        plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().numpy().flatten(), color='teal')
        #plt.title('Inverse lengthscale with SE-ARD kernel (10d)', fontsize='small')
        plt.xlabel('Latent dims', fontsize='medium')
        plt.xticks(np.arange(0,10))
        j = j+1
    plt.suptitle('Inverse Lengthscales SE-ARD Kernel (10d)')
    

def plot_train_test_error_elbo_report():
    
    import pandas as pd
    
    train = pd.read_csv('plots/train_rmse.txt', sep=', ')
    test = pd.read_csv('plots/test_rmse.txt', sep=', ')
    elbos = pd.read_csv('plots/elbos.txt', sep=', ')

    plt.style.use('ggplot')
    
    fig = plt.figure(figsize=(7,3))
    colors = ['r', 'b', 'orange', 'magenta']
    datasets = ['oilflow', 'qpcr']
    titles = ['Oilflow', 'qPCR']
    model_cols = ['point_', 'map_', 'gauss_', 'nn_gauss_']
    names = ['Point','MAP','B-SVI','AEB-SVI']
    for i in range(2): # iterate over datasets
        ax = fig.add_subplot(1,2, i + 1)
        df_train = train[train.dataset == datasets[i]]
        df_test = test[test.dataset== datasets[i]]
        for m in range(4): # iterate over models within each subplot
            mean_tr = df_train[[col for col in df_train if col.startswith(model_cols[m])][0]].item()
            mean_te = df_test[[col for col in df_test if col.startswith(model_cols[m])][0]].item()
            se_tr = df_train[[col for col in df_train if col.startswith(model_cols[m])][1]].item()
            se_te = df_test[[col for col in df_test if col.startswith(model_cols[m])][1]].item()
            ax.errorbar(y=[mean_tr, mean_te], x=[1,1.2], xerr=[se_tr, se_te], c=colors[m], marker='o', barsabove=True, capsize=4, fmt='-', label=names[m])
            ax.set_title(titles[i], fontsize='small')
            ax.set_xticks([1,1.2])
            ax.set_xticklabels(['Train', 'Test'])
        ax.legend()
        
    
    fig = plt.figure(figsize=(7,3))
    colors = ['r', 'b', 'orange', 'magenta']
    datasets = ['oilflow', 'qpcr']
    titles = ['Oilflow', 'qPCR']
    model_cols = ['point_', 'map_', 'gauss_', 'nn_gauss_']
    names = ['Point','MAP','B-SVI','AEB-SVI']
    for i in range(2): # iterate over datasets
        ax = fig.add_subplot(1,2, i + 1)
        df = elbos[elbos.dataset == datasets[i]]
        for m in range(4): # iterate over models within each subplot
            mean = df[[col for col in df if col.startswith(model_cols[m])][0]].item()
            se = df[[col for col in df if col.startswith(model_cols[m])][1]].item()
            ax.errorbar(y=m, x=mean, xerr=se, c=colors[m], marker='o', barsabove=True, capsize=4, fmt='-', label=names[m])
            ax.set_title(titles[i], fontsize='small')
            ax.set_yticks([0,1,2,3])
            ax.set_yticklabels(names)
            ax.set_xlabel('Neg. ELBO Loss', fontsize='small')
    ax.set_xscale('log')
    ax.legend()
        
    
        

def plot_report(model, losses, labels, colors, save, X_mean, X_scales=None, model_name=None):
    
    plt.figure(figsize=(12,4))
    
    inv_lengthscale = 1/model.covar_module.base_kernel.lengthscale
    print(inv_lengthscale)
    
    latent_dim = model.X.latent_dim
    
    values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2,largest=False)
    
    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    plt.subplot(131)
    #sn_samples = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((2000,))
    #sns.kdeplot(sn_samples[:,0], sn_samples[:,1], shade=True, bw=4, color='gray')    #X = model.X.q_mu.detach().numpy()
    #std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    plt.title(f'2d latent subspace [{model_name}]',fontsize='small')
    plt.xlabel('Latent dim 1', fontsize='small')
    plt.ylabel('Latent dim 2', fontsize='small')
    # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales 
    for i, label in enumerate(np.unique(labels)):
        X_i = X_mean[labels == label]
        plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
        if X_scales is not None:
            scale_i = X_scales[labels == label]
            plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:,l1], yerr=scale_i[:,l2], label=label,c=colors[i], fmt='none')
        
    plt.subplot(132)
    plt.bar(np.arange(latent_dim), height=inv_lengthscale.detach().numpy().flatten(), color='teal')
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')
    plt.xlabel('Latent dims', fontsize='small')
    
    plt.subplot(133)
    plt.plot(losses,label='batch_size=100', color='orange')
    plt.xlabel('Steps', fontsize='small')
    plt.title('Neg. ELBO Loss', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(pathlib.Path().absolute()/f'plots/{save}.png')

def plot_grid_images(dataset_name, Y): 
    '''Plots grid of images, either 'brendan_faces' or 'mnist'

    Parameters
    ----------
    dataset_name : str
    Y : array_like or None

    '''
    
    if dataset_name == 'brendan_faces':
        reshape_params = (28,20)
    elif dataset_name == 'mnist':
        reshape_params = (28, 28)
    
    fig = plt.figure(figsize=(4,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
             nrows_ncols=(5, 5),  # creates grid of axes
             axes_pad=0.01,  # pad between axes in inch.
             )
    with plt.style.context('grayscale'):
        for ax, im in zip(grid, Y):
            # Iterating over the grid returns the Axes.
            ax.imshow(im.reshape(reshape_params[0], reshape_params[1]))
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=0, length = 0)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.tight_layout()

        
def plot_single_image(im):
    
    # Assuming im is already shaped correctly to render image
     plt.figure(figsize=(4,4))
     with plt.style.context('grayscale'):
            plt.imshow(im)
            plt.grid(False)
            plt.tick_params(axis='both', labelsize=0, length = 0)

def plot_elbos(losses, losses_flow, title):
    
    plt.plot(losses, label='Gaussian')
    plt.plot(losses_flow, label='Flow')
    plt.title('ELBO Loss ' + title, fontsize='small')
    plt.xlabel('Iterations', fontsize='small')
    plt.ylabel('-ELBO', fontsize='small')
    plt.legend(fontsize='small')
    

def plot_y_reconstruction(X, Y, Y_recon=None, title=''):
    
    '''Plots reconstructed Y corresponding to true X in 3d

    Parameters
    ----------
    X : array_like or None
    Y : array_like or None
    Y_recon : array_like
        Y_recon is overlaid on Y
    title : str

    '''

    Y = Y[:, :6]
    if Y_recon is not None:
        Y_recon = Y_recon[:, :6]

    fig = plt.figure(figsize=(12, 4))
    for i in range(len(Y.T)):
        ax = fig.add_subplot(1, len(Y.T), i+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Y[:, i])
        if Y_recon is not None:
            ax.scatter(X[:, 0], X[:, 1], Y_recon[:, i])
    plt.suptitle(title)
    
def plot_synthetic_2d_X(X, lb, title):
    
    '''Plots latent 2d X

    Parameters
    ----------
    X : array_like or None
    lb : array
    title : str
    '''
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=lb)
    plt.title(title)
    

def plot_2d_learnt_X_dist(test_dist, test_flow_dist, lb_test, class_label):
    
    '''Plots latent 2d X (a corpus of latent dists on one plot)

    Parameters
    ----------
    test_dist : array
    test_flow_dist : array
   
    '''
    index = np.where(lb_test == class_label)[0] #  index : int list of latent encodings to plot dists of
    gaussian_dist = test_dist.sample_n(1000)[:,index,:]
    flow_dist = test_flow_dist.sample_n(1000)[:,index,:]
    
    plt.figure(figsize=(8,4))
    for i in np.arange(len(index)):
        plt.subplot(121)
        sns.kdeplot(gaussian_dist[:,i,:][:,0], gaussian_dist[:,i,:][:,1], shade=True, color='b', bw=2)
        plt.xlabel('Latent dimension 1', fontsize='small')
        plt.ylabel('Latent dimension 2', fontsize='small')
        plt.title('Gaussian q(X)', fontsize='small')
        plt.subplot(122)
        sns.kdeplot(flow_dist[:,i,:][:,0], flow_dist[:,i,:][:,1], shade=True, color='g', alpha=0.6, bw=0.4)
        plt.title('Flow q(X)', fontsize='small')
        plt.xlabel('Latent dimension 1', fontsize='small')
        plt.ylabel('Latent dimension 2', fontsize='small')
    
    plt.suptitle('Distribution of 2d latent embeddings', fontsize='small')
    
def plot_latent_X_clusters(X_recon_train_base, X_recon_train_flow, lb, left_title, right_title, suptitle):
    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title(left_title,fontsize='small')
    plt.scatter(X_recon_train_base[:,0], X_recon_train_base[:,1], c=lb, s=2)
    plt.xlabel('Latent dimension 1', fontsize='small')
    plt.ylabel('Latent dimension 2', fontsize='small')
    plt.subplot(122)
    plt.scatter(X_recon_train_flow[:,0], X_recon_train_flow[:,1], c=lb, s=2)
    plt.xlabel('Latent dimension 1', fontsize='small')
    plt.ylabel('Latent dimension 2', fontsize='small')
    plt.title(right_title, fontsize='small')
    plt.suptitle(suptitle, fontsize='small')

    
def plot_single_learnt_flow(gplvm_flow, title):
    
    '''Plots latent 2d X

    Parameters
    ----------
    gplvm_flow: of class GPLVF  
    title: str
    '''
    gplvm_flow.enc_flow.plot_flow()
    plt.title(title, fontsize='small')
    
def plot_latent_quiver_plot(X_recon_base, X_recon_flow, lb, no_labels, title):
    
    '''Plots latent 2d X quiver showing transformation from base gaussian to non-Gaussian map

    Parameters
    ----------
    X_recon_base: 2d array
    X_recon_flow: 2d array
    lb: index for label colors
    title : str
    '''
    if no_labels:
        c_base = 'b'; c_flow = 'r'
    else:
        c_base = lb; c_flow = lb
        
    #plt.figure(figsize=(8,8))
    plt.scatter(x=X_recon_base[:,0], y=X_recon_base[:,1], s=3, c=c_base, marker='o', label='Gaussian latent encoding')
    plt.scatter(x=X_recon_flow[:,0], y=X_recon_flow[:,1], s=3, c=c_flow, marker='o', label='Flow based latent encoding')
    plt.quiver(X_recon_base[:,0], X_recon_base[:,1], X_recon_flow[:,0] - X_recon_base[:,0], X_recon_flow[:,1] - X_recon_base[:,1], headwidth=5,scale_units='xy',angles='xy',scale=1, alpha=0.4, width=0.001)
    plt.legend(fontsize='small')
    plt.title(title, fontsize='small')
    

   
    
    
    
    