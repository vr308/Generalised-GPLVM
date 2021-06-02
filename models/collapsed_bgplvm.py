#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collapsed Bayesian GPLVM in gpflow (Titsias 2010)

"""

import gpflow
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data import load_real_data 
from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    SEED = 37 # for reproducibility

    N, d, q, X, Y, labels = load_real_data('oilflow')
    
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=50, random_state=SEED)
    lb_train, lb_test = train_test_split(labels, test_size=50, random_state=SEED)
    
    Y_train = tf.constant(Y_train, dtype='float64')
    Y_test = tf.constant(Y_test, dtype='float64')
    
    # Setting shapes
    num_data = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 10
    num_inducing = 25
    pca = False
    
    X_mean_init = ops.pca_reduce(Y_train, latent_dim)
    X_var_init = tf.ones((num_data, latent_dim), dtype='float64')
    
    inducing_variable = tf.convert_to_tensor(
    np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype='float64')

    
    lengthscales = tf.convert_to_tensor([1.0] * latent_dim, dtype='float64')
    kernel = gpflow.kernels.RBF(lengthscales=lengthscales)
    
    gplvm = gpflow.models.BayesianGPLVM(
    Y_train,
    X_data_mean=X_mean_init,
    X_data_var=X_var_init,
    kernel=kernel,
    inducing_variable=inducing_variable,
)
    gplvm.likelihood.variance.assign(0.01)
    
    opt = gpflow.optimizers.Scipy()
    maxiter = ci_niter(1000)
    _ = opt.minimize(
        gplvm.training_loss,
        method="BFGS",
        variables=gplvm.trainable_variables,
        options=dict(maxiter=maxiter, disp=True),
        compile=True
    )
    print_summary(gplvm)
    
    #X_pca = ops.pca_reduce(Y, latent_dim).numpy()
    gplvm_X_mean = gplvm.X_data_mean.numpy()
    
    # f, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    # for i in np.unique(labels):
    #     ax[0].scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=i)
    #     ax[1].scatter(gplvm_X_mean[labels == i, 0], gplvm_X_mean[labels == i, 1], label=i)
    #     ax[0].set_title("PCA")
    #     ax[1].set_title("Bayesian GPLVM")

