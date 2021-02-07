#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EncoderBase implements the parameters of base distribution of q(X), which is Gaussian.

"""
import torch
import numpy as np
import pyro.contrib.gp as gp
from sklearn.decomposition import PCA
from torch.nn.functional import relu
from utils.data import float_tensor

class EncoderBase(torch.nn.Module):
    ''' '''
    def __init__(self, Y, num_latent, model, nn_layers=(4, 2),
                 gp_inducing_num=10):
        '''Initiates the base variational distribution.

        Parameters
        ----------
        Y : torch.tensor
            Dataset for the autoencoder/backconstraint.
        num_data : int
        num_latent : int
        model : str
            One of 'mf' (meanfield), 'pca', 'nn' or 'gp'.
        nn_layers : tuple
            Defaults to (4, 2). Both neural networks (for the mean and covar)
            have the len(nn_layers) of hidden layers. Non-output layer
            activations are relu, outputs have a linear activation. The input
            shape is d = len(Y.T) for both networks.

            Mean NN: The number of nodes per hidden layer is specified in the
            tuple for the mean neural network. Output dimension is num_latent.

            Covar NN: The number of nodes for the output layer of this network
            is such that a dxd lower triangular matrix can be constructed from
            the output layer. The number of nodes in the hidden layers is an
            average between the input and output layers. The diagonal elements
            of the lower triangular dxd matrix have a relu activation.
        gp_inducing_num : int
            Defaults to 10. Number of inducing points for the encoder sparse
            gaussian process.
        '''

        super().__init__()

        num_data = len(Y)
        self._Y = Y
        self.model = model

        if model == 'mf':
            self._init_mean_field(num_data, num_latent)
        elif model == 'pca':
            self._init_pca_param(num_latent)
        elif model == 'nn':
            self._init_nn(d=len(Y.T), q=num_latent, nn_layers=nn_layers)
        elif model == 'gp':
            self._init_gp(num_latent, gp_inducing_num)
        else:
            raise NotImplementedError('Invalid encoder/backconstraint.')

        self.update_parameters()

    def update_parameters(self):
        '''Updates mu and sigma, which are dependent on model parameters.'''
        if self.model == 'gp':
            self._update_gp_params()
        elif self.model == 'nn':
            self._update_nn_params()
        else:
            self._update_sigma()

    ''' ----------Meanfield and PCA methods---------- '''

    def _init_mean_field(self, n, q):
        Y_for_init = self._Y.numpy().copy()
        Y_for_init[np.isnan(Y_for_init)] = 0.0
        mu = PCA(q).fit_transform(Y_for_init)
        mu = mu + np.random.normal(scale=0.1, size=mu.shape)
        self.mu = torch.nn.Parameter(float_tensor(mu))

        log_sigma = float_tensor(np.zeros((n, q)))
        self._log_sigma = torch.nn.Parameter(log_sigma)

    def _init_pca_param(self, q):
        self._pca = PCA(q)
        self._pca.fit(self._Y)
        mu = self._pca.transform(self._Y)
        self.mu = float_tensor(mu)

        log_sigma = float_tensor(np.zeros(q))
        self._log_sigma = torch.nn.Parameter(log_sigma)

    def _update_sigma(self):
        self.sigma = self._log_sigma.exp()

    ''' ----------Neural network encoder methods---------- '''

    # Look into layer = torch.nn.Linear(nn_layers[i], nn_layers[i + 1]) and
    # nn.ModuleList([nn.Linear...

    def _init_nn(self, d, q, nn_layers):
        n_hidden = len(nn_layers)
        mu_layers = (d,) + nn_layers + (q,)

        # tril_indices help convert from a flat vector to a lower-tri matrix
        self._tril_indices = torch.tril_indices(row=q, col=q, offset=0)
        sg_output_len = len(self._tril_indices.T)
        num_nodes_sigma = (d + sg_output_len)//2
        sg_layers = (d,) + (num_nodes_sigma,)*n_hidden + (sg_output_len,)

        def param(shape):
            return torch.nn.Parameter(
                float_tensor(np.random.uniform(size=shape)))

        max_iter = len(mu_layers) - 1
        self._nn = {}  # nn parameters
        for i in range(max_iter):
            # layer output = relu(w*input + b). Declare w and b here:
            _i = str(i)
            self._nn['mu_w' + _i] = param((mu_layers[i], mu_layers[i + 1]))
            self._nn['mu_b' + _i] = param(mu_layers[i + 1])
            self._nn['cov_w' + _i] = param((sg_layers[i], sg_layers[i + 1]))
            self._nn['cov_b' + _i] = param(sg_layers[i + 1])

            self.register_parameter('mu_w' + _i, self._nn['mu_w' + _i])
            self.register_parameter('mu_b' + _i, self._nn['mu_b' + _i])
            self.register_parameter('cov_w' + _i, self._nn['cov_w' + _i])
            self.register_parameter('cov_b' + _i, self._nn['cov_b' + _i])

    def _update_nn_params(self):
        max_iter = len(self._nn)//4
        pred_mu = self._Y
        pred_sg = self._Y
        for i in range(max_iter):
            w = self._nn['mu_w' + str(i)]
            b = self._nn['mu_b' + str(i)]
            non_lin = relu if i != (max_iter - 1) else lambda x: x
            pred_mu = non_lin(pred_mu@w + b)

            w = self._nn['cov_w' + str(i)]
            b = self._nn['cov_b' + str(i)]
            pred_sg = non_lin(pred_sg@w + b)
        self.mu = pred_mu

        n = len(pred_mu)
        q = len(pred_mu.T)

        # set diagonal to be positive
        diag_elems = ~np.logical_xor(
            self._tril_indices[0, :], self._tril_indices[1, :]).bool()
        pred_sg[:, diag_elems] = relu(pred_sg[:, diag_elems]) + 1e-10

        self.sigma = torch.zeros((n, q, q))
        self.sigma[:, self._tril_indices[0], self._tril_indices[1]] = pred_sg

        jitter = torch.eye(q).unsqueeze(0)*1e-4
        jitter = torch.cat([jitter for i in range(n)], axis=0)
        self.sigma += jitter
        # self.sigma is the cholesky factor of the covariance

    ''' ----------Sparse GP encoder methods---------- '''

    def _init_gp(self, num_latent, inducing_num):
        d = len(self._Y.T)
        self.gps = {}

        # one GP for every latent dimension
        for i in range(num_latent):
            X_inducing =\
                float_tensor(np.random.normal(size=(inducing_num, d)))
            kernel = gp.kernels.Matern52(
                input_dim=d,
                lengthscale=torch.ones(d))
            gp_model = gp.models.VariationalSparseGP(
                X=self._Y,
                y=None,
                kernel=kernel,
                Xu=X_inducing,
                likelihood=gp.likelihoods.Gaussian())

            # register gp parameters
            self.gps['model_' + str(i)] = gp_model
            self._register_module(gp_model, i)

    def _update_gp_params(self):
        n = len(self._Y)
        jitter = torch.eye(n).reshape(n, n)*1e-4

        # need to concatenate the different gps across latent dimensions
        gp_mu_sigmas =\
            [gp.forward(self._Y, full_cov=True) for _, gp in self.gps.items()]

        mu = [gp_mu_sigmas[i][0].reshape(-1, 1) for i in range(len(self.gps))]
        self.mu = torch.cat(mu, axis=1)

        # sigma_proper = ... matrix.cholesky().reshape(1, n, n) ... axis=0)
        sigmas = [gp_mu_sigmas[i][1] + jitter for i in range(len(self.gps))]
        sigmas = [matrix.diag().sqrt().reshape(-1, 1) for matrix in sigmas]
        sigma_fitc = torch.cat(sigmas, axis=1)
        self.sigma = sigma_fitc

    ''' ----------Internals---------- '''

    def _register_module(self, module, i=''):
        for (name, param) in module.named_parameters():
            self.register_parameter(name.replace('.', '_') + str(i), param)

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.parameters():
            param.requires_grad_(trainable)

