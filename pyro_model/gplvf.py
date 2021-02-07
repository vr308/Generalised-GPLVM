#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLVF Model class 

GPLVF implements the Gaussian Process Latent Variable Flows Model,
    using EncoderFlow as a variational approximation to X|Y.

"""

from uuid import uuid4
import pyro
import torch
import numpy as np
from tqdm import trange
import pyro.contrib.gp as gp
import pyro.distributions as dist
from sklearn.decomposition import PCA
import pickle as pkl
from utils.data import float_tensor
from models.encoder_base import EncoderBase
from models.encoder_flow import EncoderFlow
from models.masked_likelihood import MaskedGaussian

class GPLVF:
    def __init__(self, Y, latent_dim):
        '''Initiates the model.

        Parameters
        ----------
        Y : torch.tensor
            Data, shape nxd.
        latent_dim : int
            Latent dimension.
        '''
        self._Y = Y
        self._n, self._d = Y.shape
        self._q = latent_dim

    def init_encoder(self, model='pca', num_flows=1, activate_flow=True,
                     flow_type='planar', nn_layers=(4, 2),
                     gp_inducing_num=10, flows=None):
        '''Initiates the base dist and flow classes.
        All parameters are passed to EncoderBase and EncoderFlow.

        Parameters
        ----------
        model : str
        num_flows : int
            Number of flows.
        activate_flow : bool
            If False, the flow is set to the identity transform (not
            implemented for custom flows via the 'flows' argument).
        flow_type : str
            One of 'sylvester', 'planar' or 'radial'.
        nn_layers : tuple
        gp_inducing_num : int
        flows : list
            List of relevant pyro.distributions.transforms.
        '''

        assert model in ('mf', 'pca', 'nn', 'gp')
        self.enc_base = EncoderBase(self._Y, self._q, model,
                                    nn_layers, gp_inducing_num)
        self.enc_flow = EncoderFlow(self._q, num_flows, self.enc_base,
                                    flow_type, flows, activate_flow)

    def _register_encoder_base(self):
        id = str(uuid4())
        for i in range(len(self.enc_flow.flows)):
            if hasattr(self.enc_flow.flows[i], 'parameters'):
                pyro.module(id + 'enc_flow_' + str(i), self.enc_flow.flows[i])
        pyro.module(id + 'enc_var_params', self.enc_base)
        # pyro.module(id + 'enc_flow_main', self.enc_flow)

    def encoder_model(self, idx, mask_idx=None):
        '''To be used as a 'guide' for SVI.'''
        # register all encoder params
        self._register_encoder_base()
        N = len(self._Y)
        with pyro.poutine.scale(scale=N / len(idx)):
            self.enc_flow.generate_base_dist(self.enc_base.model,
                                             self.enc_base.mu[idx],
                                             self.enc_base.sigma[idx],
                                             mask_idx)
            pyro.sample('X', self.enc_flow.flow_dist)
        return None

    def init_decoder(self, kernel=None, inducing_n=25, likelihood=None):
        '''Initiates the pyro forward/decoder vsgp.

        Parameters
        ----------
        kernel : pyro.contrib.gp.kernels instance
        inducing_n : int
            Number of inducing points for the decoder.
        '''

        Y_for_init = self._Y.numpy().copy()
        Y_for_init[np.isnan(Y_for_init)] = 0.0
        X_init = float_tensor(PCA(self._q).fit_transform(Y_for_init))

        X_inducing =\
            float_tensor(np.random.normal(size=(inducing_n, self._q)))

        if kernel is None:
            kernel = gp.kernels.Matern52(
                input_dim=self._q,
                lengthscale=torch.ones(self._q))

        base_args = dict(
            X=X_init,
            y=self._Y.T,
            kernel=kernel,
            Xu=X_inducing,
            jitter=1e-4)

        base_args['likelihood'] =\
            MaskedGaussian() if likelihood is None else likelihood
        self.decoder = gp.models.VariationalSparseGP(**base_args)

    def decoder_model(self, idx, mask_idx=None):
        '''To be used as the model with SVI.'''
        N  = len(self._Y)
        with pyro.poutine.scale(scale=N / len(idx)):
            mu_prior = torch.zeros(len(idx),self._q)
            X_minibatch = pyro.sample("X", dist.Normal(mu_prior, 1))  # prior
            y_minibatch = self._Y[idx]
        self.decoder.set_data(X_minibatch, y_minibatch.T)
        return self.decoder.model()

    def update_parameters(self):
        self.enc_base.update_parameters()
        # check if flow params need to be updated

    def _get_batch_idx(self, batch_size, train_len, test_mode=False):
        
        N = len(self._Y)
        #import pdb; pdb.set_trace()
        if test_mode:
            valid_indices = np.arange(train_len, N)
        else:
            valid_indices = np.array(range(N))
            
        batch_indices = np.random.choice(
            valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
            # return float_tensor(self._Y[batch_indices,:].copy())

    def y_given_x(self, X, mu_only=True):
        '''This returns the parameters of distribution (Y_new | X_new, ...).

        Parameters
        ----------
        mu_only : bool
            Return the mean only
        '''
        print('Remember to set, for example gplvm.decoder.X = X_recon ' +
              'BEFORE calling this function.')

        mu, sigma = self.decoder.forward(Xnew=X, full_cov=True)
        if mu_only:
            return mu.detach().numpy().T
        else:
            return mu, sigma

    def log_p_of_y_given_x(self, X, Y):
        '''Posterior predictive log probability.'''
        if Y.isnan().any():
            mu, sigma = self.decoder(Xnew=X, full_cov=False)
            sigma = sigma + 1e-4
            distribution = dist.MaskedDistribution(
                dist.Normal(mu, sigma.sqrt()), ~Y.T.isnan())
        else:
            mu, sigma = self.y_given_x(X, mu_only=False)
            noise = 1e-4*torch.eye(len(Y)).unsqueeze(0)
            sigma = sigma + noise
            distribution = dist.MultivariateNormal(mu, sigma)
        return distribution.log_prob(Y.T).detach().sum()

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        BUG: The train function needs to be run at least once so that
        all the necessary params are present when freezing.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.decoder.parameters():
            param.requires_grad_(trainable)

    def train(self, steps=1000, lr=0.01, n_elbo_samples=3, batch_size=20,
              train_len=None, test_mode=False):
        if batch_size is None:
            batch_size = len(self._Y)
        self.update_parameters()

        svi = pyro.infer.SVI(
            model=self.decoder_model,
            guide=self.encoder_model,
            optim=pyro.optim.Adam(dict(lr=lr)),
            loss=pyro.infer.Trace_ELBO(n_elbo_samples, retain_graph=True)
        )

        losses = np.zeros(steps)
        bar = trange(steps, leave=False)
        for step in bar:
            self.update_parameters()
            idx = self._get_batch_idx(batch_size, train_len, test_mode=test_mode)
            if train_len is not None:
                mask_idx = np.argmin(idx < train_len)
            else:
                mask_idx = None
            losses[step] = svi.step(idx, mask_idx)
            bar.set_description(str(int(losses[step])))
        return losses

    def forward(self, X, Xnew=None, full_cov=False):
        if Xnew is None:
            Xnew = X
        self.decoder.set_data(X=X)
        return self.decoder.forward(Xnew, full_cov)

    def get_X_Y_train_recon(self):
        self.enc_flow.generate_base_dist(self.enc_base.model, self.enc_base.mu,
                                         self.enc_base.sigma)
        X_recon = self.enc_flow.X_map(use_base_mu=True)
        Y_recon = self.forward(X_recon)[0].detach()
        return X_recon, Y_recon.T

    def predict(self, Y_test, mf_kern=None, mf_num_inducing=25, n_restarts=3,
                n_train_mf=2000, use_base_mu_mf=True, batch_size=100):
        '''Predict latent X_test for new Y_test

        Parameters
        ----------
        Y_test : torch.tensor
            Size n_test x d

        Returns
        -------
        flow_dist : TransformedDistribution instance
            This is the distribution of the predicted X_test.
        Y_test_reconstructed : numpy.array
            Reconstructed Y_test, using (approx) mode of the dist of X_test.
        mf_kern :
            kernel passed to init_decoder.
        mf_num_inducing : int
            num_inducing passed to init_decoder.
        n_restarts : int
            Number of optimizer restarts for X_map calc.
        n_train_mf : int
            Number of training steps for mf prediction.
        use_base_mu_mf : bool
            Use flow(base_params.loc) as an approximation of X_map.
        '''

        self.enc_flow.use_grads(False)
        self.use_grads(False)

        def stupid_copy(x): return pkl.loads(pkl.dumps(x))

        base_params = stupid_copy(self.enc_base)
        flow_module = stupid_copy(self.enc_flow)
        q = len(self.enc_base.mu.T)

        base_params._Y = Y_test

        if base_params.model == 'pca':
            base_params.mu =\
                float_tensor(base_params._pca.transform(Y_test))
        if base_params.model == 'mf':
            return self._predict_mf(Y_test, q, flow_module, mf_kern,
                                    mf_num_inducing, n_restarts, n_train_mf,
                                    use_base_mu_mf, batch_size)
        base_params.update_parameters()
        flow_module.generate_base_dist(
            base_params.model, base_params.mu, base_params.sigma)
        base_params.use_grads(False)
        flow_module.use_grads(False)

        _X = self.decoder.X
        _y = self.decoder.y

        X_recon = flow_module.X_map(n_restarts, use_base_mu_mf)

        self.decoder.X = X_recon
        self.decoder.y = Y_test.T

        Y_recon = self.y_given_x(X_recon, mu_only=True)

        self.decoder.X = _X
        self.decoder.y = _y
        self.enc_flow.use_grads(True)
        self.use_grads(True)

        return flow_module.flow_dist, Y_recon

    def _predict_mf(self, Y_test, q, flow_module, mf_kern, mf_num_inducing,
                    n_restarts, n_train_mf, use_base_mu_mf, batch_size):

        Y_full = torch.cat([self._Y, Y_test], dim=0)
        
        mu_full = torch.nn.Parameter(torch.cat([
            self.enc_base.mu.detach().clone(),
            torch.zeros((len(Y_test), q))
        ]))

        log_sigma_full = torch.nn.Parameter(torch.cat([
            self.enc_base._log_sigma.detach().clone(),
            torch.zeros((len(Y_test), q))
        ]))
        
        n_train = len(self._Y)

        base_params = EncoderBase(Y_full, q, 'mf')
        base_params.mu = torch.nn.Parameter(mu_full)
        base_params._log_sigma = torch.nn.Parameter(log_sigma_full)
        base_params.update_parameters()

        gplvm_test = GPLVF(Y_full, q)

        gplvm_test.enc_base = base_params
        gplvm_test.enc_flow = flow_module
        gplvm_test.init_decoder(mf_kern, mf_num_inducing)

        gplvm_test.enc_flow.use_grads(False)
        gplvm_test.enc_flow.generate_base_dist(
            base_params.model, base_params.mu, base_params.sigma,
            mask_idx=n_train)

        state = self.decoder.state_dict()
        params_of_int = [
            'Xu',
            'noise_unconstrained',
            'kernel.variance_unconstrained',
            'kernel.lengthscale_unconstrained',
            # for VariationalSparseGP
            'u_loc',
            'u_scale_tril_unconstrained',
            'likelihood.variance_unconstrained'
        ]

        state = {k: v for k, v in state.items() if k in params_of_int}

        gplvm_test.decoder.load_state_dict(state)
        
        gplvm_test.use_grads(False)
        losses_test = gplvm_test.train(n_train_mf, train_len=n_train, batch_size=batch_size, test_mode=True)

        base_params, flow_module = gplvm_test.enc_base, gplvm_test.enc_flow

        base_params.update_parameters()
        flow_module.generate_base_dist(
            base_params.model, base_params.mu, base_params.sigma,
            mask_idx=n_train)
        base_params.use_grads(False)
        flow_module.use_grads(False)

        _X = self.decoder.X
        _y = self.decoder.y

        X_recon = flow_module.X_map(n_restarts, use_base_mu_mf)

        self.decoder.X = X_recon
        self.decoder.y = Y_full.T

        Y_recon = self.y_given_x(X_recon, mu_only=True)

        self.decoder.X = _X
        self.decoder.y = _y
        self.enc_flow.use_grads(True)
        self.use_grads(True)

        return flow_module.flow_dist, Y_recon[n_train:], losses_test

    def get_mcmc_solution(self, q, n_mcmc=100):
        '''Inference of the latent variable X through MCMC (NUTS Sampler used)

        Parameters
        ----------
        q : int
            Latent variable (X) dimension.
        n_mcmc : int
            Number of MCMC steps (n_mcmc warmup + n_mcmc samples).

        Returns
        -------
        X_mcmc : torch.tensor
            MCMC samples of X.
        X_mcmc_mean : torch.tensor
            Mean of the samples of X.
        X_mcmc_flat : torch.tensor
            Flattened X_mcmc; (n_mcmc, n_test, q) -> (n_mcmc * n_test, q)
        '''
        Unif = dist.Uniform

        # prior on hyperparams
        self.decoder.base_model.kernel.lengthscale =\
            pyro.sample('lengthscale', Unif(torch.zeros(q), torch.ones(q)))
        self.decoder.base_model.kernel.variance =\
            pyro.sample('variance', Unif(torch.zeros(1), torch.ones(1)))

        X_init = float_tensor(PCA(q).fit_transform(self._Y))

        nuts = pyro.infer.NUTS(self.decoder_model)
        MCMC = pyro.infer.MCMC(
            kernel=nuts,
            num_samples=n_mcmc,
            warmup_steps=n_mcmc,
            num_chains=1,
            initial_params={'X': X_init})
        MCMC.run()

        X_mcmc = MCMC.get_samples()['X']
        return X_mcmc, X_mcmc.mean(axis=0), X_mcmc.reshape(-1, 2)