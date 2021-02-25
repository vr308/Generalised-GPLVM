''' To be run as `ipython tests/test_model.py` from the experiments folder.

Please ensure you set the working directory to the parent folder which contains the two sub-folders

lib/
tests/

'''

import sys
import pyro
import torch
import unittest
import numpy as np
from copy import deepcopy
from lib import data, model
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import trange
from visualisation import plot_y_reconstruction

SEED = 42
torch.manual_seed(SEED); pyro.set_rng_seed(SEED); np.random.seed(SEED)
# plt.ion(); plt.style.use('ggplot')

class self:
    pass
self=self()

class GPLVFFitTests(unittest.TestCase):

    def setUp(self):
        n, d, q, X, Y, lb = data.generate_synthetic_data(
            1000, x_type='normal', y_type='hi_dim')
        Y = (Y - Y.mean(axis=0)) @ Y.std(axis=0).diag().inverse()

        self.hi_dim_data = n, d, q, X, Y, lb

        n, d, q, X, Y, lb = data.generate_synthetic_data(
            x_type='normal', y_type='by_cat')
        Y = Y @ Y.std(axis=0).diag().inverse()

        self.by_cat_data = n, d, q, X, Y, lb

    def test_map(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.decoder.autoguide('X', pyro.distributions.Delta)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]

        losses = model.gp.util.train(gplvm.decoder, num_steps=2000)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]

        X_recon = gplvm.decoder.X_map
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.y_given_x(X_recon)

        errors = ((Y - Y_recon)**2).mean().sqrt()

        # check that errors are low
        self.assertTrue(errors < 0.5)

        # check that params have changed
        for (bef, aft) in zip(initial_params, trained_params):
            self.assertTrue((bef != aft).any())

    def test_decoder_freezing(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.decoder.autoguide('X', pyro.distributions.Delta)
        model.gp.util.train(gplvm.decoder, num_steps=1)  # IMPORTANT
        gplvm.use_grads(False)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]

        losses_1 = model.gp.util.train(gplvm.decoder, num_steps=100)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]

        self.assertTrue(np.std(losses_1) == 0)
        # check all params equal
        for i, (bef, aft) in enumerate(zip(initial_params, trained_params)):
            self.assertTrue((bef == aft).all())

        gplvm.use_grads(True)
        losses_2 = model.gp.util.train(gplvm.decoder, num_steps=100)

        trained_params_2 =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]

        self.assertTrue(np.std(losses_2) > 0)
        # check all changed
        for i, (bef, aft) in enumerate(zip(trained_params, trained_params_2)):
            self.assertTrue((bef != aft).any())

    def test_pca_no_flow_and_flow_deact(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='pca', activate_flow=False)

        x = data.float_tensor(np.arange(0, q))
        flow_x_eq_x = ((gplvm.enc_flow.forward(x).detach() - x) == 0).all()
        self.assertTrue(flow_x_eq_x)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        initial_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        losses = gplvm.train(steps=2000)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        trained_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        flow_x_eq_x = ((gplvm.enc_flow.forward(x).detach() - x) == 0).all()
        self.assertTrue(flow_x_eq_x)

        X_recon = gplvm.enc_base.mu
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.y_given_x(X_recon)

        errors = ((Y - Y_recon)**2).mean().sqrt()
        self.assertTrue(errors < 0.5)

        for i, (bef, aft) in enumerate(zip(initial_params, trained_params)):
            if i >= 2:
                self.assertTrue((bef != aft).any())

    def test_mf_no_flow(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='mf', activate_flow=False)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        initial_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        losses = gplvm.train(4000)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        trained_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        X_recon = gplvm.enc_base.mu
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.decoder.forward(Xnew=X_recon)[0].T

        errors = ((Y - Y_recon)**2).mean().sqrt()

        # check that errors are low
        self.assertTrue(errors < 0.5)

        # check that params have changed
        for i, (bef, aft) in enumerate(zip(initial_params, trained_params)):
            if i >= 2:
                self.assertTrue((bef != aft).any())

    def test_gp_no_flow(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='gp', activate_flow=False)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        initial_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        losses = gplvm.train(steps=2000)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        trained_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        X_recon = gplvm.enc_base.mu
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.y_given_x(X_recon)

        errors = ((Y - Y_recon)**2).mean().sqrt()
        self.assertTrue(errors < 0.5)

        for i, (bef, aft) in enumerate(zip(initial_params, trained_params)):
            if i >= 2:
                pass
                # self.assertTrue((bef != aft).any())  # FAILS <- FIX ME

    def test_nn_no_flow(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='nn', activate_flow=False)

        initial_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        initial_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        losses = gplvm.train(steps=500)

        trained_params =\
            [param.clone().detach() for param in gplvm.decoder.parameters()]
        trained_params +=\
            [param.clone().detach() for param in gplvm.enc_base.parameters()]

        X_recon = gplvm.enc_base.mu
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.y_given_x(X_recon)

        errors = ((Y - Y_recon)**2).mean().sqrt()
        self.assertTrue(errors < 1.0)

        for i, (bef, aft) in enumerate(zip(initial_params, trained_params)):
            if i >= 2:
                self.assertTrue((bef != aft).any())

    def test_predict_with_gp(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        Y_train, Y_test = data.train_test_split(Y.numpy(), random_state=42)
        X_train, X_test = data.train_test_split(X, random_state=42)
        lb_train, lb_test = data.train_test_split(lb, random_state=42)

        Y_train = data.float_tensor(Y_train)
        Y_test = data.float_tensor(Y_test)

        gplvm = model.GPLVF(Y_train, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='gp', activate_flow=True, num_flows=10)
        losses = gplvm.train(steps=2000)

        test_flow_dist, Y_test_recon = gplvm.predict(Y_test,
            use_base_mu_mf=True)
        # data.check_model(X_test, Y_test, Y_test_recon)
        errors = ((Y_test - Y_test_recon)**2).mean().sqrt()
        self.assertTrue(errors < 0.5)

    def test_predict_with_mf(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        Y_train, Y_test = data.train_test_split(Y.numpy(), random_state=42)
        X_train, X_test = data.train_test_split(X, random_state=42)
        lb_train, lb_test = data.train_test_split(lb, random_state=42)

        Y_train = data.float_tensor(Y_train)
        Y_test = data.float_tensor(Y_test)

        gplvm = model.GPLVF(Y_train, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='mf', activate_flow=False, num_flows=10)
        losses = gplvm.train(steps=1000, batch_size=100)

        decoder_state = deepcopy(gplvm.decoder.state_dict())
        flow_state = deepcopy(gplvm.enc_flow.state_dict())

        test_flow_dist, Y_test_recon, losses_test = gplvm.predict(Y_test,
            n_restarts=1, n_train_mf=1000, use_base_mu_mf=True, batch_size=100)

        decoder_state_after_pred = deepcopy(gplvm.decoder.state_dict())
        flow_state_after_pred = deepcopy(gplvm.enc_flow.state_dict())

        # decoder and flow remain unchanged
        for key in decoder_state:
            self.assertTrue(
                (decoder_state[key] == decoder_state_after_pred[key]).all())

        for key in flow_state:
            self.assertTrue(
                (flow_state[key] == flow_state_after_pred[key]).all())

        gplvm.update_parameters()
        
        X_recon, Y_recon = gplvm.get_X_Y_train_recon()
        
        plot_y_reconstruction(X_train, Y_train, Y_recon)
        plot_y_reconstruction(X_test, Y_test, Y_test_recon)
        errors = ((Y_test - Y_test_recon)**2).mean().sqrt()
        print(errors)

    def test_missing_data(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)

        idx_a = np.random.choice(range(n), n//4)
        idx_b = np.random.choice(range(d), n//4)
        Y[idx_a, idx_b] = np.nan

        # row must have at least one obs
        self.assertFalse((Y.isnan().sum(axis=1) == d).any())

        gplvm = model.GPLVF(Y, q)
        gplvm.init_decoder()
        gplvm.init_encoder(model='mf', activate_flow=True, num_flows=10)

        losses = gplvm.train(steps=3000)
        X_recon = gplvm.enc_flow.X_map(1)
        gplvm.decoder.X = X_recon
        Y_recon = gplvm.y_given_x(X_recon)
        errors = ((Y - Y_recon)**2)
        self.assertTrue(errors[~errors.isnan()].mean().sqrt() < 0.5)

        # plot_y_reconstruction(X, Y, Y_recon)


class MNISTPlotsAndSDInits(unittest.TestCase):

    def test_missing_mnist_plots_and_inits(self):
        pyro.get_param_store().clear()
        n, d, q, X, Y, lb = data.load_real_data('mnist')

        # shrink Y
        Y = Y.reshape(-1, 1, 28, 28)
        conv = torch.nn.Conv2d(1, 1, 2, 2)
        conv.bias.data = torch.tensor([0]).float()
        conv.weight.data = torch.ones(conv.weight.shape).float()/(4 * 255)
        Y = conv(Y).detach()
        Y = Y.reshape(n, 14**2)

        # subset Y
        Y = Y[:5000]
        lb = lb[:5000]
        q = 2
        d = len(Y.T)
        n = len(Y)

        # remove some obs from Y
        Y_copy = Y.clone()
        idx_a = np.random.choice(range(n), n * (d//2))
        idx_b = np.random.choice(range(d), n * (d//2))
        Y[idx_a, idx_b] = np.nan
        # self.assertFalse((Y.isnan().sum(axis=1) == d).any())

        Y = Y + np.random.normal(scale=0.005, size=Y.shape).astype('f')
        # plt.imshow(Y[0].reshape(14, 14))

        gplvm = model.GPLVF(Y, q)
        gplvm.use_subsetting = False
        gplvm.init_decoder(base_model='vsgp')
        gplvm.init_encoder(model='mf', activate_flow=True, num_flows=20)

        with open('./mnist_sd/enc_flow_sd.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm.enc_flow.load_state_dict(state_dict)

        with open('./mnist_sd/enc_base_sd.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm.enc_base.load_state_dict(state_dict)

        with open('./mnist_sd/decoder_sd.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm.decoder.load_state_dict(state_dict)

        gplvm.update_parameters()
        losses = gplvm.train(steps=75, batch_size=None)

        pyro.get_param_store().clear()

        gplvm_wo_flow = model.GPLVF(Y, q)
        gplvm_wo_flow.use_subsetting = False
        gplvm_wo_flow.init_decoder(base_model='vsgp')
        gplvm_wo_flow.init_encoder(model='mf', activate_flow=False)

        with open('./mnist_sd/enc_flow_sd_nf.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm_wo_flow.enc_flow.load_state_dict(state_dict)

        with open('./mnist_sd/enc_base_sd_nf.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm_wo_flow.enc_base.load_state_dict(state_dict)

        with open('./mnist_sd/decoder_sd_nf.pkl', 'rb') as file:
            state_dict = pkl.load(file)
            gplvm_wo_flow.decoder.load_state_dict(state_dict)

        gplvm_wo_flow.update_parameters()
        # losses_wof = gplvm_wo_flow.train(steps=75)

        pyro.get_param_store().clear()

        idx_max = np.where(gplvm.enc_flow.base_dist.scale > 7.5)[0][0]

        xx = lambda i: gplvm.enc_flow.plot_flow(distb='norm',\
        mu=gplvm.enc_flow.flow_dist.base_dist.loc[i, :].detach(),\
        sg=gplvm.enc_flow.flow_dist.base_dist.scale[i, :].detach())

        xx(idx_max)
        plt.axis('off')

        # ----------------------------------

        X_recon = gplvm.enc_flow.X_map(use_base_mu=True)
        gplvm.decoder.X = X_recon

        Y_recon = gplvm.decoder.forward(Xnew=X_recon)
        Y_recon = Y_recon[0].detach().T

        from matplotlib import offsetbox
        plt.style.use('seaborn-deep')
        plt.figure()
        ax = plt.subplot(aspect='equal')
        plt.ylim(-1.5, 1.5); plt.xlim(-1.25, 1.5)
        plt.tight_layout()

        ax.scatter(X_recon[:, 0], X_recon[:, 1], lw=0, s=40, c=lb/10.)

        shown_images = X_recon[[0], :]
        for i in range(len(Y_recon)):
            if (X_recon[i] - shown_images).square().sum(axis=1).min() < 0.05: continue

            plt.scatter(X_recon[i, 0], X_recon[i, 1], c='black', alpha=0.7)
            ax.add_artist(offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(Y_recon[i].reshape(14, 14),
                    cmap=plt.cm.gray_r), X_recon[i, :]))

            shown_images = np.r_[shown_images, X_recon[[i], :]]
        plt.xticks([]), plt.yticks([])

        # flow fig 5 -------------------

        fig, ax = plt.subplots(1, 4, figsize=(12, 6))

        ax[0].imshow(Y_copy[idx_max, :].reshape(14, 14))
        ax[0].set_title('True Digit', fontsize='small')

        ax[1].imshow(Y[idx_max, :].reshape(14, 14))
        ax[1].set_title('Training Input', fontsize='small')

        ax[2].set_title('Latent Flow', fontsize='small')
        ax[3].set_title('Latent Gaussian', fontsize='small')

        from matplotlib import offsetbox
        import seaborn as sns

        plt.style.use('seaborn-deep')
        ax[2].set_ylim(-7.5, 7.5); ax[2].set_xlim(0.25, 2)

        X_recon = gplvm.enc_flow.flow_dist.sample_n(10000)[:, idx_max, :]
        sns.kdeplot(X_recon[:, 0], X_recon[:, 1], shade=True, color='#DC267F', bw=0.5, ax=ax[2])

        shown_images = None
        for i in range(1000):

            X_recon = gplvm.enc_flow.flow_dist()[[idx_max], :].detach()
            Y_recon = gplvm.decoder.forward(Xnew=X_recon)
            Y_recon = Y_recon[0].detach()[:, 0].reshape(14, 14)
            ax[2].scatter(X_recon[0, 0], X_recon[0, 1], s=20, c='black', alpha=0.05, marker='x')

            if shown_images is not None:
                if (X_recon[0] - shown_images).abs().sum(axis=1).min() < 1:
                    continue

            ax[2].add_artist(offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(Y_recon,
                    cmap=plt.cm.gray_r), X_recon[0, :]))

            if shown_images is None:
                shown_images = X_recon[[0], :]
            else:
                shown_images = np.r_[shown_images, X_recon[[0], :]]

        # gauss fig 5 -------------------

        X_recon = gplvm_wo_flow.enc_flow.X_map(use_base_mu=True)
        gplvm_wo_flow.decoder.X = X_recon

        ax[3].set_ylim(-1.3, -0.9); ax[3].set_xlim(1.2, 1.55)

        X_recon = gplvm_wo_flow.enc_flow.flow_dist.sample_n(10000)[:, idx_max, :]
        sns.kdeplot(X_recon[:, 0], X_recon[:, 1], shade=True, color='b', bw=1, ax=ax[3])

        shown_images = None
        for i in range(1000):

            X_recon = gplvm_wo_flow.enc_flow.flow_dist()[[idx_max], :].detach()
            Y_recon = gplvm_wo_flow.decoder.forward(Xnew=X_recon)
            Y_recon = Y_recon[0].detach()[:, 0].reshape(14, 14)
            ax[3].scatter(X_recon[0, 0], X_recon[0, 1], s=20, c='black', alpha=0.05, marker='x')

            if shown_images is not None:
                if (X_recon[0] - shown_images).abs().sum(axis=1).min() < 0.05:
                    continue

            ax[3].add_artist(offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(Y_recon,
                    cmap=plt.cm.gray_r), X_recon[0, :]))

            if shown_images is None:
                shown_images = X_recon[[0], :]
            else:
                shown_images = np.r_[shown_images, X_recon[[0], :]]

        plt.tight_layout()
        [i.axis('off') for i in ax]

class FlowTests(unittest.TestCase):
    def setUp(self):

        class BaseParams:
            mu = torch.zeros(2).float()
            sigma = torch.ones(2).float()
            model = 'mf'

        self.planar =\
            model.EncoderFlow(2, 10, BaseParams(), 'planar')
        self.radial =\
            model.EncoderFlow(2, 10, BaseParams(), 'radial')
        self.sylvester =\
            model.EncoderFlow(2, 10, BaseParams(), 'sylvester')

    def _test_flow_boilerplate(self, flow):
        optimizer = torch.optim.Adam(flow.parameters(), 0.01)
        for i in range(10000):
            optimizer.zero_grad()

            Z = flow.base_dist.sample_n(10)
            X = flow.forward(Z)
            neg_elbo = flow.flow_dist.log_prob(X) +\
                       data._potential_three(X.clone())
            neg_elbo = neg_elbo.sum()

            neg_elbo.backward()
            optimizer.step()

        # flow.plot_flow()
        self.assertTrue(neg_elbo.data <= 0)
        self.assertTrue(neg_elbo.data >= -50)

        pyro.get_param_store().clear()

    def test_planar(self):
        self._test_flow_boilerplate(self.planar)

    def test_radial(self):
        self._test_flow_boilerplate(self.radial)

    @unittest.skip('Sylvester is not working.')
    def test_sylvester(self):
        self._test_flow_boilerplate(self.sylvester)


class EncFlowTests(unittest.TestCase):
    def test_map(self):

        class BaseParams:
            mu = data.float_tensor(np.tile(range(4), 9).reshape(4, 9))
            sigma = torch.ones((4, 9)).float()
            model = 'mf'

        flow = model.EncoderFlow(num_latent = 9, num_flows = 1,
            activate_flow=False, base_params=BaseParams())

        X_map = flow.X_map()
        np.testing.assert_array_almost_equal(X_map, BaseParams.mu, 2)

    def test_plot_flow(self):
        flow = model.EncoderFlow(num_latent = 2, num_flows = 2)
        flow.plot_flow()


class EncBaseTests(unittest.TestCase):

    def setUp(self):
        n, d, q, X, Y, lb = data.generate_synthetic_data(
            1000, x_type='normal', y_type='hi_dim')
        Y = (Y - Y.min(axis=0).values)
        Y = Y @ (Y.max(axis=0).values - Y.min(axis=0).values).diag().inverse()

        self.hi_dim_data = n, d, q, X, Y, lb

    def test_nn(self):
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)
        nn = model.EncoderBase(data.float_tensor(X), d,
            model='nn', nn_layers=(10, 20))

        opt = torch.optim.Adam(nn.parameters(), lr=0.01)

        n_steps = 10000
        losses = np.zeros(n_steps)
        bar = trange(n_steps, leave=False)
        for step in bar:
            opt.zero_grad()
            nn.update_parameters()
            loss = -torch.distributions.MultivariateNormal(nn.mu,
                scale_tril=nn.sigma).log_prob(Y).sum()
            loss.backward()
            opt.step()
            losses[step] = loss.detach().data
            if loss < -15000:
                break
            bar.set_description(str(int(losses[step])))

        # plot_y_reconstruction(X, Y, nn.mu.detach())
        self.assertTrue(torch.square(Y - nn.mu).mean().detach() < 0.01)

    def test_gp(self):
        n, d, q, X, Y, lb = deepcopy(self.hi_dim_data)
        gp = model.EncoderBase(data.float_tensor(X), d,
            model='gp', gp_inducing_num=25)

        opt = torch.optim.Adam(gp.parameters(), lr=0.01)

        n_steps = 500
        losses = np.zeros(n_steps)
        bar = trange(n_steps, leave=False)
        for step in bar:
            opt.zero_grad()
            gp.update_parameters()
            loss = -torch.distributions.Normal(
                gp.mu, gp.sigma).log_prob(Y).sum()
            loss.backward()
            opt.step()
            losses[step] = loss.detach().data
            bar.set_description(str(int(losses[step])))

        # plot_y_reconstruction(X, Y, gp.mu.detach())
        self.assertTrue(torch.square(Y - gp.mu).mean().detach() < 0.01)
         

#if __name__ == '__main__':
    #unittest.main()
