
import torch, os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

import pickle as pkl
from utils.data import load_real_data
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import VariationalIAF
from models.likelihoods import GaussianLikelihoodWithMissingObs
from utils.visualisation import plot_y_reconstruction

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class GPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, context_size=2, n_flows=10):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)

        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
        X = VariationalIAF(n, latent_dim, context_size, prior_x, data_dim, n_flows)

        super(GPLVM, self).__init__(X, q_f)

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

def train(model, likelihood, Y, steps=1000, batch_size=100):

    elbo = VariationalELBO(likelihood, model, num_data=len(Y))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    losses = []
    iterator = trange(steps)
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample_batch = model.sample_latent_variable(batch_idx=batch_index)
        output_batch = model(sample_batch)
        loss = -elbo(output_batch, Y[batch_index].T).sum()
        losses.append(loss.item())
        iterator.set_description(
            '-elbo: ' + str(np.round(loss.item(), 2)) +\
            ". Step: " + str(i))
        loss.backward()
        optimizer.step()

    return losses

if __name__ == '__main__':

    torch.manual_seed(42)

    n, d, q, X, Y, lb = load_real_data('brendan_faces')
    q = 5; Y /= 255

    # remove some obs from Y
    Y_full = Y.clone()
    idx_a = np.random.choice(range(n), n * int(d/2))
    idx_b = np.random.choice(range(d), n * int(d/2))
    Y[idx_a, idx_b] = np.nan
    # (Y.isnan().sum(axis=1) == d).any() # False hopefully

    # plt.imshow(Y[0].reshape(28, 28))

    model = GPLVM(n, d, q, n_inducing=120, n_flows=0)
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    if torch.cuda.is_available():
        device = 'cuda'
        model = model.cuda()
        likelihood = likelihood.cuda()
    else:
        device = 'cpu'

    Y = torch.tensor(Y, device=device)
    model.X.jitter = model.X.jitter.to(device=device)
    losses = train(model, likelihood, Y, steps=10, batch_size=450)

    if os.path.isfile('for_paper/faces_cpu.pkl'):
        with open('for_paper/faces_cpu.pkl', 'rb') as file:
            model_sd, likl_sd = pkl.load(file)
            model.load_state_dict(model_sd)
            likelihood.load_state_dict(likl_sd)

    #with open('for_paper/faces_5dim_latent.pkl', 'wb') as file:
    #   pkl.dump((model.state_dict(), likelihood.state_dict()), file)

    samples = model.X.get_latent_flow_means().detach().cpu()
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, c=lb)
    
    
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    fig = plt.figure(figsize=(16,3))
    
    grid1 = ImageGrid(fig, 133, nrows_ncols=(3,7), axes_pad=0.001)
    
    k = 10
    #images = model(samples[[k], :]).loc[:, 0]
    for axs in grid1:
              k += 1
              axs.imshow(model(samples[[k], :]).loc[:, 0].detach().reshape(28, 20))
              axs.axis('off')
    grid1.axes_all[3].set_title('Reconstructions', fontsize='small')
        
    grid2 = ImageGrid(fig, 132, nrows_ncols=(3,7), axes_pad=0.001)
    
    k = 10
    #images = model(samples[[k], :]).loc[:, 0]
    for axs in grid2:
              k += 1
              axs.imshow(Y[k].reshape(28, 20).cpu())
              axs.axis('off')
    grid2.axes_all[3].set_title('Train', fontsize='small')
    
    grid3 = ImageGrid(fig, 131, nrows_ncols=(3,7), axes_pad=0.01)
    
    k = 10
    #images = model(samples[[k], :]).loc[:, 0]
    for axs in grid3:
              k += 1
              axs.imshow(Y_full[k].reshape(28, 20).cpu())
              axs.axis('off')
    grid3.axes_all[3].set_title('Ground Truth', fontsize='small')

     
    

    # plt.style.use('seaborn-deep')
    # axs = plt.subplots(3, 7)
    # fig.suptitle('Reconstructions')
    # k = 10
    # for i in range(3):
    #     for j in range(7):
    #         k += 1
    #         axs[i, j].imshow(model(samples[[k], :]).loc[:, 0].detach().reshape(28, 20))
    #         axs[i, j].axis('off')
    # plt.subplots_adjust(wspace=0, hspace=0.02)

    # plt.style.use('seaborn-deep')
    # fig, axs = plt.subplots(3, 7)
    # fig.suptitle('Faces Data')
    # k = 10
    # for i in range(3):
    #     for j in range(7):
    #         k += 1
    #         axs[i, j].imshow(Y[k].reshape(28, 20).cpu())
    #         axs[i, j].axis('off')
    # plt.subplots_adjust(wspace=0, hspace=0.02)

    # plt.style.use('seaborn-deep')
    # fig, axs = plt.subplots(3, 7)
    # fig.suptitle('Ground Truth')
    # k = 10
    # for i in range(3):
    #     for j in range(7):
    #         k += 1
    #         axs[i, j].imshow(Y_full[k].reshape(28, 20).cpu())
    #         axs[i, j].axis('off')
    # plt.subplots_adjust(wspace=0, hspace=0.02)
