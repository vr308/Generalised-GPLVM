
import torch, os, pods
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

import pickle as pkl
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import VariationalLatentVariable
from models.likelihoods import GaussianLikelihoodWithMissingObs

from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class GPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        X = VariationalLatentVariable(X_prior_mean, prior_x, data_dim)

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

def get_children(vertex, recursive=False):
    children = data['skel'].vertices[vertex].children
    if not recursive or len(children) == 0:
        return children
    else:
        result = children.copy()
        for child in children:
            result.extend(get_children(child, True))
        return result

def get_all_missing_verts(missing_verts=set(), recursive=False):
    for vertex in missing_verts.copy():
        missing_verts = missing_verts.union(get_children(vertex, recursive))
    return missing_verts

def get_y_dims_to_nullify(missing_verts):
    indices = []
    for vertex in missing_verts:
        vertex = data['skel'].vertices[vertex]
        indices.extend(vertex.meta['pos_ind'] + vertex.meta['rot_ind'])
    indices = set(indices)
    indices.remove(-1)
    return indices

def plot_skeleton(Y_vec, missing_verts=set(), recursive=False):

    missing_verts = get_all_missing_verts(missing_verts, recursive)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    Z = data['skel'].to_xyz(Y_vec)
    idx_to_show = ~np.isin(np.arange(len(Z)), list(missing_verts))
    ax.scatter(Z[idx_to_show, 0], Z[idx_to_show, 2], Z[idx_to_show, 1], marker='.', color='b')

    connect = data['skel'].connection_matrix() # Get the connectivity matrix.
    I, J = np.nonzero(connect)
    xyz = np.zeros((len(I)*3, 3)); idx=0
    for i, j in zip(I, J):
        if i in missing_verts:
            continue
        xyz[idx]     = Z[i, :]
        xyz[idx + 1] = Z[j, :]
        xyz[idx + 2] = [np.nan]*3
        idx += 3
    line_handle = ax.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], '-', color='b')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)

if __name__ == '__main__':

    torch.manual_seed(42)

    motions = [f'{i:02d}' for i in range(1, 3)]
    data = pods.datasets.cmu_mocap('35', motions)
    data['Y'][:, 0:3] = 0.0

    Y = torch.tensor(data['Y']).float()
    n = len(Y); d = len(Y.T); q = 2
    lb = np.where(data['lbls'])[1]

    # [vertex.name for vertex in data['skel'].vertices]
    # plot_skeleton(Y[0, :], {1}, True)
    # plot_skeleton(Y[0, :], {17}, True)

    Y_full = Y.clone()

    leg_idx = list(get_y_dims_to_nullify(get_all_missing_verts({1}, True)))
    for idx in leg_idx:
        Y[lb == 0, idx] = np.nan

    hand_idx = list(get_y_dims_to_nullify(get_all_missing_verts({17}, True)))
    for idx in hand_idx:
        Y[lb == 1, idx] = np.nan

    Y[:, :3] = 0.0
    # plt.imshow(Y)

    model = GPLVM(n, d, q, n_inducing=30)
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    if torch.cuda.is_available():
        device = 'cuda'
        model = model.cuda()
        likelihood = likelihood.cuda()
    else:
        device = 'cpu'

    Y = torch.tensor(Y, device=device)
    losses = train(model, likelihood, Y, steps=15000, batch_size=n//4)

    # if os.path.isfile('for_paper/mocap.pkl'):
    #     with open('for_paper/mocap.pkl', 'rb') as file:
    #         model_sd, likl_sd = pkl.load(file)
    #         model.load_state_dict(model_sd)
    #         likelihood.load_state_dict(likl_sd)

    with open('for_paper/mocap.pkl', 'wb') as file:
        pkl.dump((model.state_dict(), likelihood.state_dict()), file)

    Y_recon = model(model.X.q_mu).loc.T.detach().cpu()

    plt.ioff()
    for i in range(90):
        plot_skeleton(Y_full[i, :], {1}, True)
        plt.title('Training Data')
        plt.savefig('img/data_' + str(i) + '.png')

        plot_skeleton(Y_recon[i, :])
        plt.title('Reconstruction Walking')
        plt.savefig('img/recons_' + str(i) + '.png')
