
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
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=False)

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

def plot_skeleton(fig, subplot_index, Y_vec, missing_verts=set(), recursive=False):

    missing_verts = get_all_missing_verts(missing_verts, recursive)

    ax = fig.add_subplot(subplot_index, projection='3d')

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
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_zticks([])

if __name__ == '__main__':

    torch.manual_seed(42)

    # motions = [f'{i:02d}' for i in range(1, 17)]
    motions = ['01', '02', '15', '16', '35', '36'] # jump x2, walk x2, run x2
    data = pods.datasets.cmu_mocap('16', motions)
    data['Y'][:, 0:3] = 0.0

    Y = torch.tensor(data['Y']).float()
    n = len(Y); d = len(Y.T); q = 6
    lb = np.where(data['lbls'])[1]

    # [f'{i}. ' + vertex.name for i, vertex in enumerate(data['skel'].vertices)]
    # plot_skeleton(plt.figure(), 111, Y[0, :], {1}, True)
    # plot_skeleton(plt.figure(), 111, Y[0, :], {17}, True)
    # plot_skeleton(plt.figure(), 111, Y[0, :], {1, 24}, True)
    # plot_skeleton(plt.figure(), 111, Y[0, :], {1, 14, 18}, True) # missing head, right leg and forearm
    # plot_skeleton(plt.figure(), 111, Y[0, :], {6, 25, 18}, True) # missing forearms, left leg
    # plot_skeleton(plt.figure(), 111, Y[0, :], {12}, True) # missing upper body
    # plot_skeleton(plt.figure(), 111, Y[0, :], {1, 6}, True) # missing lower body

    Y_full = Y.clone()

    sets_for_removal = [{1, 14, 18}, {6, 25, 18}, {12}, {1, 6}]

    for i, set_to_rm in enumerate(sets_for_removal):
        remove_idx = list(get_y_dims_to_nullify(get_all_missing_verts(set_to_rm, True)))
        for idx in remove_idx:
            Y[(lb+1) % 4 == i, idx] = np.nan

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
    losses = train(model, likelihood, Y, steps=15000, batch_size=200)

    if os.path.isfile('for_paper/mocap_cpu_diff_motions.pkl'):
        with open('for_paper/mocap_cpu_diff_motions.pkl', 'rb') as file:
            model_sd, likl_sd = pkl.load(file)
            model.load_state_dict(model_sd)
            likelihood.load_state_dict(likl_sd)

    # with open('for_paper/mocap_cpu_diff_motions.pkl', 'wb') as file:
    #    pkl.dump((model.cpu().state_dict(), likelihood.cpu().state_dict()), file)

    Y_recon = model(model.X.q_mu).loc.T.detach().cpu()

    fig = plt.figure(figsize=(8,3))
    plot_skeleton(fig, 132, Y_full[0, :], {1, 14, 18}, True)    
    plt.title('Train', fontsize='small')

    plot_skeleton(fig, 133, Y_recon[0, :])
    plt.title('Reconstruction', fontsize='small')

    plot_skeleton(fig, 131, Y_full[0, :])
    plt.title('Ground Truth', fontsize='small')

    # plot_skeleton(Y_full[205, :], {12}, True)    
    # plt.title('Training Data')

    # plot_skeleton(Y_recon[205, :])
    # plt.title('Model Reconstruction')

    # plot_skeleton(Y_full[205, :])
    # plt.title('Data without missing values')

    plt.ioff()
    for i in range(n):
        fig = plt.figure(figsize=(8,3))
        plot_skeleton(fig, 132, Y_full[i, :], sets_for_removal[(lb[i]+1) % 4], True)
        plt.title('Training Data: ' + str(lb[i]))
        plot_skeleton(fig, 133, Y_recon[i, :])
        plt.title('Reconstruction: ' + str(lb[i]))
        plot_skeleton(fig, 131, Y_full[i, :])
        plt.title('Ground Truth: ' + str(lb[i]))
        plt.savefig('img/' + f'{i:03d}' + '.png')
        plt.close()

    # scp -r aditya@192.168.1.154:~/gplvf/img/ . && cd img
    # convert -delay 10 -loop 0 *.png plot.gif && rm *.png

    Y_test_full = torch.tensor(pods.datasets.cmu_mocap('16', ['21', '45', '03'])['Y']).float() # walk, run, high jump
    Y_test_full[:, 0:3] = 0.0
    Y_test = Y_test_full.clone()
    n_test = len(Y_test)

    remove_idx = list(get_y_dims_to_nullify(get_all_missing_verts({1, 14, 18}, True)))
    Y_test[:, remove_idx] = np.nan

    losses_test, X_test = model.cpu().predict_latent(Y.cpu(), Y_test,
        lr=0.005, likelihood=likelihood.cpu(), seed=1,
        prior_x=NormalPrior(torch.zeros(n_test, q), torch.ones(n_test, q)),
        ae=False, model_name='gauss', pca=False, steps=40000)

    Y_test_recon = model(X_test.q_mu).loc.T.detach().cpu()
    np.save('y_test_recon.npy', Y_test_recon)

    plt.ioff()
    for i in range(n_test):
        fig = plt.figure(figsize=(8,3))
        plot_skeleton(fig, 132, Y_test_full[i, :], {1, 14, 18}, True)
        plt.title('Test Data')
        plot_skeleton(fig, 133, Y_test_recon[i, :])
        plt.title('Reconstruction')
        plot_skeleton(fig, 131, Y_test_full[i, :])
        plt.title('Ground Truth')
        plt.savefig('img/' + f'{i:03d}' + '.png')
        plt.close()
