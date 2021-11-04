
import torch, os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

import pickle as pkl
import gc
from utils.data import load_real_data
from utils.metrics import rmse_missing
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import VariationalLatentVariable, VariationalDenseLatentVariable
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

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.matmul(Y, V[:,:latent_dim])

class GPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, X_init):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        X_prior_mean = torch.zeros(n, latent_dim)

        #prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
        X = VariationalDenseLatentVariable(X_init, prior_x, data_dim)

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

def get_Y_missing(Y, percent):
    
    idx = np.random.binomial(n=1, p=percent, size=Y.shape).astype(bool)
    Y_missing = Y.clone()
    Y_missing[idx] = np.nan
    return Y_missing

def get_Y_test_missing(Y_full, lb, N_test, percent):
    
    idx = np.random.binomial(n=1, p=percent, size=(N_test, Y_full.shape[1])).astype(bool)
    test_idx = np.random.randint(0, Y_full.shape[0], N_test)
    Y_test_missing = Y_full.clone()[test_idx]
    Y_test_missing[idx] = np.nan
    lb_test = lb[test_idx]
    return Y_test_missing, lb_test

    
if __name__ == '__main__':

    SEED = 42
    torch.manual_seed(SEED)
    model_name = 'gauss'

    n, d, q, X, Y, lb = load_real_data('mnist')
    q = 5; Y /= 255
    #Y = Y[:5000]; n = len(Y)
    
    Y = Y[np.isin(lb, (0, 7)), :]
    lb = lb[np.isin(lb, (0, 7))]
    n = len(Y)

    Y_full = Y.clone()
    
    Y_missing = get_Y_missing(Y_full, 0.1)
   
    #(Y.isnan().sum(axis=1) == d).any() # False
    
    print('The % missing from the Y matrix is: ' + str(Y_missing.isnan().float().mean()))

    #plt.imshow(Y_missing[0].reshape(28, 28))
    X_init = torch.nn.Parameter(torch.randn(n, q)) ##_init_pca(Y, q) ## 

    model = GPLVM(n, d, q, n_inducing=100, X_init=X_init)
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    if torch.cuda.is_available():
        device = 'cuda'
        model = model.cuda()
        likelihood = likelihood.cuda()
    else:
        device = 'cpu'

    print('The device is ' + device)
    
    Y_missing = torch.tensor(Y_missing, device=device)
    losses = train(model, likelihood, Y_missing, steps=20000, batch_size=100)
    
    # plt.figure()
    # plt.plot(losses)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # ###### plot latent space for a sanity check
    
    # samples = model.X.q_mu.detach()
    # plt.figure(figsize=(3,3))
    # plt.scatter(samples[:, 2].cpu(), samples[:, 4].cpu(), alpha=0.8, c=lb, marker='x')
    # plt.title('2d Latent Space [MNIST]' + '\n' + '10% missing pixels', fontsize='small')
    
    # # #### plot train reconstructions for sanity check
    
    # plt.style.use('seaborn-deep')
    # fig, axs = plt.subplots(3, 7)
    # fig.suptitle('Reconstructions')
    # k = 1
    # for i in range(3):
    #     for j in range(7):
    #         k += 1
    #         axs[i, j].imshow(model(samples[[k], :]).loc[:, 0].detach().reshape(28, 28).cpu())
    #         axs[i, j].axis('off')
    # plt.tight_layout()
    # plt.suptitle('Train Reconstructions [10% missing pixels]', fontsize='small')

    
    # ## Reset the prior for the full dataset size
    # X_prior_mean = torch.zeros(n, q)

    # model.X.prior_x.loc = torch.zeros(n, q)
    # model.X.prior_x.scale = torch.ones_like(X_prior_mean)
    
    # with open('pre_trained_models/mnist_60_missing.pkl', 'wb') as file:
    #     pkl.dump((model.cpu().state_dict(), likelihood.cpu().state_dict()), file)
        
    # #### Prepare Y_test data with a % of missing 

    N_test = 1000
    Y_test_missing, lb_test = get_Y_test_missing(Y_full, lb, N_test, percent=0.1)

    #### Testing framework 
    
    with torch.no_grad():
        
              model.eval()
              likelihood.eval()
              optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
             
              X_prior_mean_test = torch.zeros(N_test, q)

              prior_x_test = MultivariateNormalPrior(X_prior_mean_test, torch.eye(X_prior_mean_test.shape[1]))

              losses_test,  X_test = model.predict_latent(Y.to(device), Y_test_missing.to(device), optimizer.defaults['lr'], 
                          likelihood, SEED, prior_x=prior_x_test.to(device), ae=False, 
                          model_name=model_name,pca=False, steps=20000)
                 
    rmse = []
    seq = np.arange(0,1001,100)
    
    for i in range(len(seq)-1):
        
        lower = seq[i]
        upper = seq[i+1]
        
        Y_test_recon = model(X_test.q_mu[lower:upper]).loc.T.detach().cpu()
        #Y_train_recon = model(model.X.q_mu[lower:upper]).loc.T.detach().cpu()

        # # Compute the metrics:
        
        # Reconstruction error - Test
        
        #rmse_train = rmse_missing(Y_missing[lower:upper].cpu(), Y_train_recon.detach().cpu())
        rmse_test = rmse_missing(Y_test_missing[lower:upper], Y_test_recon.detach().cpu())
        
        #print(f'Train Reconstruction error {model_name} = ' + str(rmse_train))
        print(f'Test Reconstruction error {model_name} = ' + str(rmse_test))
        
        rmse.append(rmse_test)
        
    print(f'Test Reconstruction error {model_name} = ' + str(torch.mean(torch.Tensor(rmse))))
    
    ### plot some of the test reconstructions for sanity check
    
    plt.style.use('seaborn-deep')
    fig, axs = plt.subplots(3, 7)
    fig.suptitle('Reconstructions')
    k = 10
    for i in range(3):
        for j in range(7):
            k += 1
            axs[i, j].imshow(Y_test_recon[k].reshape(28, 28).cpu())
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.suptitle('Test Reconstructions [10% missing pixels]', fontsize='small')


    samples = model.X.q_mu.detach()
    
    plt.style.use('seaborn-deep')
    fig, axs = plt.subplots(3, 7)
    fig.suptitle('Reconstructions')
    k = 10
    for i in range(3):
        for j in range(7):
            k += 1
            axs[i, j].imshow(model(samples[[k], :]).loc[:, 0].detach().reshape(28, 28).cpu())
            axs[i, j].axis('off')

    plt.style.use('seaborn-deep')
    fig, axs = plt.subplots(3, 7)
    #fig.suptitle('Training digits')
    k = 1
    for i in range(3):
        for j in range(7):
            k += 1
            axs[i, j].imshow(Y_missing[k].reshape(28, 28).cpu())
            axs[i, j].axis('off')
