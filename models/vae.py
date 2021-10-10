#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE based models for small datasets

"""

import torch
import gc
import torchvision
from PIL import Image
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets
from torchvision.utils import save_image
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
float_tensor = lambda x: torch.tensor(x, dtype=torch.float)
torch.manual_seed(345)
#device='cpu'

class Dataset(torch.utils.data.Dataset):
  
  '''Characterizes a dataset for PyTorch'''
  
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

class VAE(nn.Module):
    
        def __init__(self, latent_dims):
            super(VAE, self).__init__()
            
            # encoder
            self.encoder = nn.Sequential(
                    nn.Linear(in_features=12, out_features=5),
                    nn.ReLU(), )
            # decoder
            self.decoder = nn.Sequential(
                nn.Linear(in_features=latent_dims, out_features=5),
                nn.ReLU(),
                nn.Linear(in_features=5, out_features=12))
            
            self.mu_layer = nn.Linear(5, latent_dims)
            self.sigma_layer = nn.Linear(5, latent_dims)
            
        def forward(self, x):
            
            # encoding
            x = F.relu(self.encoder(torch.flatten(x, start_dim=1)))
            
            # get mu and log_var
            mu = self.mu_layer(x)
            sigma = F.softplus(self.sigma_layer(x))
            
            # get the latent representation
            eps = torch.randn_like(sigma)
            z = mu + eps*sigma
            log_var = torch.log(sigma**2)
            
            #decoding
            reconstruction = torch.sigmoid(self.decoder(z))
            
            return reconstruction, mu, log_var
        
        @staticmethod
        def get_batch(x, batch_size):
            
            N = len(x)
            valid_indices = np.array(range(N))
            batch_indices = np.random.choice(valid_indices,size=batch_size,replace=False)
            return float_tensor(x[batch_indices,:])
        

def final_loss(x, reconstruction, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    bce = nn.BCELoss(reduction='sum')
    BCE = bce(reconstruction, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
            
def train(vae, train_loader, test_loader, epochs=10):
    
    vae.train()
    opt = torch.optim.Adam(vae.parameters())
    #mse_loss = nn.MSELoss(reduce=True, reduction='sum')
    total_loss = 0.
    bs = train_loader.batch_size
    
    size = train_loader.dataset.shape[0]
    batches_per_epoch = size//bs
    
    loss_list = []
    for epoch in range(epochs):
        
        # Training
        vae.train() # Turn on the train mode
        epoch_start_time = time.time()
        start_time = time.time()
        
        with tqdm(train_loader, unit='batch') as tepoch:
            
            for batch_idx, batch_data in enumerate(tepoch):
                
                tepoch.set_description(f"Epoch {epoch}")
                    
                x = batch_data
                x = x.to(device) # GPU
                x = x.view(bs, 12)
                opt.zero_grad()
                reconstruction, mu, log_var = vae(x)
                loss = final_loss(x, reconstruction, mu, log_var)
                loss.backward()
                opt.step()
                loss_list.append(loss.item())
                total_loss += loss.item()
                #log_interval = 1
                
                # if batch_idx % log_interval == 0 and batch_idx > 0:
                #     cur_loss = total_loss / log_interval
                #     elapsed = time.time() - start_time
                #     print('| epoch {:3d} | {:5d}/{:5d} batches | '
                #       '| ms/batch {:5.2f} | '
                #       'loss {:5.2f} |'.format(
                #         epoch, batch_idx, batches_per_epoch,
                #         elapsed * 1000 / log_interval,
                #         loss.item()))
                #     total_loss = 0
                #     start_time = time.time()
            
            #print(loss)
            #train_loss = total_loss/len(train_loader.dataset)
            #print(train_loss)
            ## Validate
             
            # val_loss = validate(vae, test_loader, epoch)
                    
            # print('-' * 89)
            # print('| end of epoch {:3d} | time: {:5.2f}s | train loss: {:5.2f}|  val loss: {:5.2f}'
            #          .format(epoch, (time.time() - epoch_start_time), cur_loss, val_loss))
            # print('-' * 89)
            
            # epoch_start_time = time.time()
         
            ## clearing memory cache 
            
            gc.collect()
            torch.cuda.empty_cache()

    return vae, loss_list

def validate(model, val_loader, epoch):
    model.eval()
    running_loss = 0.0
    num_data = len(val_loader.dataset)
    bs = val_loader.batch_size
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            x = data
            x = x.to(device)
            x = x.view(bs, 12)
            #x = x.view(data.size(0), -1)
            reconstruction, mu, sigma = model(x)
            loss = final_loss(x, reconstruction, mu, sigma)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            #if i == int(num_data/val_loader.batch_size) - 1:
            #    num_rows = 8
            #    both = torch.cat((x.view(val_loader.batch_size, 1, 28, 28)[:8], 
            #                      reconstruction.view(val_loader.batch_size, 1, 28, 28)[:8]))
            #    save_image(both.cpu(), f"Neural Nets/results/output{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(val_loader.dataset)
    return val_loss

def plot_latent(model, test_X, test_y):
        z = model.encoder(test_X.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 1], z[:, 2], c=test_y, cmap='tab10')

if __name__ == "__main__":

    ## Load oilflow
    import urllib.request
    import tarfile
    
    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')
        f.extract('DataTst.txt')
        f.extract('DataTstLbls.txt')
        f.extract('DataVdn.txt')
        f.extract('DataVdnLbls.txt')
    
    Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
    labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)
    ## Initialise model
    latent_dim = 2
    vae = VAE(latent_dim).to(device) # GPU
    
    ## Data-loaders
    train_loader = torch.utils.data.DataLoader(Y, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(Y, batch_size=100, shuffle=True)
    
    ## Check forward pass

    inputs = train_loader.__iter__().next()
    inputs = inputs.to(device)
    outputs, mu, sigma = vae(inputs)
    
    ### Training loop
    
    trained_vae, losses = train(vae, train_loader, test_loader, epochs=1000)

    plot_latent(trained_vae, Y, labels)
    #inputs = inputs.view(100, 784)
    
    #vae.loss_fn(inputs, outputs, mu, sigma)