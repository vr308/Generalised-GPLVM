#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing metrics for GPLVM models

- Test Reconstruction error for Y_train with and without missing data 
- Negative test log-likelihood

"""

import torch
import numpy as np
#from pyro.distributions.transforms import AffineTransform

def float_tensor(X): return torch.tensor(X).float()

def rmse(Y_test, Y_recon):
    
    return torch.mean((Y_test - float_tensor(Y_recon))**2).sqrt()

def rmse_missing(Y_test, Y_recon):
    
    return torch.sqrt(torch.Tensor([np.nanmean(np.square(Y_test - Y_recon))]))

# def decomm_test_log_likelihood(model, Y_test, test_dist):
    
#      if isinstance(model.enc_flow.flows[0],  AffineTransform):
#          model.decoder.X = model.enc_base.mu.detach()
#      else:
#          model.decoder.X = model.enc_flow.X_map(n_restarts=1,use_base_mu=True)
     
#      test_log_lik_samples = torch.Tensor([model.log_p_of_y_given_x(test_dist(), Y_test) for _ in range(100)])
#      return torch.mean(test_log_lik_samples)/len(Y_test)

