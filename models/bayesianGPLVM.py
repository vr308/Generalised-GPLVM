#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from prettytable import PrettyTable
from tqdm import trange
import numpy as np

class BayesianGPLVM(ApproximateGP):
    def __init__(self, X, variational_strategy):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param X (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    VariationalLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
     
        super(BayesianGPLVM, self).__init__(variational_strategy)
        
        # Assigning Latent Variable 
        self.X = X 
    
    def forward(self):
        raise NotImplementedError
          
    def sample_latent_variable(self, *args, **kwargs):
        sample = self.X(*args, **kwargs)
        return sample
    
    def initialise_model_test(self, trained_model, Y_star):
        
        return;
    
    def predict_latent(self, Y_train, Y_star, trained_model, optimizer, elbo, ae=True):
        
        if ae is True:
            # encoder based model
            mu_star = self.X.mu(Y_star)
            sigma_star = self.X.sigma(Y_star)
            return mu_star, sigma_star
            
        else:
            # Train for test X
            
            # The idea here is to initialise a new test model but import all the trained 
            # params from the training model. The training data / variational params
            # do not affect the test data.
            
            # Initialise test model at training params
            test_model = self.initialise_model_test(trained_model, Y_star)
            
            loss_list = []
            iterator = trange(5000, leave=True)
            batch_size = 100
            for i in iterator: 
                batch_index = test_model._get_batch_idx(batch_size)
                optimizer.zero_grad()
                sample = test_model.sample_latent_variable()  # a full sample returns latent x across all N
                sample_batch = sample[batch_index]
                output_batch = test_model(sample_batch)
                loss = -elbo(output_batch, Y_star[batch_index].T).sum()
                loss_list.append(loss.item())
                iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
                loss.backward()
                optimizer.step()
                
            return test_model.X

    
    def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
    
   
