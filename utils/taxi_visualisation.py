#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reconstructions for taxi-cab

"""

import os
import pickle as pkl
import matplotlib.pylab as plt

with open('pre_trained_models/point_recons.pkl', 'rb') as file:
    Y_test_point, Y_recon_point = pkl.load(file)
    
with open('pre_trained_models/map_recons.pkl', 'rb') as file:
    Y_test_map, Y_recon_map = pkl.load(file)

with open('pre_trained_models/gauss_recons.pkl', 'rb') as file:
    Y_test_gauss, Y_recon_gauss = pkl.load(file)

with open('pre_trained_models/nn_recons.pkl', 'rb') as file:
    Y_test_nn, Y_recon_nn = pkl.load(file)
    

plt.figure(figsize=(8,3))
plt.plot(Y_test_map, marker='+', c='grey')
plt.plot(Y_recon_nn[:,0] , c='yellow', alpha=0.6, label='yellow cabs')
plt.plot(Y_recon_nn[:,1], c='green',alpha=0.7, label='green cabs')
plt.plot(Y_recon_nn[:,2], c='purple', alpha=0.7, label='for-hire')
plt.legend(fontsize='small')
plt.title('Test reconstructions - AEB-SVI model', fontsize='small')
plt.xlabel('Time in hours (10 days)',fontsize='small')
plt.ylabel('Vehicle counts', fontsize='small')
plt.tight_layout()
plt.savefig('plots/nn_taxi.png')
