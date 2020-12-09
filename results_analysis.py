# -*- coding: utf-8 -*-
"""
Plot reconstruction losses and some 3D representations (locally aster runing "network_outputs_saving.py")
"""

###############################################################
# Loading the log training data and network (plotting log data and analyze results)
###############################################################

import pickle
import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# import tensorflow as tf
import random
import scipy.io
from scipy.ndimage import rotate
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import tensorflow.keras as keras


def transform_rotate_3D(flatten_grid_test):
    # Assuming 32 x 32 x 32 voxels
    rotated_3d_version = flatten_grid_test.reshape([32, 32, 32], order='C')
    rotated_3d_version = rotate(rotated_3d_version, 110, axes=(0, 2))
    return rotated_3d_version


save_fig_bool = False

# Sidney + icab dataset
# dir_test_input = 'Sidney_iCab_Merged/FeaturesTest_Tunning.mat'
# dir_test_output = 'Sidney_iCab_Merged/DecoderTestingData_Tunning.mat'

model_net_40_bool = True
name_files = ['network_lr0.0001_mse_3D_fine_tuned', 'network_lr0.0001_mse_3D']
name_file = name_files[model_net_40_bool]

if model_net_40_bool:
    # Modelnet40 dataset
    dir_test_input = 'Modelnet_40/FeaturesTest_modelnet40.mat'
    dir_test_output = 'Modelnet_40/DecoderTestingData_modelnet40.mat'
    name_loss_plot = 'loss_function_MSE'
    ind_test = [441, 653, 2196, 1668, 500]
    BINARY_THRESHOLD = 0.1
else:
    # iCab dataset
    dir_test_input = 'test_iCab/FeaturesTest_u2.mat'
    dir_test_output = 'test_iCab/DecoderTestingData_u2.mat'
    name_loss_plot = 'loss_function_MSE_fine_tuned'
    # name_file = 'network_lr0.0001_mse_3D'
    ind_test = [2196, 214, 2303, 1552, 156]
    BINARY_THRESHOLD = 0.05

##################################################################
# Plot the loss function through time
##################################################################
file_log = open(name_file + '.pkl', 'rb')
history = pickle.load(file_log)
file_log.close()
figure = plt.figure()
plt.plot(history['loss'][1:], c='red', label='training loss')
plt.plot(history['val_loss'][1:], c='blue', label='validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.show()

if save_fig_bool:
    figure.savefig(name_loss_plot + '.pdf', bbox_inches='tight')

######################################################################
# Plot some 3D representations
######################################################################
# Load the ground truth reconstructions
flatten_grid_test = scipy.io.loadmat(dir_test_output)
flatten_grid_test = flatten_grid_test['TestInputData']

# Load reconstructed version of the test set of selected case (model_net or iCab)
file_log = open(name_file + '_reconstructions' + '.pkl', 'rb')
flatten_grid_test_reconstructed = pickle.load(file_log)
file_log.close()

random.seed(123)
flatten_grid_test_reconstructed = flatten_grid_test_reconstructed > BINARY_THRESHOLD
n_test_plot = len(ind_test)
# n_test_plot = 5
# ind_test = [random.randint(0, len(flatten_grid_test)) for i in range(n_test_plot)]

figure = plt.figure()
for i in range(1, n_test_plot+1):
    # Original voxels
    ax = figure.add_subplot(2, n_test_plot, i, projection='3d')
    rotated_3D_original = transform_rotate_3D(flatten_grid_test[ind_test[i-1], :])
    ax.voxels(rotated_3D_original > 0)
    plt.axis('off')

    # Reconstructured voxels
    ax = figure.add_subplot(2, n_test_plot, n_test_plot + i, projection='3d')
    # flatten_grid_test_reconstructed = net.predict(features_test[ind_test[i-1],:])
    rotated_3D_original = transform_rotate_3D(flatten_grid_test_reconstructed[ind_test[i-1], :]*1)
    ax.voxels(rotated_3D_original)
    plt.axis('off')

# Save image and corresponding output
if save_fig_bool:
    figure.savefig(name_file + 'voxel_comparison.pdf', bbox_inches='tight')
