# -*- coding: utf-8 -*-
"""
Save reconstruction network outputs to be further analyzed locally
"""
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
import scipy.io
import mat4py as m4p

# Specify if a .mat version will be generated
SAVE_MATLAB = True

# Select whether the reconstruction network based on the modelnet40 data is used or the fine-tuned version
model_net_40_bool = False
name_files = ['network_lr0.0001_mse_3D_fine_tuned', 'network_lr0.0001_mse_3D']
name_file = name_files[model_net_40_bool]

# Select whether Sidney or iCab is going to be reconstructed
only_iCab = False
name_field = 'testingFeatures'

if model_net_40_bool:
    if only_iCab:
        dir_test_input = 'Modelnet_40/FeaturesTest_modelnet40.mat'
        mat_name = 'reconstruction_modelnet40_test.mat'
    else:
        dir_test_input = 'Sidney_and_iCab/FeaturesTrain_Tunning.mat'
        mat_name = 'reconstruction_Sidney_test.mat'
        name_field = 'trainingFeatures'
else:  
    if only_iCab:
        mat_name = 'reconstructions_iCab_test.mat'
        dir_test_input = 'test_iCab/FeaturesTest_u2.mat'
    else:
        dir_test_input = 'Sidney_and_iCab/FeaturesTrain_Tunning.mat'
        mat_name = 'reconstruction_Sidney_test_fine_tuned.mat'
        name_field = 'trainingFeatures'
    # name_file = 'network_lr0.0001_mse_3D'

# Load the features designed for testing
features_test = scipy.io.loadmat(dir_test_input)
features_test = features_test[name_field].transpose()

# Load network and perform predictions over the whole test set
net = load_model(name_file + '.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
flatten_grid_test_reconstructed = net.predict(features_test)  # > BINARY_THRESHOLD

# Save the vector with all the reconstructions of the test set
file = open(name_file + '_reconstructions'+'.pkl', 'wb')
pickle.dump(flatten_grid_test_reconstructed, file)
file.close()

# Load the vector with all reconstructions (in case needed to be loaded for future applications)
# file = open(name_file +'_reconstructions'+'.pkl', 'rb')
# flatten_grid_test_reconstructed = pickle.load(file)
# file.close()

# Save a matlab version of the reconstructions of the test set
if SAVE_MATLAB is True:
    reconstructions_mat = {'reconstructions': flatten_grid_test_reconstructed.tolist()}
    m4p.savemat(mat_name, reconstructions_mat)
