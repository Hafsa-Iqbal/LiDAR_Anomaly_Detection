# -*- coding: utf-8 -*-
"""
Fine-tuning of initial decoder network trained on modelnet40 data by using iCab data
"""
# Module to open .mat files
import scipy.io

# Import the necessary modules
import numpy as np
import random
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras import Sequential
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.layers import Dense, Conv3DTranspose, Reshape, Flatten
# from tensorflow.keras.layers import Dropout


def save_net(network_reconstruct, name_net, hist_train_loss, hist_val_loss):
    history_losses = {'loss': hist_train_loss, 'val_loss': hist_val_loss}
    network_reconstruct.save(name_net + '.h5')
    file = open(name_net+'.pkl', 'wb')
    pickle.dump(history_losses, file)
    file.close()
    print('Network saved! \n')


# Load training and validation data
features_train = scipy.io.loadmat('test_iCab/FeaturesTrain_u2.mat')
features_train = features_train['trainingFeatures'].transpose()

flatten_grid_train = scipy.io.loadmat('test_iCab/DecoderTrainingData_u2.mat')
flatten_grid_train = flatten_grid_train['TrainInputData']

features_test = scipy.io.loadmat('test_iCab/FeaturesTest_u2.mat')
features_test = features_test['testingFeatures'].transpose()
flatten_grid_test = scipy.io.loadmat('test_iCab/DecoderTestingData_u2.mat')
flatten_grid_test = flatten_grid_test['TestInputData']

name_net_load = 'network_lr0.0001_mse_3D'
name_net_save = name_net_load + '_fine_tuned'
network_reconstruct = load_model(
    name_net_load + '.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
hist_train_loss = []
hist_val_loss = []

latentDim = features_train.shape[1]
LEARNING_RATE = 0.001  # 0.0001 (for MSE_LOSS = TRUE)
MAX_EPOCHS = 100
BATCH_SIZE = 32
SAVE_NET_EACH_EPS = 10
EARLY_STOPPING = False
optimizer = Adam(learning_rate=LEARNING_RATE)
for i in range(MAX_EPOCHS):
    print(f'episode: {i+1} out of {MAX_EPOCHS}')
    hist = network_reconstruct.fit(x=features_train, y=flatten_grid_train,
                                   batch_size=BATCH_SIZE, epochs=1,
                                   validation_data=(features_test, flatten_grid_test),
                                   shuffle=True)
    hist_train_loss.append(hist.history['loss'][0])
    hist_val_loss.append(hist.history['val_loss'][0])

    if (i+1) % SAVE_NET_EACH_EPS == 0:
        loss_train_diff = hist_train_loss[-SAVE_NET_EACH_EPS] - hist_train_loss[-1]
        loss_test_diff = hist_val_loss[-SAVE_NET_EACH_EPS] - hist_val_loss[-1]

        print(
            f'training improvement (in the last {SAVE_NET_EACH_EPS} epochs): {loss_train_diff/hist_train_loss[-SAVE_NET_EACH_EPS]*100}%')
        print(
            f'validation improvement (in the last {SAVE_NET_EACH_EPS} epochs): {loss_test_diff/hist_val_loss[-SAVE_NET_EACH_EPS]*100}%')
        print('\n')

        # Early stopping based on no significant improvement of losses between each episode (they should improve at least 0.1%)
        if loss_train_diff < hist_train_loss[-SAVE_NET_EACH_EPS]*0.001 or loss_test_diff < 0:
            print('Early stoping due to no significant improvement \n')
            EARLY_STOPPING = True
            break
        save_net(network_reconstruct, name_net_save, hist_train_loss, hist_val_loss)

if (i+1) == MAX_EPOCHS and EARLY_STOPPING is False:
    save_net(network_reconstruct, name_net_save, hist_train_loss, hist_val_loss)
