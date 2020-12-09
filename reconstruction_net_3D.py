# -*- coding: utf-8 -*-
"""
Network for reconstructing voxel activations (Decoder version with transpose 3D convolutions)
This network is initially trained with data coming from 'Modelnet_40/' folder
"""
# Module to open .mat files
import scipy.io

# Import the necessary modules
import numpy as np
import random
import h5py

# import progressbar

import tensorflow as tf

import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam


# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv3DTranspose, Reshape, Flatten
from tensorflow.keras.layers import Dropout

# from tensorflow.keras.callbacks import TensorBoard


def save_net(network_reconstruct, name_net, hist_train_loss, hist_val_loss):
    history_losses = {'loss': hist_train_loss, 'val_loss': hist_val_loss}
    network_reconstruct.save(name_net + '.h5')
    file = open(name_net+'.pkl', 'wb')
    pickle.dump(history_losses, file)
    file.close()
    print('Network saved! \n')


# Load training and validation data
#############################################################
# Note: scale training data would be an interesting move here
#############################################################

features_train = scipy.io.loadmat('Modelnet_40/FeaturesTrain_modelnet40.mat')
features_train = features_train['trainingFeatures'].transpose()

# flatten_grid_train = scipy.io.loadmat('Modelnet_40/DecoderTrainingData_modelnet40.mat')
# flatten_grid_train = flatten_grid_train['Modelnet_40/TrainInputData']

file_flatten_grid_train = h5py.File('Modelnet_40/DecoderTrainingData_modelnet40.mat', 'r')
flatten_grid_train = file_flatten_grid_train.file['TrainInputData'][:][:].transpose()


features_test = scipy.io.loadmat('Modelnet_40/FeaturesTest_modelnet40.mat')
features_test = features_test['testingFeatures'].transpose()
flatten_grid_test = scipy.io.loadmat('Modelnet_40/DecoderTestingData_modelnet40.mat')
flatten_grid_test = flatten_grid_test['TestInputData']

latentDim = features_train.shape[1]
MSE_LOSS = True
SIGMOID_LAYER = True
LEARNING_RATE = 0.0001  # 0.0001 (for MSE_LOSS = TRUE)
MAX_EPOCHS = 1000
BATCH_SIZE = 64
SAVE_NET_EACH_EPS = 10
name_net = 'network_lr' + str(LEARNING_RATE)
optimizer = Adam(learning_rate=LEARNING_RATE)
# optimizer = SGD(lr=0.001, momentum=0.99)

# Number of filters and filter-dimensions based on Modelnet40 architecture
num_filters_1 = 32
dim_filters_1 = 2
stride_1 = 2

num_filters_2 = 32
dim_filters_2 = 3
stride_2 = 1

num_filters_3 = 1
dim_filters_3 = 6
stride_3 = 2

network_reconstruct = Sequential()
network_reconstruct.add(
    Dense(6 * 6 * 6 * 32, activation=tf.nn.leaky_relu, input_shape=(latentDim, )))
network_reconstruct.add(Reshape((6, 6, 6, 32)))
network_reconstruct.add(Conv3DTranspose(num_filters_1, dim_filters_1,
                                        activation=tf.nn.leaky_relu, strides=stride_1))
network_reconstruct.add(Dropout(0.2))
network_reconstruct.add(Conv3DTranspose(num_filters_2, dim_filters_2,
                                        activation=tf.nn.leaky_relu, strides=stride_2))
# network_reconstruct.add(Dropout(0.3))
if SIGMOID_LAYER:
    network_reconstruct.add(Conv3DTranspose(num_filters_3, dim_filters_3,
                                            activation='sigmoid', strides=stride_3))
else:
    network_reconstruct.add(Conv3DTranspose(num_filters_3, dim_filters_3,
                                            activation=tf.nn.leaky_relu, strides=stride_3))

# network_reconstruct.add(Dropout(0.4))
network_reconstruct.add(Flatten())
network_reconstruct.summary()

if MSE_LOSS:
    network_reconstruct.compile(loss='mse', optimizer=optimizer)
    name_net = name_net + '_mse'
else:
    network_reconstruct.compile(loss=tf.losses.categorical_crossentropy, optimizer=optimizer)
    name_net = name_net + '_crossentropy'

name_net = name_net + '_3D'

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
hist_train_loss = []
hist_val_loss = []

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
            break
        save_net(network_reconstruct, name_net, hist_train_loss, hist_val_loss)

if (i+1) == MAX_EPOCHS:
    save_net(network_reconstruct, name_net, hist_train_loss, hist_val_loss)

file_flatten_grid_train.close()
