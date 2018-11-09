'''
    Created on July, 2018
    
    author: chunyuan
'''

# coding=utf-8

from deepLoco_bn import deepLoco_net
from trainer import Trainer_bn

import image_util, util
import scipy.io as sio
import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split
import sys

####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind   # 1,2,3
# parallel

# Because we have real & imaginary part of our input, data_channels is set to 2
data_channels = 1

np.random.seed(123)

# def main(data_path, file_path):
####################################################
####                DATA LOADING                 ###
####################################################

"""
    here loads all the data we need for training and validating.

"""

# data_path = "data/TrainingSet_deepLoco_64.mat"
data_path = "data/TrainingSet_deepLoco_64.mat"
file_path = "i_run/"

# data_path = "data/TrainingSet_iso_du.mat"
# file_path = "result/"
if not os.path.exists(file_path):
    os.makedirs(file_path)


matfile = h5py.File(data_path, 'r')
data = np.array(matfile['images'])
positions = np.array(matfile['positions'])
weights = np.array(matfile['weights'])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(data, positions, weights, test_size=0.25, random_state=42)
print('Number of Training Examples: %d' % X_train.shape[0])
print('Number of Validation Examples: %d' % X_test.shape[0])
# mdict = {"X_test": X_test, "y_test": y_test, "w_test": w_test}
# sio.savemat("data/TrainingSet_deepLoco_64_test", mdict)
# mdict = {"X_train": X_train, "y_train": y_train, "w_train": w_train}
# sio.savemat("data/TrainingSet_deepLoco_64_train", mdict)

# Setting type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
w_train = w_train.astype('float32')
w_test = w_test.astype('float32')

#-- Training Data --#
max_train = X_train.max()
min_train = X_train.min()
X_train_norm = (X_train-min_train)/(max_train-min_train)

# data_mat = spio.loadmat('train_np/obhatGausWeak128_40.mat', squeeze_me=True)
# truths_mat = spio.loadmat('train_np/obGausWeak128_40.mat', squeeze_me=True)
#
# data = data_mat['obhatGausWeak128']
# data = preprocess(data, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
# truths = preprocess(truths_mat['obGausWeak128'], truth_channels)

#-- Validating Data --#
max_test = X_test.max()
min_test = X_test.min()
X_test_norm = (X_test-min_test)/(max_test-min_test)

# vdata_mat = spio.loadmat('valid_np/obhatGausWeak128val_40.mat', squeeze_me=True)
# vtruths_mat = spio.loadmat('valid_np/obGausWeak128val_40.mat', squeeze_me=True)
#
# vdata = vdata_mat['obhatGausWeak128val']
# vdata = preprocess(vdata, data_channels)
# vtruths = preprocess(vtruths_mat['obGausWeak128val'], truth_channels)

# patch size
psize =  X_train_norm.shape[1]

# Reshaping
X_train_norm = X_train_norm.reshape(X_train.shape[0], psize, psize, 1)
X_test_norm = X_test_norm.reshape(X_test.shape[0], psize, psize, 1)

# normalizing position label
# y_train = y_train/psize
# y_test = y_test/psize

Y_train = np.zeros([y_train.shape[0],y_train.shape[1],y_train.shape[2]+1])
Y_train[:,:,0] = w_train
Y_train[:,:,1:] = y_train
Y_test = np.zeros([y_test.shape[0],y_test.shape[1],y_test.shape[2]+1])
Y_test[:,:,0] = w_test
Y_test[:,:,1:] = y_test

data_provider = image_util.SimpleDataProvider(X_train_norm, Y_train)
valid_provider = image_util.SimpleDataProvider(X_test_norm, Y_train)


####################################################
####                  NETWORK                    ###
####################################################

"""
    here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the deepLoco net
conf = {'batch_size' : 32, # 256//2,
        'eval_batch_size' : 256,
        'iters_per_eval' : 10*4*4,
        'stepsize' : 1E-4,
        'use_cuda' : True,
        'is_2d' : False,
        'max_point_n' : 64,
        'n_workers' : 8,
        'resnet_out_dim' : 2048,
        'max_points' : 256,
        'kernel_sigmas' :  [64.0, 320.0, 640.0, 1920.0],
        'lr_schedule' : [(1.0, 300), (1.0/2, 200), (1.0/4, 100) ,
                         (1.0/8, 100), (1.0/16, 100),(1.0/32, 100),(1.0/64, 100)]
}

kwargs = {
    "layers": 5,           # how many resolution levels we want to have
    "conv_times": 2,       # how many times we want to convolve in each level
    "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
    "filter_size": 3,      # filter size used in convolution
    "pool_size": 2,        # pooling size used in max-pooling
    "summaries": True
}

psize = X_train_norm.shape[1]
net = deepLoco_net(img_channels=data_channels, psize=psize, cost="deepLoco_error", **kwargs)


####################################################
####                 TRAINING                    ###
####################################################

# args for training
batch_size = 32  # batch size for training
valid_size = 32  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# output paths for results
output_path = file_path
prediction_path = file_path
# restore_path = 'gpu001/models/50099_cpkt'

# optional args
opt_kwargs = {
        'learning_rate': 0.0001
}
# 0.001 - all position (0.5,0.5); weights extreme small
# 0.0005 - position various a little bit around (0.5,0.5)

# make a trainer for scadec
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, training_iters=100, epochs=3, display_step=20, save_epoch=100, prediction_path=prediction_path)

#
# if __name__ == "__main__":
#
#     data_path = "data/TrainingSet_deepLoco_iso.mat"
#     file_path = "result/"
#
#     print(sys.argv)
#
#     if sys.argv[1]:
#         data_path = sys.argv[1]
#     if sys.argv[2]:
#         file_path = sys.argv[2]
#
#     main(data_path,file_path)
