import numpy as np
import csv
import os
import mnist_data
from binary_ops import mnist_process
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from encoder import Encoder, StateSpace
from manager import NetworkManager
from model import model_fn
import ast
from tensorflow.examples.tutorials.mnist import input_data
from mnist import download_mnist



def get_action(s):
    S = ""
    for i in range(len(s)):
        if(s[i] == " " or s[i] == "	"):
            S+=","
        else:
            S+= s[i]
    return ast.literal_eval("[ " + S + " ]")

B = 5  # number of blocks in each cell

MAX_EPOCHS = 200  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
REGULARIZATION = 0  # regularization strength


# # ------------Edit This Block For Each Experiment-------------


os.environ["CUDA_VISIBLE_DEVICES"]="3"
drop_in = 0.2
drop_hidden = 0.5
DROPOUT = (False ,drop_in, drop_hidden)
EXPERIMENT_NAME = "MODEL-3mul-10pow5"
ACTION = "[[1. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0.]]"
LOAD_SAVED = False
INPUT_LOOKBACK_DEPTH = 0
INPUT_LOOKFORWARD_DEPTH = 0

operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
              '3x3 maxpool']
NUM_CELLS = 3
NUM_CELL_FILTERS = [16, 24 ,32]
DENSE_LAYERS = [64, 32, 10]

MAX_EPOCHS = 50  # maximum number of epochs to train
USE_EXPANSION = True


# # ------------End Block---------------------------------------


# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

# construct a state space
state_space = StateSpace(B, input_lookback_depth=INPUT_LOOKBACK_DEPTH, input_lookforward_depth=INPUT_LOOKFORWARD_DEPTH,
                         operators=operators)

dataset = build_dataset(USE_EXPANSION)

# create the Network Manager
manager = NetworkManager(dataset, EXPERIMENT_NAME,  epochs=MAX_EPOCHS, batchsize=BATCHSIZE)

action = get_action(ACTION)

with policy_sess.as_default():
    K.set_session(policy_sess)

print("Predicted actions : ", state_space.parse_state_space_list(action))
reward = manager.get_rewards(model_fn, state_space.parse_state_space_list(action), NUM_CELLS, NUM_CELL_FILTERS, DENSE_LAYERS, LOAD_SAVED, DROPOUT)
print("Final Accuracy : ", reward)
