import numpy as np
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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





# # ------------Copy This Block For Each Experiment-------------
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# EXPERIMENT_NAME = "Test-Exp4"
# ACTION = "[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]	[[1. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = 0
# INPUT_LOOKFORWARD_DEPTH = 0
# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool']
# NUM_CELLS = 10
# NUM_CELL_FILTERS = [8, 16, 16, 16, 16,16, 16, 16, 16,32 ]
# DENSE_LAYERS = [10]
# REGULARIZATION = 0  # regularization strength

# # ------------End Block---------------------------------------



# # ------------Copy This Block For Each Experiment-------------
# EXPERIMENT_NAME = "Maxpool-Sainty-Test"
# ACTION = "[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = 0
# INPUT_LOOKFORWARD_DEPTH = 0
# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [16, 24 ,32]
# DENSE_LAYERS = [10]
# # ------------End Block---------------------------------------


# # ------------Copy This Block For Each Experiment-----------gave: 98.60% --
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# EXPERIMENT_NAME = "LOOKBACK-UNTIL-BEST"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 7
# NUM_CELL_FILTERS = [8, 16 ,16, 16 ,16, 16 ,32]
# DENSE_LAYERS = [32, 10]
# # ------------End Block---------------------------------------


# # ------------Copy This Block For Each Experiment-------------

"""
WITH-DROPOUT-EXP
Epoch 00199: val_acc did not improve from 0.98380
Epoch 200/200
55000/55000 [==============================] - 29s 527us/step - loss: 0.0629 - acc: 0.9461 - val_loss: 0.0257 - val_acc: 0.9835

"""

"""
WITHOUT-DROPOUT-EXP
Epoch 00199: val_acc did not improve from 0.98520
Epoch 200/200
55000/55000 [==============================] - 29s 522us/step - loss: 0.0077 - acc: 0.9913 - val_loss: 0.0123 - val_acc: 0.9835

"""


# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# # dropout
# drop_in = 0.2
# drop_hidden = 0.5
# DROPOUT = (False ,drop_in, drop_hidden)
# EXPERIMENT_NAME = "WITHOUT-DROPOUT-EXP"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [8, 16 ,24]
# DENSE_LAYERS = [64, 32, 10]



# # ------------End Block---------------------------------------


# # ------------Copy This Block For Each Experiment-------------
"""

Epoch 00009: val_acc did not improve from 0.97560
Epoch 10/10
275000/275000 [==============================] - 140s 509us/step - loss: 0.0367 - acc: 0.9528 - val_loss: 0.0230 - val_acc: 0.9698

Epoch 00010: val_acc did not improve from 0.97560
5000/5000 [==============================] - 1s 184us/step
10000/10000 [==============================] - 2s 186us/step

Manager: Accuracy =  0.9756
Manager: Test Accuracy =  0.9733
Final Accuracy :  (0.9756, -59713136.0)

I think to train this more! 


"""


# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# # dropout
# drop_in = 0.2
# drop_hidden = 0.5
# DROPOUT = (False ,drop_in, drop_hidden)
# EXPERIMENT_NAME = "EXPANSION-EXP"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [8, 16 ,24]
# DENSE_LAYERS = [64, 32, 10]

# MAX_EPOCHS = 10  # maximum number of epochs to train
# USE_EXPANSION = True


# # ------------End Block---------------------------------------



# # # ------------Copy This Block For Each Experiment-------------

# """
# Epoch 00199: val_acc did not improve from 0.99160
# Epoch 200/200
# 55000/55000 [==============================] - 28s 504us/step - loss: 0.0053 - acc: 0.9984 - val_loss: 0.0105 - val_acc: 0.9895

# Epoch 00200: val_acc did not improve from 0.99160
# 10000/10000 [==============================] - 2s 180us/step
# 5000/5000 [==============================] - 1s 175us/step

# Manager: Accuracy =  0.9916
# Manager: Test Accuracy =  0.9916
# Final Accuracy :  (0.9916, -59713136.0)



# """

# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# # dropout
# drop_in = 0.2
# drop_hidden = 0.5
# DROPOUT = (False ,drop_in, drop_hidden)
# EXPERIMENT_NAME = "NOACT-EXP"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = True
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [8, 16 ,24]
# DENSE_LAYERS = [64, 32, 10]

# MAX_EPOCHS = 200  # maximum number of epochs to train
# USE_EXPANSION = False


# # # ------------End Block---------------------------------------



# # ------------Copy This Block For Each Experiment  -------------

"""

Epoch 00039: val_acc did not improve from 0.99240
Epoch 40/40
275000/275000 [==============================] - 121s 441us/step - loss: 0.0135 - acc: 0.9892 - val_loss: 0.0190 - val_acc: 0.9884

Epoch 00040: val_acc did not improve from 0.99240
5000/5000 [==============================] - 1s 174us/step
10000/10000 [==============================] - 2s 175us/step

Manager: Accuracy =  0.9924
Manager: Test Accuracy =  0.9926
Final Accuracy :  (0.9924, -59713136.0)



"""


# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# # dropout
# drop_in = 0.2
# drop_hidden = 0.5
# DROPOUT = (False ,drop_in, drop_hidden)
# EXPERIMENT_NAME = "NOACT-EXPANSION-EXP"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = True
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [8, 16 ,24]
# DENSE_LAYERS = [64, 32, 10]

# MAX_EPOCHS = 40  # maximum number of epochs to train
# USE_EXPANSION = True


# # ------------End Block---------------------------------------





# # ------------Copy This Block For Each Experiment 99.10% val last time -------------




# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# # dropout
# drop_in = 0.2
# drop_hidden = 0.5
# DROPOUT = (False ,drop_in, drop_hidden)
# EXPERIMENT_NAME = "NOACT-EXPANSION-EXP"
# ACTION = "[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[1. 0. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]	[[0. 2. 0. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = -2
# INPUT_LOOKFORWARD_DEPTH = 1

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool', 'identity']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [8, 16 ,24]
# DENSE_LAYERS = [64, 32, 10]

# MAX_EPOCHS = 60  # maximum number of epochs to train
# USE_EXPANSION = True


# # ------------End Block---------------------------------------








# # ------------Copy This Block For Each Experiment-------------
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# EXPERIMENT_NAME = "Test-Exp3"
# ACTION = "[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 0. 0. 5.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]	[[1. 0. 0. 0. 0.]]	[[0. 0. 3. 0. 0.]]"
# LOAD_SAVED = False
# INPUT_LOOKBACK_DEPTH = 0
# INPUT_LOOKFORWARD_DEPTH = 0

# operators = ['3x3 bconv','5x5 bconv','7x7 bconv', '1x7-7x1 bconv',
#               '3x3 maxpool']
# NUM_CELLS = 3
# NUM_CELL_FILTERS = [16, 24 ,32]
# DENSE_LAYERS = [128, 10]
# # ------------End Block---------------------------------------






# # ------------Copy This Block For Each Experiment-------------


os.environ["CUDA_VISIBLE_DEVICES"]="3"
# dropout
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





if(USE_EXPANSION):
	train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(True)
	train_data = train_total_data[:, :-10]
	train_labels = train_total_data[:, -10:]

	x = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
	x_train, y_train, x_val, y_val, x_test, y_test = mnist_process(x)

else: 
	download_mnist.maybe_download('./mnist/MNIST_data/')
	mnist = input_data.read_data_sets('./mnist/MNIST_data/', one_hot=True)

	x = [ mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, mnist.validation.images, mnist.validation.labels]
	x_train, y_train, x_val, y_val, x_test, y_test = mnist_process(x)

dataset = [x_train, y_train, x_val, y_val, x_test, y_test]

for i in range(len(dataset)):
	print("Dataset", i, " :", dataset[i].shape)

action = get_action(ACTION)
# create the Network Manager
manager = NetworkManager(dataset, EXPERIMENT_NAME,  epochs=MAX_EPOCHS, batchsize=BATCHSIZE)
with policy_sess.as_default():
    K.set_session(policy_sess)
rewards = []
print("Predicted actions : ", state_space.parse_state_space_list(action))
reward = manager.get_rewards(model_fn, state_space.parse_state_space_list(action), NUM_CELLS, NUM_CELL_FILTERS, DENSE_LAYERS, LOAD_SAVED, DROPOUT)
print("Final Accuracy : ", reward)
