import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import csv
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from pnas.encoder import Encoder, StateSpace
from pnas.manager import NetworkManager
from pnas.model import model_fn
from mnist.mnist_data import get_dataset
import ast

if not os.path.exists('architectures/'):
    os.makedirs('architectures/')

def get_action(s):
    S = ""
    for i in range(len(s)):
        if(s[i] == " " or s[i] == "	"):
            S+=","
        else:
            S+= s[i]
    print(S)
    return ast.literal_eval("[ " + S + " ]")

def log_architecture(experiment_name, log_string):
    f = open('architectures/' + experiment_name + '.txt','a')
    f.write(log_string)
    f.close()

def get_architecure_from_action(action):
    arc = ""
    for i in range(len(action)):
        arc += np.array_str(action[i]) + " "
    return '"' + arc[:-1] + '"'


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-ta", "--train_arc", dest="train_arc",
                    help="Set this to True for training an architecture. Default = False.", default = False)

args = parser.parse_args()


# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

EXPERIMENT_NAME = "HARD-LIMIT-3mul10pow6-REMOVE-SKIP"



# -------Controller Training Settings-------
B = 3   # number of blocks in each cell
K_ = 64  # number of children networks to train
REGULARIZATION = 0  # regularization strength on RNN controller
CONTROLLER_CELLS = 100  # number of cells in RNN controller
RNN_TRAINING_EPOCHS = 15
RESTORE_CONTROLLER = True  # restore controller to continue training
# ------------------------------------------




# ------- Common Settings --------
DROP_INPUT = 0.2
DROP_HIDDEN = 0.5
DROPOUT= (False, DROP_INPUT, DROP_HIDDEN) # dropout only applied to the dense layers and the input
MAX_EPOCHS = 6  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
NUM_CELLS = 3 # No. of cells in each architecture
NUM_CELL_FILTERS = [16, 24, 32]
DENSE_LAYERS = [32, 10]
USE_EXPANSION = False # Data augmentation for MNIST
operators = ['3x3 sep-bconv','5x5 sep-bconv', '1x7-7x1 conv',
              '3x3 bconv'] 
# --------------------------------




# -------Architecture Training Settings-------
NUM_EPOCHS = 200 
REPRESENTATION_STRING = "[[1. 0. 0.]] [[1. 0. 0. 0.]] [[1. 0. 0.]] [[1. 0. 0. 0.]]"
LOAD_SAVED = False # Use this to continue training a saved architecture 
# ------------------------------------------




TRAIN_ARCHITECTURE = args.train_arc

# get the dataset 
dataset = get_dataset(USE_EXPANSION)

# construct a state space
state_space = StateSpace(B, input_lookback_depth=0, input_lookforward_depth=0,
                         operators=operators)





# Execute PNAS
if(TRAIN_ARCHITECTURE is False):
    # create the Network Manager
    manager = NetworkManager(dataset, EXPERIMENT_NAME,  epochs=MAX_EPOCHS, batchsize=BATCHSIZE)
    # print the state space being searched
    LOAD_SAVED = False # this is used to load a saved model in architecture training 
    state_space.print_state_space()
    NUM_TRAILS = state_space.print_total_models(K_)

    with policy_sess.as_default():
        # create the Encoder and build the internal policy network
        controller = Encoder(policy_sess, state_space, EXPERIMENT_NAME ,B=B, K=K_,
                             train_iterations=RNN_TRAINING_EPOCHS,
                             reg_param=REGULARIZATION,
                             controller_cells=CONTROLLER_CELLS,
                             restore_controller=RESTORE_CONTROLLER)


    print()

    log_architecture(EXPERIMENT_NAME, 'All the evaluated architectures will be logged in this file. \n \n \n')

    # train for number of trails with the corresponding B. 
    for trial in range(B):
        log_architecture(EXPERIMENT_NAME, '---- B= ' + str(trial) + " Architectures ---- \n" )

        with policy_sess.as_default():
            K.set_session(policy_sess)

            if trial == 0:
                k = None
            else:
                k = K_

            actions = controller.get_actions(top_k=k)  # get all actions for the previous state
        rewards = []
        for t, action in enumerate(actions):

            state_space.print_actions(action)
            print("Model #%d / #%d" % (t + 1, len(actions)))
            print("Predicted actions : ", state_space.parse_state_space_list(action))

            # build a model, train and get reward and accuracy from the network manager
            reward, mul_ops = manager.get_rewards(model_fn, state_space.parse_state_space_list(action), NUM_CELLS, NUM_CELL_FILTERS, DENSE_LAYERS, LOAD_SAVED, DROPOUT)
            print("Final Accuracy : ", reward)
            rewards.append(reward)
            print("\nFinished %d out of %d models ! \n" % (t + 1, len(actions)))

            # write the results of this trial into a file
            train_hist_name = EXPERIMENT_NAME + '_train_history.csv'

            log_str = "\nSr. No: " + str(t+1)+ "\nReward: " + str(reward) + "\nArchitecture: " + str(state_space.parse_state_space_list(action)) + "\nRepresentation String: " + get_architecure_from_action(action) + "\n"
            log_architecture(EXPERIMENT_NAME, log_str)

        with policy_sess.as_default():
            K.set_session(policy_sess)
            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step(rewards)
            print("Trial %d: Encoder loss : %0.6f" % (trial + 1, loss))

            controller.update_step()
            print()
        log_architecture(EXPERIMENT_NAME, "\n \n --------------------EXPERIMENT FINISHED------------------- \n \n")

# Execute architecture training
else:
    # create the Network Manager
    manager = NetworkManager(dataset, EXPERIMENT_NAME,  epochs=NUM_EPOCHS, batchsize=BATCHSIZE)
    action = get_action(REPRESENTATION_STRING)
    print("Predicted actions : ", state_space.parse_state_space_list(action))
    reward = manager.get_rewards(model_fn, state_space.parse_state_space_list(action), NUM_CELLS, NUM_CELL_FILTERS, DENSE_LAYERS, LOAD_SAVED, DROPOUT)
    print("Final Accuracy : ", reward)

print("Finished !")