import numpy as np
import csv
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from binarized.binary_ops import mnist_process
from pnas.encoder import Encoder, StateSpace
from pnas.manager import NetworkManager
from pnas.model import model_fn
from mnist import mnist_data


if not os.path.exists('architectures/'):
    os.makedirs('architectures/')

def print_dataset_metrics(dataset):
    print()
    print("--------DATASET-------")
    print("X train shape: ", dataset[0].shape)
    print("Y train shape: ", dataset[1].shape)

    print("X val shape: ", dataset[2].shape)
    print("Y val shape: ", dataset[3].shape)

    print("X test shape: ", dataset[4].shape)
    print("Y test shape: ", dataset[5].shape)

def build_dataset(use_expansion):
    train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(use_expansion)
    train_data = train_total_data[:, :-10]
    train_labels = train_total_data[:, -10:]
    x = [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]
    x_train, y_train, x_val, y_val, x_test, y_test = mnist_process(x)
    dataset = [x_train[:10], y_train[:10], x_val[:10], y_val[:10], x_test[:10], y_test[:10]]
    print_dataset_metrics(dataset)
    return dataset

def log_architecture(experiment_name, log_string):
    f = open('architectures/' + experiment_name + '.txt','a')
    f.write(log_string)
    f.close()

def get_architecure_from_action(action):
    arc = ""
    for i in range(len(action)):
        arc += np.array_str(action[i]) + " "
    return '"' + arc[:-1] + '"'


# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

EXPERIMENT_NAME = "HARD-LIMIT-3mul10pow6-REMOVE-SKIP"

B = 3   # number of blocks in each cell
K_ = 3  # number of children networks to train
DROP_INPUT = 0.2
DROP_HIDDEN = 0.5
DROPOUT= (False, DROP_INPUT, DROP_HIDDEN) # dropout only applied to the dense layers and the input
MAX_EPOCHS = 1  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
REGULARIZATION = 0  # regularization strength
CONTROLLER_CELLS = 100  # number of cells in RNN controller
RNN_TRAINING_EPOCHS = 15
RESTORE_CONTROLLER = True  # restore controller to continue training
NUM_CELLS = 3
NUM_CELL_FILTERS = [16, 24, 32]
DENSE_LAYERS = [32, 10]
USE_EXPANSION = False


LOAD_SAVED = False

operators = ['3x3 sep-bconv','5x5 sep-bconv', '1x7-7x1 conv',
              '3x3 bconv']  # use the default set of operators, minus identity and conv 3x3

# construct a state space
state_space = StateSpace(B, input_lookback_depth=0, input_lookforward_depth=0,
                         operators=operators)

# print the state space being searched
state_space.print_state_space()
NUM_TRAILS = state_space.print_total_models(K_)
dataset = build_dataset(USE_EXPANSION)


with policy_sess.as_default():
    # create the Encoder and build the internal policy network
    controller = Encoder(policy_sess, state_space, EXPERIMENT_NAME ,B=B, K=K_,
                         train_iterations=RNN_TRAINING_EPOCHS,
                         reg_param=REGULARIZATION,
                         controller_cells=CONTROLLER_CELLS,
                         restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, EXPERIMENT_NAME,  epochs=MAX_EPOCHS, batchsize=BATCHSIZE)
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

        # print("Action Type: ", (action[0]))
        # print("Architecture: ", get_architecure_from_action(action))
        # for i in range(len(action)):
            # print("Printing action " + str(i) + " :",action[i])
        # print the action probabilities
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

        log_str = "\nSr. No: " + str(t+1) + "\nArchitecture: " + str(state_space.parse_state_space_list(action)) + "\nRepresentation String: " + get_architecure_from_action(action) + "\n"
        log_architecture(EXPERIMENT_NAME, log_str)

        # with open(train_hist_name, mode='a+', newline='') as f:
        #     data = [reward]
        #     data.extend(state_space.parse_state_space_list(action))
        #     data.extend(action)
        #     writer = csv.writer(f)
        #     writer.writerow(data)

    with policy_sess.as_default():
        K.set_session(policy_sess)
        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step(rewards)
        print("Trial %d: Encoder loss : %0.6f" % (trial + 1, loss))

        controller.update_step()
        print()
    log_architecture(EXPERIMENT_NAME, "\n \n --------------------EXPERIMENT FINISHED------------------- \n \n")

print("Finished !")