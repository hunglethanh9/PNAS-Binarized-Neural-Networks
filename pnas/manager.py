import numpy as np

from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.losses import squared_hinge
import tensorflow as tf
import os 

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, exp_name,  epochs=5, batchsize=128):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: whether to clip rewards in [-0.05, 0.05] range to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize
        self.exp_name = exp_name
    def get_rewards(self, model_fn, actions, num_cells, num_cell_filters, dense_layers, load_saved, dropout):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)
            lr_decay = (lr_end / lr_start)**(1. / self.epochs)

            # generate a submodel given predicted actions
            model, mul_ops = model_fn(actions, self.epochs, num_cells, num_cell_filters, dense_layers, dropout)  # type: Model


            print("---------------Mul Ops----------")
            print(type(mul_ops))
            optimizer = Adam(lr=1e-3, amsgrad=True)
            # model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
            model.compile(loss='squared_hinge', optimizer=optimizer, metrics=['acc'])
            model.summary()
            # print("-------------Model Params------------------")
            # for layer in model.layers:
            #     print(layer.get_output_at(0).get_shape().as_list())


            lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)

            # unpack the dataset
            X_train, y_train, X_val, y_val, X_test, y_test = self.dataset

            print("X_train: ", X_train.shape)
            print("X_val: ", X_val.shape)
            print("X_test: ", X_test.shape)


            weights_path = 'weights/' + self.exp_name + '/'

            if not os.path.exists(weights_path):
                os.makedirs(weights_path)

            if(load_saved):
                model.load_weights(weights_path + 'temp_network.h5')

            # train the model using Keras methods
            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs,
                      verbose=1, validation_data=(X_val, y_val),
                      callbacks=[lr_scheduler, ModelCheckpoint(weights_path + 'temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)])

            # load best performance epoch in this training session
            model.load_weights(weights_path + 'temp_network.h5')

            # evaluate the model
            loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize)

            loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=self.batchsize)

            # compute the reward
            reward = acc

            print()
            print("Manager: Accuracy = ", reward)
            print("Manager: Test Accuracy = ", acc_test)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, mul_ops