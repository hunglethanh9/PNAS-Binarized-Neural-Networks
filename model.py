import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Activation, SeparableConv2D, MaxPool2D, AveragePooling2D, concatenate
from keras.layers import BatchNormalization, Flatten
from keras import backend as K
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D, DropoutNoScale
import tensorflow as tf

H = 1.0
kernel_lr_multiplier = 'Glorot'
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4

# BN
epsilon = 1e-6
momentum = 0.9



def binary_tanh(x):
    return (x)

# generic model design
def model_fn(actions, epochs, num_cells, num_cell_filters, dense_layers, dropout):
    print("len actions", len(actions))
    B = len(actions) // 4
    action_list = np.split(np.array(actions), len(actions) // 2)
    model_mulps = 0

    ip = Input(shape=(28, 28, 1))

    print("------------------------------------------")
    print("Type: ", (ip))
    print("Type: ", tf.tile(ip,(-1,1,1,3)))

    print("------------------------------------------")

    x = ip
    for i in range(num_cells):
        print("num cell: ", num_cells)
        print("num cell filters: ", num_cell_filters)

        x, ops = build_cell(x, num_cell_filters[i], action_list, B, stride=(2, 2))
        model_mulps += ops


    print("Model mulps ", model_mulps)
    x_shape = x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*x.get_shape().as_list()[3]
    x = Flatten()(x)
    print("-----------------X----------------")
    print(x_shape)
    
    print("Dropout: ", dropout)
    if(dropout[0]):
        x = DropoutNoScale(dropout[1], input_shape=(x_shape,) )(x)

    for i in range(len(dense_layers)):
        ops = x_shape*dense_layers[i]
        model_mulps += ops
        x = (BinaryDense(dense_layers[i], H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias)(x))
        x = (BatchNormalization(epsilon=epsilon, momentum=momentum)(x))
        if(i <= len(dense_layers)-2):
            x = (Activation(binary_tanh)(x))
            if(dropout[0]):
                x = DropoutNoScale(dropout[2])(x)
        x_shape = dense_layers[i]

    model = Model(ip, x)
    return model, model_mulps

def parse_action(ip, filters, action, strides=(1, 1)):
    '''
    Parses the input string as an action. Certain cases are handled incorrectly,
    so that model can still be built, albeit not with original specification

    Args:
        ip: input tensor
        filters: number of filters
        action: action string
        strides: stride to reduce spatial size

    Returns:
        a tensor with an action performed
    '''
    # applies a 3x3 separable conv
    if action == '3x3 bconv':
        # get ip_height and ip_width 
        x = BinaryConv2D(filters, kernel_size=(3, 3), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        print("-------------Print Shapes-------------")
        print("Ip", ip.get_shape().as_list())
        print("op", x.get_shape().as_list())

        mul_ops = 3*3*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops


    # applies a 5x5 separable conv
    if action == '5x5 bconv':
        x = BinaryConv2D(filters, kernel_size=(5, 5), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 5*5*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]

        return x, mul_ops

    # applies a 7x7 separable conv
    if action == '7x7 bconv':
        x = BinaryConv2D(filters, kernel_size=(7, 7), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 7*7*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]

        return x, mul_ops



    # # applies a 5x5 separable conv
    # if action == '5x5 bconv':
    #     x = SeparableConv2D(filters, (5, 5), strides=strides, padding='same')(ip)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     return x

    # # applies a 7x7 separable conv
    # if action == '7x7 bconv':
    #     x = SeparableConv2D(filters, (7, 7), strides=strides, padding='same')(ip)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     return x

    # applies a 1x7 and then a 7x1 standard conv operation
    if action == '1x7-7x1 bconv':
        x = BinaryConv2D(filters, kernel_size=(1, 7), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)
        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 1*7*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        ip_2 = x.get_shape().as_list()[3]
        x = BinaryConv2D(filters, kernel_size=(7, 1), strides = (1,1),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(x)
        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 7*1*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip_2

        return x, mul_ops

    # # applies a 3x3 standard conv
    # if action == '3x3 conv':
    #     x = Conv2D(filters, (3, 3), strides=strides, padding='same')(ip)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     return x

    # applies a 3x3 maxpool
    if action == '3x3 maxpool':
        return MaxPool2D((3, 3), strides=strides, padding='same')(ip), 0
 


    # applies a 3x3 avgpool
    if action == '3x3 avgpool':
        return AveragePooling2D((3, 3), strides=strides, padding='same')(ip), 0

    # This is used for the identity layer. 
    # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
    if strides == (2, 2):
        print("-----------------------Yes-----------------------")
        channel_axis = -1
        input_filters = ip.get_shape().as_list()[channel_axis]
        print("Input Filters: ", input_filters)
        x = BinaryConv2D(input_filters, kernel_size=(1, 1), strides = (2,2),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)
        mul_ops = 1*1*input_filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        return x, mul_ops
    else:
        print("-----------------------No-----------------------")
        # else just submits a linear layer if shapes match
        return Activation(binary_tanh)(x), 0


def build_cell(ip, filters, action_list, B, stride):
    # if cell size is 1 block only
    # print("Print B: ", B)
    print("Actions: ", action_list)
    # print("IP: ", ip)
    mul_ops = 0

    if B == 1:
        left, ops = parse_action(ip, filters, action_list[0][1], strides=stride)
        print("ops: ",ops)
        mul_ops += ops

        right, ops = parse_action(ip, filters, action_list[1][1], strides=stride)
        print("ops: ",ops)
        
        mul_ops += ops

        # print("Left shape", left.get_shape().as_list())
        # print("Right shape", right.get_shape().as_list())
        return concatenate([left, right], axis=-1), mul_ops

    # else concatenate all the intermediate blocks
    actions = []
    for i in range(B):
        left_action, ops = parse_action(ip, filters, action_list[i * 2][1], strides=stride)
        mul_ops += ops
        
        right_action, ops = parse_action(ip, filters, action_list[i * 2 + 1][1], strides=stride)
        mul_ops += ops
        
        action = concatenate([left_action, right_action], axis=-1)
        actions.append(action)

    # concatenate the final blocks as well
    op = concatenate(actions, axis=-1)
    return op, mul_ops




