import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, MaxPool2D, AveragePooling2D, concatenate
from keras.layers import BatchNormalization, Flatten
from keras import backend as K
from binarized.binary_ops import binary_tanh as binary_tanh_op
from binarized.binary_layers import BinaryDense, BinaryConv2D, DropoutNoScale, DepthwiseBinaryConv2D
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
    return binary_tanh_op(x)

# generic model design
def model_fn(actions, epochs, num_cells, num_cell_filters, dense_layers, dropout):

    B = len(actions) // 4
    action_list = np.split(np.array(actions), len(actions) // 2)
    model_mulps = 0
    ip = Input(shape=(28, 28, 1))

    x = ip
    for i in range(num_cells):
        x, ops = build_cell(x, num_cell_filters[i], action_list, B, stride=(2, 2))
        model_mulps += ops
    x_shape = x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*x.get_shape().as_list()[3]
    x = Flatten()(x)
    
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

def parse_action(ip, filters, action, strides=(1, 1), depth_multiplier = 1):
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
    

    # applies a 3x3 binary conv
    if action == '3x3 bconv':
        
        # get ip_height and ip_width 
        x = BinaryConv2D(filters, kernel_size=(3, 3), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 3*3*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops

    
    # applies a 5x5 binary conv
    if action == '5x5 bconv':
        
        x = BinaryConv2D(filters, kernel_size=(5, 5), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 5*5*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops

    
    # applies a 7x7 binary conv
    if action == '7x7 bconv':
        
        x = BinaryConv2D(filters, kernel_size=(7, 7), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 7*7*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops

    
    # applies a 1x7 and then a 7x1 binary conv operation
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


    # applies a 7x7 separable binary conv
    if action == '7x7 sep-bconv':
        
        x = DepthwiseBinaryConv2D(filters, kernel_size=(7, 7), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, depth_multiplier= depth_multiplier)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 7*7*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops


    # applies a 5x5 separable binary conv
    if action == '5x5 sep-bconv':
        
        x = DepthwiseBinaryConv2D(filters, kernel_size=(5, 5), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, depth_multiplier= depth_multiplier)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 5*5*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops


    # applies a 3x3 separable binary conv
    if action == '3x3 sep-bconv':
        
        x = DepthwiseBinaryConv2D(filters, kernel_size=(3, 3), strides = strides,
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, depth_multiplier= depth_multiplier)(ip)

        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        mul_ops = 3*3*filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        return x, mul_ops


    # applies a 3x3 maxpool
    if action == '3x3 maxpool':
        return MaxPool2D((3, 3), strides=strides, padding='same')(ip), 0
 
    # applies a 3x3 avgpool
    if action == '3x3 avgpool':
        return AveragePooling2D((3, 3), strides=strides, padding='same')(ip), 0

    # This is used for the identity layer. 
    # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
    if strides == (2, 2):
        channel_axis = -1
        input_filters = ip.get_shape().as_list()[channel_axis]
        x = BinaryConv2D(input_filters, kernel_size=(1, 1), strides = (2,2),
                       data_format='channels_last',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias)(ip)
        mul_ops = 1*1*input_filters*x.get_shape().as_list()[1]*x.get_shape().as_list()[2]*ip.get_shape().as_list()[3]
        x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1)(x)
        x = Activation(binary_tanh)(x)
        return x, mul_ops
    else:
        # else just submits a linear layer if shapes match
        return Activation(binary_tanh)(x), 0

def build_cell(ip, filters, action_list, B, stride):
    print("Actions: ", action_list)
    mul_ops = 0

    if B == 1:
        left, ops = parse_action(ip, filters, action_list[0][1], strides=stride)
        mul_ops += ops
        right, ops = parse_action(ip, filters, action_list[1][1], strides=stride)
        mul_ops += ops
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




