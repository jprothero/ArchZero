import numpy as np
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Conv1D, merge, concatenate, MaxPooling1D, GlobalMaxPooling1D, 
Embedding, Dropout, Activation, SeparableConv2D, Conv2D, BatchNormalization, Flatten, GlobalMaxPooling2D, 
                          GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Layer, PReLU)
from keras.optimizers import Nadam, Adam
from keras.layers.core import Reshape
# import database as db
from factor_tools import sorted_factors, most_common_factors, middle_factors
from os.path import join
import re
import sys
from IPython.core.debugger import set_trace
import pickle
from keras import backend as K
from copy import copy
from keras.layers import Input
from copy import copy, deepcopy
    
def zero_pad_layers(layers):
    max_w = 0
    max_h = 0
    for layer in layers:   
        print(layer.shape)
        if layer.shape[1].value is not None:
            max_w = max(max_w, layer.shape[1].value)
            if len(layer.shape) > 2:
                 max_h = max(max_h, layer.shape[2].value)
            else:
                max_h = copy(max_w)
           
    for k, layer in enumerate(layers):
        print(layer.shape)
        if layer.shape[1].value is not None:
            w_diff = max_w - layer.shape[1].value
            if len(layer.shape) > 2:
                h_diff = max_h - layer.shape[2].value
            else:
                h_diff = copy(w_diff)

            if w_diff > 0 or h_diff > 0:
                layers[k] = ZeroPadding2D(((int(np.ceil(w_diff/2)), int(np.floor(w_diff/2))), 
                                                  (int(np.ceil(h_diff/2)), int(np.floor(h_diff/2)))))(layer)
    return layers

from IPython.core.debugger import set_trace

def convert_graph_to_keras(G, inputs):
    G = deepcopy(G)
        
    node_inputs = []

    for i in range(len(G[1])):
        node_inputs.append([])
    node_inputs[0].append(inputs)
    
    curr_output = inputs

    for i in range(len(G)):
        if len(node_inputs[i]) > 1:
           # print(node_inputs[i])
            node_inputs[i] = zero_pad_layers(node_inputs[i])
            node_inputs[i] = [concatenate(node_inputs[i], axis = -1)]
            curr_output = node_inputs[i][0]
        for j in range(len(G[0])):
            if len(G[i][j]) > 0:
                curr_output = convert_to_keras_layer(G[i][j], prev = curr_output)
                node_inputs[j].append(curr_output)
                if len(node_inputs[i]) > 0 and node_inputs[i][0] == inputs:
                    node_inputs[i] = [curr_output]
                        
    return curr_output

def convert_to_keras_layer(layer_type, prev, num_filters = 32, dense_size = 256, max_params = 1e6, no_repeat = False):
    max_params = max_params // dense_size
    layer_type = layer_type.lower()
    
    if layer_type == 'conv2d1x1':
        x = Conv2D(num_filters, 1, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    elif layer_type == 'conv2d3x3':
        x = Conv2D(num_filters, 3, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    elif layer_type == 'conv2d1x1stride':
        x = Conv2D(num_filters, kernel_size=1, strides=2, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    elif layer_type == 'conv2d3x3stride':
        x = Conv2D(num_filters, kernel_size=3, strides=2, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    elif layer_type == 'conv2dsep1x1':
        x = SeparableConv2D(num_filters, 1, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    elif layer_type == 'conv2dsep3x3':
        x = SeparableConv2D(num_filters, 3, padding = 'same')(prev)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    else:
        print("Layer type not found")
        print(layer_type)
        sys.exit(1)

# def load_weights(model, weights):
#     for layer in model.layer()
#         set_trace
#         test = "hi"
    
def assemble_model(adj_matrix,
                   input_shape = (32, 32, 3), 
                   optim = Adam(lr = 5e-4), 
                   num_outputs = 10,
                   output_type = 'categorical'):
    inputs = Input(shape = input_shape)
    
    x = convert_graph_to_keras(adj_matrix, inputs)
    
    if len(x.shape) != 2:
        x = Flatten()(x)
#x = Dense(128)(x)
       # x = BatchNormalization()(x)
        #x = Activation("relu")(x)
        #x = Dropout(.2)(x)
        #x = Dense(128)(x)
        #x = BatchNormalization()(x)
        #x = Activation("relu")(x)
        #x = Dropout(.2)(x)
        
    if output_type is 'categorical':
        outputs = Dense(num_outputs, activation = 'softmax')(x)
        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer=optim,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    else:
        outputs = Dense(num_outputs)(curr_output)
        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer=optim,
                  loss='mse',
                  metrics=['mae'])
    
#     if weights is not None:
#         load_weights(model, weights)
    
    return model