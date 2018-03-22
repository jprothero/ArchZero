from copy import deepcopy

def load_weights(model, weights, debug = False, metrics = None, finetune=True):
    if weights is None:
        raise Exception("Weights not found.")
    
    origModelWeights = deepcopy(model.get_weights())
    weight_idx = 0
    setWeights = True
    for i, layer in enumerate(model.layers):
        if "batch" in layer.name:
            if debug: print(layer.name)
        if layer.get_weights() != []:
            if layer.get_weights()[0].shape == weights[weight_idx].shape and i < len(model.layers)-1:
                temp = []
                for j, weight in enumerate(layer.get_weights()):
                    if weight_idx+j < len(weights)-1:
                        weight_idx += j
                        temp.append(weights[weight_idx])
                    else:
                        setWeights = False
                if setWeights:
                    layer.set_weights(temp)
                    if finetune: layer.trainable = False
                    weight_idx += 1
                if debug: print("Weight group {} of {} used".format(weight_idx, len(weights)))
                if debug: print(layer.name, "Weight Changed")
        else:
            if debug: print(layer.name, "Weight Not Changed")
            
    if debug: 
        for weight in weights[weight_idx-1:]:
            print(weight.shape)
    
    assert (origModelWeights[0] != model.get_weights()[0]).any()
    
    return model

from keras.optimizers import Nadam
from clr_callback import CyclicLR

def fit(model, ds, loss="categorical_crossentropy", metrics=["acc"], epochs=3, finetune=False, verbose=1):
    optim = Nadam()
    base_lr = 0.001
    max_lr = 0.006
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
                   step_size=2000., mode='triangular')
    
    model.compile(optimizer=optim,
                  loss=loss,
                  metrics=metrics)
    
    if finetune:
        orig_epochs = epochs
        epochs //= 2
        model.fit_generator(generator=ds.train_gen, steps_per_epoch=ds.train_steps,
            epochs=epochs, verbose=verbose, callbacks=[clr], validation_data=ds.val_gen,
            validation_steps=ds.val_steps)
        
        base_lr /= 2
        max_lr /= 2
        
        for layer in model.layers:
            layer.trainable = True
            
        model.compile(optimizer=optim,
                  loss=loss,
                  metrics=metrics)
        
        model.fit_generator(generator=ds.train_gen, steps_per_epoch=ds.train_steps,
            epochs=epochs, verbose=verbose, callbacks=[clr], validation_data=ds.val_gen,
            validation_steps=ds.val_steps)
        
        epochs = orig_epochs
    
    return model.fit_generator(generator=ds.train_gen, steps_per_epoch=ds.train_steps,
                        epochs=epochs, verbose=verbose, callbacks=[clr], validation_data=ds.val_gen,
                        validation_steps=ds.val_steps).history['val_loss'][-1]
    

def twosFloor(num):
    if num % 2 == 1:
        return (num - 1) // 2
    else:
        return num // 2

def shiftedTwosFloor(num):
    num += 3
    if num % 2 == 1:
        return (num - 1) // 2
    else:
        return num // 2

from model_assembly import assemble_model
import numpy as np
from dataset import Dataset
from IPython.core.debugger import set_trace
ds = Dataset()

def convert_to_matrix(state, new_rep):
    M = []
    for _ in range(state.max_depth):
        M.append([])
        for _ in range(state.max_depth):
            M[-1].append("")
    for i in range(state.size):
        M[twosFloor(i)][shiftedTwosFloor(i)] = new_rep[i]
        
    return M

import pickle as p
import database as db

def eval_arch(state, node):
    board = state.board
    layer_types = state.layer_types
    new_rep = []
    for i in range(len(board)):
        new_rep.append(layer_types[board[i]])
    M = convert_to_matrix(state, new_rep)
    
    print(M)
    try:
        model = assemble_model(M)
    except Exception as e:
        print(e)
        return -1
    prev_loss = 0 
    results = db.select("""SELECT loss FROM data WHERE depth={}""".format(node["depth"]))
    if len(results) > 0:
        for res in results:
            prev_loss += res[0]
        prev_loss /= len(results)
    if prev_loss == 0:
        prev_loss = None
            
    loss = fit(model, ds, finetune=False, verbose=1)
    
    db.insert("""INSERT INTO data VALUES (?, ?, ?, ?, ?)""", (node["depth"], loss, '', '', 0))
    print("prev_loss: {}, new_loss: {}".format(prev_loss, loss))
        
    if prev_loss is None:
        return 0
    elif loss < prev_loss:
        return 1
    else:
        return -1