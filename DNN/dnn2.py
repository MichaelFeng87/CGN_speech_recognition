#!/usr/bin/env python

"""
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import math
import lasagne
import tables
from dnn_data import Split_dataset, load_dataset
# set to true to use functions which load data into memory once. Set to false to load
# data in minibatches.
load=False
# open the data file
data_loc= '/data/processed/mfcc.h5'
data_file = tables.open_file(data_loc, mode='r+') 
#get a list of feature nodes
f_nodes = data_file.root.features._f_list_nodes()
#get a list of label nodes
l_nodes = data_file.root.labels._f_list_nodes()
# total number of nodes (i.e. files) 
n_nodes= len(f_nodes)

batch_size=1028
#when using convolution we do not want to splice data from different files together.
# the splice is used to generate a list of indices which have enough surrounding 
# frames to splice
splice_size=5

if load==True:
    print('loading data')
    [Train_index,Val_index,Test_index,l_data ,f_data]=load_dataset(f_nodes,l_nodes,splice_size)
else:
    print('creating train, val and test sets')
    [Train_index, Val_index,Test_index]=Split_dataset(f_nodes,l_nodes,splice_size)  

def build_dnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual: #195
    network = lasagne.layers.InputLayer(shape=(None, 1, 11, 39),
                                        input_var=input_var)
    network = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(network, p=.2),
           num_units=250,
           nonlinearity=lasagne.nonlinearities.rectify)
   #network = lasagne.layers.DenseLayer(
   #        lasagne.layers.dropout(network, p=.2),
   #        num_units=512,
   #        nonlinearity=lasagne.nonlinearities.rectify)
   #network = lasagne.layers.DenseLayer(
   #        lasagne.layers.dropout(network, p=.2),
   #        num_units=256,
   #        nonlinearity=lasagne.nonlinearities.rectify)
   #network = lasagne.layers.DenseLayer(
   #        lasagne.layers.dropout(network, p=.2),
   #        num_units=128,
   #        nonlinearity=lasagne.nonlinearities.rectify)
   

    # And, finally, the 10-unit output layer with dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=4,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

#iterate minibatches where data is loaded to memory at every iteration. Use if 
# data is too big for working memory
def iterate_minibatches(index,batchsize,splice_size, shuffle):  
    if shuffle:
        np.random.shuffle(index)
    for start_idx in range(0, len(index) - batchsize + 1, batchsize):       
        if shuffle:
            excerpt = index[start_idx:start_idx + batchsize]
        else:
            excerpt = [x for x in range (start_idx, start_idx + batchsize)]
        inputs=[]
        targets=[]
        for ex in excerpt:
            # retrieve the frame indicated by index and splice it with surrounding frames
            inputs.append([f_nodes[ex[1]][ex[2]+x] for x in range (-splice_size,splice_size+1)])
            targets.append(l_nodes[ex[1]][ex[2]][7])
        shape= np.shape(inputs)
        inputs = np.float32(np.reshape(inputs,(shape[0],1,shape[1],shape[2])))
        targets = np.uint8(targets)
        yield inputs, targets
# iterate minibatches in case data can fit in working memory. 
def iterate_minibatches_mem(index, features, labels, batchsize,splice_size, shuffle):
    assert len(features) == len(labels)
    if shuffle:
        np.random.shuffle(index)
    for start_idx in range(0, len(index) - batchsize + 1, batchsize):
        excerpt = index[start_idx:start_idx + batchsize] 
        #splice features
        inputs=[]
        targets=[]
        for ex in excerpt:
            inputs.append([features[ex[0]+x] for x in range (-splice_size,splice_size+1)])
            targets.append(labels[ex[0]][-1])
            
        shape= np.shape(inputs)
        # in rare cases a excerpt of size 1 may be left such that the shape of t        # input returns just 2 values
        if len(shape)!=3:
            temp=(1,shape[0],shape[1])
            shape=temp
        inputs = np.float32(np.reshape(inputs,(shape[0],1,shape[1],shape[2])))
        targets = np.uint8(targets)
        yield inputs, targets
        
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=5):
    # Load the dataset
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_dnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    ################################################################
    #uncomment the for loop and comment the 
    #the while loop  to train for a set amount of epochs
    
    #for epoch in range(num_epochs):
    ################################################################

    ################################################################
    #use the while loop to train for as long as the validation accuracy is   
    #increasing, comment the for loop above
    val_acc=1
    prev_val_acc=0
    epoch=0
    while val_acc > prev_val_acc:
    ################################################################

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(Train_index,batch_size,splice_size,shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        prev_val_acc=val_acc
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(Val_index,batch_size,splice_size,shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        epoch=epoch+1
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(Test_index,batch_size,splice_size,shuffle=True):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
