#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:05:37 2017

@author: danny
"""
import numpy as np
import math

# contains function to handle the h5 data file for the DNNs. Split_dataset
# creates indices for train test and val set, for the minibatch iterator to load data: use 
# if the data is to big for working memory. Load_dataset actually loads all data in working memory
# and splits into train test and val set. use if data fits working memory

def Split_dataset(f_nodes,splice_size):
# prepare dataset. This is meant for datasets to big for working memory, data 
# is split into train test and validation based on indexes. The indexes are 
# used to retrieve data in minibatches. load_dataset is faster but only useable if the dataset fits in working memory
    index=[]
    offset=0
    # index triple, first number is the index of each frame. Because different wav files are stored in different
    #leaf nodes of the h5 file, we also keep track of the node number and the index of the frame internal to the node.
    for x in range (0,len(f_nodes)):
        for y in range (splice_size,len(f_nodes[x])-splice_size):
            index.append((y+offset,x,y))
        offset=offset+len(f_nodes[x])
    #shuffle data
    np.random.shuffle(index)
    # get length of dataset
    data_size=len(index)
    #split in train, validation and test. test set defaults to everything not in train or validation
    train_size = math.floor(data_size*0.8)
    val_size= math.floor(data_size*0.1)
    Train_index = index[0:train_size]
    Val_index = index[train_size:train_size+val_size]  
    Test_index = index[train_size+val_size:]
    return (Train_index, Val_index, Test_index)

def load_dataset(f_nodes,l_nodes,splice_size):    
# load feature data. only use when data fits in working memory. Use split data otherwise,
# which creates just indexes so that data can be loaded by the minibatch iterator when needed
    
    # if we splice the features in the time dimension we need to skip some frames at the edges of each
    # file (or pad the file with empty frames feel free to implement it). otherwise we can load all data
    if splice_size >0:
        for x in f_nodes:
            if 'f_data' in locals():
                f_data= np.concatenate([f_data,np.array(x[splice_size:-splice_size])],0) 
            else:
                f_data=np.array(x[splice_size:-splice_size])

        for x in l_nodes:
            if 'l_data' in locals():
                l_data= np.concatenate([l_data,np.array(x[splice_size:-splice_size])],0) 
            else:
                l_data=np.array(x[splice_size:-splice_size])
    else: 
        for x in f_nodes:
            if 'f_data' in locals():
                f_data= np.concatenate([f_data,np.array(x)],0) 
            else:
                f_data=np.array(x)

        for x in l_nodes:
            if 'l_data' in locals():
                l_data= np.concatenate([l_data,np.array(x)],0) 
            else:
                l_data=np.array(x)
                
    index =[x for x in range (0,len(f_data))]
    np.random.shuffle(index)
    data_size= len(f_data)
    train_size = math.floor(data_size*0.8)
    val_size= math.floor(data_size*0.1)
    test_size= (data_size-train_size)-val_size
    shape=np.shape(f_data)
    
    Train_index = index[0:train_size]
    Val_index = index[train_size:train_size+val_size]  
    Test_index = index[train_size+val_size:]
    
    print ('slice training data')
    # slice training data and labels from the table
    X_train = np.float32(np.reshape(f_data[index[0:train_size]],(train_size,1,1,shape[1])))
    y_train = np.uint8(l_data[index[0:train_size]])
    # slice validation data
    print ('slice validation data')
    X_val = np.float32(np.reshape(f_data[index[train_size:train_size+val_size]],(val_size,1,1,shape[1])))
    y_val = np.uint8(l_data[index[train_size:train_size+val_size]].astype(int))
    # slice test data
    print('slice test data')
    X_test = np.float32(np.reshape(f_data[index[train_size+val_size:]],(test_size,1,1,shape[1])))
    y_test = np.uint8(l_data[index[train_size+val_size:]].astype(int))
    return (Train_index, Val_index, Test_index, l_data ,f_data)  