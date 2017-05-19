#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:05:37 2017

@author: danny
"""
import numpy as np
import math
import pickle
# contains function to handle the h5 data file for the DNNs. Split_dataset
# creates indices for train test and val set, for the minibatch iterator to load data: use 
# if the data is to big for working memory. Load_dataset actually loads all data in working memory
# and splits into train test and val set. use if data fits working memory

def Split_dataset(f_nodes,l_nodes,splice_size):
# prepare dataset. This is meant for datasets to big for working memory, data 
# is split into train test and validation based on indexes. The indexes are 
# used to retrieve data in minibatches. load_dataset is faster but only useable if the dataset fits in working memory
    index=[]
    offset=0
    try:
        f=open('/data/Preprocessing/DNN/index.py','rb')
        x=True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    index.append(y)
            except:
                x=False
    except:
    # index triple, first number is the index of each frame. Because different wav files are stored in different
    #leaf nodes of the h5 file, we also keep track of the node number and the index of the frame internal to the node.
        f=open('/data/Preprocessing/DNN/index2.py', 'wa')
        print('creating index')
        for x in range (0,len(f_nodes)):
            temp=[]
            for y in range (splice_size,len(f_nodes[x])-splice_size):
            # 999 was used to indicate out of vocab values. These are removed
            # from the training data, however they are still valid for splicing
            # with a valid training frame
                if l_nodes[x][y][1]!=b'999':
                    temp.append((y+offset,x,y))
            np.random.shuffle(temp)
            for i in temp:
                index.append(i)
            offset=offset+len(f_nodes[x])
            pickle.dump(temp,f)
        f.close()
    # get length of dataset
    data_size=len(index)
    #split in train, validation and test. test set defaults to everything not in train or validation
    train_size = int(math.floor(data_size*0.008))
    val_size= int(math.floor(data_size*0.001))
    test_size = int(math.floor(data_size*0.001))
    Train_index = index[0:train_size]
    Val_index = index[train_size:train_size+val_size]  
    Test_index = index[train_size+val_size:train_size+val_size+test_size]
    return (Train_index, Val_index, Test_index)

def load_dataset(f_nodes,l_nodes,splice_size):    
# load feature data. only use when data fits in working memory. Use split data otherwise,
# which creates just indexes so that data can be loaded by the minibatch iterator when needed
        # for each node (audio file) we skip splice_size number of frames at the front and back of the file.
        # if we do not do that, the first frame would be spliced with frames from the previous audio file. Another 
        # option would be to pad each node with splice_size zero frames. 
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

    # for splicing we need to keep our data chronologically sorted, so we create a random
    # index to divide our data into test, train and val sets.
    index=[]
    offset=0
    for x in range (0,len(f_nodes)):
        for y in range (splice_size,len(f_nodes[x])-splice_size):
            index.append((y+offset,x,y))
        offset=offset+len(f_nodes[x])
    #shuffle data
    np.random.shuffle(index)
    # get length of dataset
    data_size=len(index)
    #split in train, validation and test. test set defaults to everything not in train or validation
    train_size = int(math.floor(data_size*0.8))
    val_size= int(math.floor(data_size*0.1))
    test_size = int(math.floor(data_size*0.1))
    Train_index = index[0:train_size]
    Val_index = index[train_size+1000000:train_size+val_size+1000000]
    Test_index = index[train_size+val_size+2000000:train_size+val_size+test_size+2000000]  
    return (Train_index, Val_index, Test_index, l_data ,f_data)  
