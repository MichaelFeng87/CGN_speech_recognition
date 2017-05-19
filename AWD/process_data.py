#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016

@author: danny
"""
from label_func import label_frames, parse_transcript, label_transcript
from data_functions import list_files, phoneme_dict, check_files
from cepstrum import get_mfcc
from scipy.io.wavfile import read
import numpy
import tables

def proc_data (pattern,f_ex,params,l_path,d_path,conv_table):
    # get list of audio and transcript files 
    audio_files = [d_path+"/"+x for x in list_files(d_path)]
    audio_files.sort()  
    
    label_files = [l_path+"/"+ x for x in list_files (l_path)]
    label_files.sort()
    
    # create h5 file for the processed data
    data_file = tables.open_file(params[5] + '.h5', mode='a')
    
    # create pytable atoms
    feature_shape= (params[1]+1)*3
    f_atom= tables.Float64Atom()
    # N.B. label size is hard coded. It provides phoneme and 7 articulatory feature
    # labels
    l_atom = tables.StringAtom(itemsize=5)
    # create a feature and label group branching of the root node
    features = data_file.create_group("/", 'features')    
    labels = data_file.create_group("/", 'labels') 
    # create a dictionary from the conv table
    cgndict = phoneme_dict(conv_table)
    
    # check if the audio and transcript files match 
    if check_files(audio_files, label_files, f_ex):
    
    # len(audio_files) 
        for x in range (0,len(audio_files)): #len(audio_files)
            print ('processing file ' + str(x) )
            # create new leaf nodes in the feature and leave nodes for every audio file
            f_table = data_file.create_earray(features, audio_files[x][-12:-4], f_atom, (0,feature_shape),expectedrows=100000)
            print('l_table')
            l_table = data_file.create_earray(labels, audio_files[x][-12:-4], l_atom, (0,8),expectedrows=100000)
          
            # read audio samples
            input_data = read(audio_files[x])
            # sampling frequency
            fs=input_data[0]
            # get window and frameshift size in samples
            s_window = int(fs*params[2])
            s_shift=int(fs*params[3])
            
            # create mfccs
            [mfcc, frame_nrs] = get_mfcc (input_data,params[0],params[1],s_window,s_shift,params[4])    
            
            # read datatranscript
            trans = parse_transcript(pattern, label_files[x] )
            # convert phoneme transcript to articulatory feature transcript
            l_trans=label_transcript(trans,fs,cgndict)
            nframes = mfcc.shape[0]
            # label frames using the labelled transcript
            l_data= numpy.array(label_frames(nframes,l_trans,s_shift))
            
            # append new data to the tables
            f_table.append(mfcc)
            l_table.append(l_data)
    else: 
        print ('audio and transcript files do not match')
    # close the output files
    data_file.close()
    data_file.close()
    return(mfcc,l_data)