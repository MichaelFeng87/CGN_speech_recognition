#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:19:53 2017

@author: danny
"""
# possible improvements: setting parameters from command line,
# remove the need for some hard coding. More options to stop/skip some
# preprocessing steps. Only fBANKS and MFCCs enabled now.

#N.B. system paths are hard coded


from process_data import proc_data
# regex for parsing of the label files
pattern= '"N[0-9]+_SEG"'
# regex for checking the file extensions
f_ex ='fn[0-9]+.' 
# set data path for label files
#l_path ="/scratch/danny/kaldi/egs/myexp/exp/tri4a_ali/split_ali"
l_path="/data/split_ali"
# set data path for audio files
d_path = "/data/comp-o/nl"
# set data path for phoneme-articulatory feature conversion table
conv_table ="/data/Preprocessing/feature_table.txt"
#files to save the features and labels in
data_loc = '/data/processed/mfcc'
# indicate whether you use the CGN awd transcripts (1) or transcripts created
# with kaldi (0)
CGN=0
# some parameters for mfcc creation
params=[]
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired mel filter banks
nfilters = 24
# windowsize and shift in time
t_window=.025
t_shift=.01
# option to get raw filterbank energies (true) or mfccs (false)
filt = False
# option to include delta and double delta features (true)
use_deltas=True
# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(filt)
params.append(data_loc)
params.append(use_deltas)
[mfcc,labels]=proc_data (pattern,f_ex,params,l_path,d_path,conv_table,CGN)

