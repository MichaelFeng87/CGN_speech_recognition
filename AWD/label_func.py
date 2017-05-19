#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:21:45 2016

@author: danny
"""
import gzip
import re
import numpy as np
import math
import codecs
# functions for cleaning the transcript and labelling the data

def cleanup (text,pattern):
    #cleans up the raw text file
    match= pattern.search(text)
    begin=match.span()[0]
    cl_text=text[begin:-1]
    return cl_text
    
def parse_transcript (pattern, loc):
    # parse the raw transcript
    regex= re.compile(pattern)
    try:
        with gzip.open(loc,'rb', encoding='iso-8859-1') as file:
            x=file.read()
    except:
        with codecs.open(loc,'rb',encoding= 'iso-8859-1') as file:
            x=file.read()
    # turn the input data into a string
    raw_trscrpt = str(x)
    # cleanup removes the parts of the transcript we dont need 
    cleaned_trscrpt = cleanup(raw_trscrpt,regex)
    # split the transcript in seperate lines
    split_trscrpt = cleaned_trscrpt.split('\\r\\n')
    # some files just have \\n instead of \r\n for some reason
    if len (split_trscrpt)==1:
        split_trscrpt=split_trscrpt[0].split('\\n')
    if len (split_trscrpt)==1:
        split_trscrpt = cleaned_trscrpt.split('\r\n')
    if len (split_trscrpt)==1:
        split_trscrpt=split_trscrpt[0].split('\n')
    trscpt_size = int(split_trscrpt[3])*3
    # at last remove some empty lines and header lines so only the phonemes and their
    # start and end time are left
    final_trscrpt= split_trscrpt[4:trscpt_size+4]
    # regex for the pattern in the text file which indicates the section
    # we are interested in.    
    return (final_trscrpt)
    
def label_transcript (trans, fs,cgndict):
    # labels the cleaned transcript with articulatory features
    l_trans=[]
    if np.mod(len(trans),3) == 0:
        # for each phone extract begin and end sample# and articulatory features
        
        # set a default in case an unknown phone is found (a mistake in CGN)
        # default is set to sil phone for the first phone as this is always
        # sil. After the first phone the scripts defaults to the previous phone if
        # an unknown is found
        default=['40','8','8','2','1','4','1','1']
        for x in range (0,int(len(trans)/3)):
            af= trans[x*3:(x*3)+3]
            if len(l_trans)>0:
                label=([int(float(af[0])*fs),int(float(af[1])*fs)],cgndict.get(af[2],l_trans[-1][1]))
            else:
                label=([int(float(af[0])*fs),int(float(af[1])*fs)],cgndict.get(af[2],default))
            l_trans.append(label)
    return (l_trans)

def label_frames (nframes, trans_labels,frameshift):
    # labels the data frames, input is #frames, 
    #labelled data list
    ld=[]
    # calculate begin and end samples of the frames
    t = [[t*frameshift,(t*frameshift)+frameshift] for t in range (0,nframes)]
    for t in t:
        # if  the begin sample# and end sample# of the frame are smaller
        # than the end sample # of the phoneme, the frame is fully overlapping
        # only one phoneme, and gets that label.
        if t[0] < trans_labels[0][0][1] and t[1] < trans_labels[0][0][1]:
            ld.append(trans_labels[0][1])
        # however if the begin sample# is smaller but the end sample# is larger
        # than the end sample# of the phoneme, the frame is partially on 2 phonemes
        elif t[0] < trans_labels[0][0][1] and t[1] >= trans_labels[0][0][1]:
            # so the frame gets the label of the phoneme which more than half of the
            # frame is overlapping (can easily be checked by looking at the middle 2
            # samples). In case of a tie, the frame gets the label of the first phoneme.
            if math.floor((t[0]+t[1]-1)/2) <trans_labels[0][0][1]:
                ld.append(trans_labels[0][1])
                trans_labels.pop(0)
            elif math.floor((t[0]+t[1]-1)/2) >=trans_labels[0][0][1]:
                trans_labels.pop(0)
                ld.append(trans_labels[0][1])
    return(ld)
    

            