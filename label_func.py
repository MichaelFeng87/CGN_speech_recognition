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
    
def parse_transcript (pattern, loc,CGN):
    # parse the raw transcript
    regex= re.compile(pattern)
    try:
        with gzip.open(loc,'rb') as file:
             raw_trscrpt= file.read().decode('latin-1') 
    except:
        with codecs.open(loc,'rb',encoding= 'latin-1') as file:
             raw_trscrpt=file.read()
    # there are some slight differences between CGN transcripts and our own created kaldi transcripts
    if CGN==1:
        # awd transcripts have an orthographic and phonetic part, remove the part we do not need.
        cleaned_trscrpt = cleanup(raw_trscrpt,regex)    
        # split the transcript in seperate lines
        split_trscrpt = cleaned_trscrpt.splitlines()
        # replace CGN silence "" notation with 'sil'
        for x in range(0, len(split_trscrpt)):
            split_trscrpt[x]= split_trscrpt[x].replace('""', 'sil').replace('"','')
        # get the size of transcript from the header
        trscpt_size = int(split_trscrpt[3])*3
        # at last remove some empty lines and header lines so only the phonemes and their
        # start and end time are left
        final_trscrpt= split_trscrpt[4:trscpt_size+4]
    else:
        # our transcripts do not have headers empty lines etc.
        final_trscrpt = raw_trscrpt.splitlines()
    return (final_trscrpt)
    
def label_transcript (trans, fs,cgndict):
    # labels the cleaned transcript with articulatory features
    l_trans=[]
    if np.mod(len(trans),3) == 0:
        # for each phone extract begin and end sample# and articulatory features
        
        # set a default just in case an unknown phoneme is found.
        # default is set to sil phone for the first phone as this is always
        # sil. After the first phone the scripts defaults to the previous phone if
        # an unknown is found
        default=cgndict['sil']
        for x in range (0,len(trans),3):
            af= trans[x:x+3]
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
    

            
