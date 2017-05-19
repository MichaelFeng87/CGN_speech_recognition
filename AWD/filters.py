#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:23:01 2017

@author: danny
"""
# functions for creating and applying the filter banks
from melfreq import freq2mel, mel2freq
import numpy

def create_filterbanks (nfilters,freqrange,fc):
    # function to create filter banks. takes as input
    # the number of filters to be created, the frequency range and the
    # filter centers
    filterbank=[]
    # for the desired # of filters do
    for n in range (0,nfilters):
        # set the begin center and end frequency of the filters
        begin = fc[n]
        center= fc[n+1]
        end = fc[n+2]
        f=[]
        # create triangular filters
        for x in freqrange:
            # 0 for f outside the filter
            if x < begin:
                f.append(0)
            #increasing to 1 towards the center
            elif begin <= x and x <= center:
                f.append((x-begin)/(center-begin))
            # decreasing to 0 upwards from the center
            elif center <= x and x <= end:
                f.append((end-x)/(end-center))
            # 0 outside filter range
            elif x > end:
                f.append(0)
        filterbank.append(f)
    return(filterbank)
    
def filter_centers(nfilters,freqrange):
    # calculates the center frequencies for the mel filters
    
    # get the begin and end of the filter frequency range 
    # expressed in mels
    melbegin= freq2mel(freqrange[0])
    melend=freq2mel(freqrange[-1])
    
    #space the filters equally in mels
    spacing=numpy.linspace(melbegin,melend,nfilters+2)
    #back from mels to frequency
    spacing= mel2freq(spacing)
    # round the filter frequencies to the nearest availlable fft bin and return
    # the centers for the filters. Note: the first and last value are note flter 
    #centers but the start and end of the frequency range. the 2nd entry is the 
    #center of the first filter, and the start of the 2nd filter, the 3rd entryis the 
    #end of the 1st filter, center of the 2nd filter and start of the 3rd filter etc.
    filters = [freqrange[numpy.argmin(numpy.abs(freqrange-x))]for x in spacing]
    
    return (filters)
    
def apply_filterbanks(data,filters):
    # function to apply the filterbanks
    # dot product of data and filterbanks
    filtered_freq = numpy.log(numpy.dot(data,numpy.transpose(filters)))  
    # same as with energy, taking the log of a filter bank with 0 power results in -inf
    # we approximate 0 power with -50 the log of 2e-22
    filtered_freq[filtered_freq == numpy.log(0)] = -50     
    return (filtered_freq)