#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:47:50 2017

@author: danny
"""
from scipy.fftpack import fft
import numpy
# provides some basic preprocessing functions for audio files, such as
# padding the frames, hammingwindow for the  frames, data preemphasis and fourrier
# transform 


def four(frame, fs, windowsize):
   # pad each frame to 512 samples  N.B. pad size is hardcoded for 25ms windows at
   #16KHz
   frame=numpy.pad(frame,[(0,0),(0,112)],'constant',constant_values=0)
        # fourrier transform, works on matrices and arrays (not on 1 dim arrays)
        # set cutoff at the nyquist frequency
   cutoff = int(windowsize/2)+56
        # perform fast fourier transform
   Y=fft(frame)
    
        # take absolute power and collapse spectrum around nyquist freq 
   Yamp=2/windowsize* numpy.abs(Y[:,0:cutoff])
        # first amp is not to be doubled
   Yamp[0]=Yamp[0]/2
   return (Yamp)


def pad (data,windowsize,frameshift):
    # function to pad the data to fit the frameshift
    contextsize= (windowsize-frameshift)/2
    padsize= contextsize - numpy.mod(data.size,frameshift) 
    # if needed add padding to the end of the data
    if padsize > 0:
        data = numpy.append(data,numpy.zeros(int(padsize)))
    #always add padding to the front of the data
    data = numpy.append(numpy.zeros(int(contextsize)),data)
    return(data)
    
def preemph(data,alpha):
    # preemphasises the data
    x= data
    y=data*alpha
    # pad rows with a 0
    y=numpy.insert(y,0,0,1)[:,:-1]
    #subtract t-1*alpha from each sample
    z=x-y  
    return(z)
    
def hamming(data):
    # apply hamming windowing to a frame of data
    L= numpy.shape(data)[1]
    hammingwindow = 0.54-(0.46*numpy.cos(2*numpy.pi*numpy.arange(L)/(L-1)))
    data=numpy.multiply(data,hammingwindow)
    return data