#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:41:47 2017

@author: danny
"""
from preproc import four,pad,preemph, hamming
from filters import apply_filterbanks,filter_centers, create_filterbanks
from scipy.fftpack import dct
import numpy
import math

# functions to get the cepstrum and delta features of the cepstrum

def get_cepstrum(frames,fs, windowsize,filterbanks,filt):
    # this function calls other preprocessing steps and returns the cepstrum
    # get the frequency spectrum of the frames
    freq_spectrum = four(frames,fs,windowsize)
    #apply the filterbanks to the frequency spectrum
    filtered_freq=apply_filterbanks(freq_spectrum,filterbanks)
        # if the option f_banks is choosen, return the filterbanks
        # else return the cepstrum
    if filt:
        cepstrum = filtered_freq
    else:
        cepstrum = dct(filtered_freq[:,1:])
        # remove the first coefficient.
        cepstrum = cepstrum[:,1:13]
    return (cepstrum)
    
def delta (data,n):
    dt=[]
    for j in range (0,data.shape[0]):
        temp=[]
        for i in range (1,n+1):
            if j-i >=0 and not j+i > (data.shape[0]-1):
                temp.append(n*(data[j+i] - data[j-i]))
                
            elif j-i <0:
                temp.append(n*(data[j+i]))
            else:
                temp.append(n* (0 - data[j-i]))
            temp2 = 2 * sum([x*x for x in range (1,n+1) ])
        dt.append(sum(temp)/temp2)
    return (numpy.array(dt))

def get_mfcc (input_data,alpha,nfilters,windowsize,frameshift,filt,use_deltas):
    #sampling frequency
    fs=input_data[0]

    #determine the number of frames to be extracted
    #subtract the surrounding window as the first frame starts from 
    # the contextsize not 0.
    nframes=math.floor(input_data[1].size/frameshift)
    # pad the data
    data=pad(input_data[1],windowsize,frameshift)
    # slice the frames from the wav file
    # keep a list with the frames and all the values of the samples and 
    # list with the start and end sample# of each frame
    frames=[]
    frame_nrs=[]
    energy =[]
    for x in range (0,nframes):
        frame=data[x*frameshift:x*frameshift+windowsize]
        frame_nrs.append([x*frameshift,x*frameshift+frameshift])      
        # frame energy and frame
        
        energy.append(numpy.log(numpy.sum(numpy.square(frame),0)))
        frames.append(frame)
    frames= numpy.array(frames)
    energy= numpy.array(energy)
    
    # if energy is 0 , the log can not be taken(results in -inf) so we set the 
    # log energy to -50 (log of 2e-22 or approx 0 )
    energy[energy==numpy.log(0)]=-50
    #apply preemphasis
    frames=preemph(frames,alpha)
    #apply hamming window 
    frames=hamming(frames)
    
    # frequency range for filter centers (fft bins spaced equally over the freqs up to the nyquist rate)
    # windowsize/2 is the # of fft bins, 1/(2/fs) is the nyquist frequency
    
    #xf = numpy.linspace(64.0, 1.0/(2/fs), (windowsize/2)+56)
    
    # frequency range for filter response
    xf= numpy.linspace(0.0, 1.0/(2/fs), (windowsize/2)+56)
    # get the filter centers
    fc = filter_centers (nfilters,xf)

    # create filterbanks
    filterbanks = create_filterbanks(nfilters,xf,fc)

    # get the cepstum and add it to the energy
    mfcc= get_cepstrum(frames,fs, windowsize,filterbanks,filt)
    mfcc = numpy.concatenate([energy[:,None],mfcc],1)
    #plt.plot(xf,numpy.log10(x))
    # add delta and double delta features if needed
    if use_deltas:
        single_delta= delta (mfcc,2)
        double_delta= delta(single_delta,2)
        mfcc= numpy.concatenate([mfcc,single_delta,double_delta],1)
    
    return (mfcc,frame_nrs)
