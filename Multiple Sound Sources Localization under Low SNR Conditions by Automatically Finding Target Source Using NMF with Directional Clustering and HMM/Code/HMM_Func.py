# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:51:40 2019

@author: sonoh
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from hmmlearn import hmm

import os
import warnings
import scipy.stats as sp
import pandas as pd
from scipy import signal
import librosa
import scipy.io.wavfile as wav
from sklearn.cluster import KMeans
import math
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft, dct
from scipy.signal import find_peaks, find_peaks_cwt
import scipy.io
from sklearn import preprocessing
from sklearn.decomposition import NMF
import cmath


from MFCC_Frame_Func import MFCC_FULL


def getTransmatPrior(inumstates, ibakisLevel):
    transmatPrior = (1 / float(ibakisLevel)) * np.eye(inumstates)

    for i in range(inumstates - (ibakisLevel - 1)):
        for j in range(ibakisLevel - 1):
            transmatPrior[i, i + j + 1] = 1. / ibakisLevel
    
    for i in range(inumstates - ibakisLevel + 1, inumstates):
        for j in range(inumstates - i):
            transmatPrior[i, i + j] = 1. / (inumstates - i)

    return transmatPrior


def initByBakis(inumstates, ibakisLevel):
    startprobPrior = np.zeros(inumstates)
    startprobPrior[0: ibakisLevel - 1] = 1/float((ibakisLevel - 1))
    transmatPrior = getTransmatPrior(inumstates, ibakisLevel)
    return startprobPrior, transmatPrior


def HMM_Audioread_feature(Final_MFCC1, Final_MFCC2):
    warnings.filterwarnings("ignore")

    fpaths = []
    labels = []
    spoken = []
    features = []
    
    for f in os.listdir('audio'):
        for w in os.listdir('audio/' + f):
            print('audio1/' + f + '/' + w)
            fpaths.append('audio/' + f + '/' + w)
            labels.append(f)
            if f not in spoken:
                spoken.append(f)
    print('Words spoken:', spoken)
    
    
    N=24000
    for n, file in enumerate(fpaths):
        samplerate, d = wavfile.read(file)
        K=len(d.shape)
        if K >1:
            d=d.reshape((len(d),len(d[:][0]))) 
        else:
            d=d.reshape((len(d),len([d[:][0]])))
    
        if max(d[:,0]) > 1:
            d=d[0:N,0]/max(d[:,0])
        Final_MFCC = MFCC_FULL(sig=d, sample_rate=samplerate)
        features.append(Final_MFCC)
    
    features.append(Final_MFCC1)
    features.append(Final_MFCC2)

    return features