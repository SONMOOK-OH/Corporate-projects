# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:06:16 2019

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


def Init_Shifting(X1, X2):
    
    N = len(X2)-1
    X1=X1[0:N]
    for i in range(0,N):
        X2[i] = X2[i+1]
    X2=X2[0:N]
    
    return X1, X2