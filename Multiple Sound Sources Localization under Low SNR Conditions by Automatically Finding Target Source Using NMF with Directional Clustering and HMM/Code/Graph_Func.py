# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:58:50 2019

@author: sonmookoh
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


def Graph_Waveform_InitSources(X1, X2, white1, white2, fs):
    
    lim=0.15
    N = len(X2)
    time = np.arange(N) / float(fs)
    
    plt.subplot(2, 1, 1)
    plt.plot(X1)
    plt.ylim(-lim,lim)
    plt.gcf().set_size_inches(10.5, 10.5)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(X2)
    plt.ylim(-lim,lim)
    plt.grid()
    plt.show()
    
    plt.plot(white1)
    plt.gcf().set_size_inches(10.5, 5.3)
    plt.ylim(-lim,lim)
    plt.grid()
    plt.show()
    
    
    plt.subplot(2, 1, 1)
    plt.plot(time, white1[0:N], time, X1)
    plt.ylim(-lim,lim)
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['Noise', 'Target'])
    plt.gcf().set_size_inches(10.5, 10.5)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(time, white2[0:N], time, X2)
    plt.ylim(-lim,lim)
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['Noise', 'Target'])
    plt.grid()
    plt.show()
    
    
def Graph_Waveform_finalComparison(s1, s3, s2, s4, xrec1_NMF, xrec_NMF, fs):
    
    lim=0.15
    N = len(s1)
    time = np.arange(N) / float(fs)
    
    plt.figure()
    plt.plot(time, s1, time, s3, time, (xrec1_NMF))
    plt.ylim(-lim,lim)
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['SUM', 'clean', 'Final'])
    plt.grid()
    plt.show()
    
    
    plt.figure()
    plt.plot(time, s2, time, s4, time, (xrec_NMF))
    plt.ylim(-lim,lim)
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['SUM', 'clean', 'Final'])
    plt.grid()
    plt.show()
    
    
def Graph_Spectrogram(s, fs, ylim=20000):
    
    f, t, Zxx = signal.stft(s, fs, nperseg=256)
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylim(0, ylim)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    
    
    
def Graph_Kmeans(centroids, k, labels, C):
    
    print(centroids) # From sci-kit learn
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['*','o', 'x', '+', 'v', '^', '<', '>', 's', 'd','H']
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(k):
             points = np.array([C[j] for j in range(len(C)) if labels[j] == i])
             ax.scatter(points[:, 0], points[:, 1], s=10, marker=markers[i])
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#050505',s=100)
    plt.grid()
                
    fig, ax = plt.subplots(figsize=(5,5))                  
    for i in range(k):         
            ax.scatter(centroids[i, 0], centroids[i, 1], marker=markers[i], c='#050505',s=100)                  
    plt.grid() 
    
def Graph_Kmeans_AfterClustering(centroids, Label_target, Label_noise, labels, C):
    
    markers = ['*','o', 'x', '+', 'v', '^', '<', '>', 's', 'd','H']
    
    fig, ax = plt.subplots(figsize=(5,5))
    i=Label_target[0]
    points = np.array([C[j] for j in range(len(C)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=10, marker=markers[i])
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#050505',s=100)
    plt.grid()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    i=Label_noise[0]
    points = np.array([C[j] for j in range(len(C)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=10, marker=markers[i])
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#050505',s=100)
    plt.grid()
    plt.show()
    
    
    