# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:27:58 2019

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

def MFCC_FULL(sig, frame_size = 0.025, frame_stride = 0.01, sample_rate=48000,pre_emphasis = 0.97):
    
    emphasized_signal = np.append(sig[0], sig[1:] - pre_emphasis * sig[:-1])
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames3 = pad_signal[indices.astype(np.int32, copy=False)]
    frames3 *= np.hamming(frame_length)
    
    
    NFFT = 256
    mag_frames = np.absolute(np.fft.rfft(frames3, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    
    nfilt = 40
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks1 = np.dot(pow_frames, fbank.T)
    filter_banks1 = np.where(filter_banks1 == 0, np.finfo(float).eps, filter_banks1)  # Numerical Stability
    filter_banks1 = 20 * np.log10(filter_banks1)  # dB
    
    
    num_ceps = 12
    cep_lifter=22
    mfcc1 = dct(filter_banks1, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc1.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc1 *= lift  #*
    filter_banks1 -= (np.mean(filter_banks1, axis=0) + 1e-8)
    
    mfcc1 -= (np.mean(mfcc1, axis=0) + 1e-8)
    
    delta=np.zeros((num_frames,num_ceps))
    for i in range(1, num_frames-1):
        for j in range(num_ceps):
            delta[i][j]=(mfcc1[i+1][j]-mfcc1[i-1][j])/2
            
            
            
    delta_delta=np.zeros((num_frames,num_ceps))
    for i in range(1, num_frames-1):
        for j in range(num_ceps):
            delta_delta[i][j]=(delta[i+1][j]-delta[i-1][j])/2
    
    Final_MFCC=(np.vstack((mfcc1.T,delta.T,delta_delta.T))).T
    
    return mfcc1

def MFCC_exceptFraming(frames, sample_rate=48000, singleframeflag=0, num_frames=0):
        
    NFFT = 256
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    
    nfilt = 40
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks1 = np.dot(pow_frames, fbank.T)
    filter_banks1 = np.where(filter_banks1 == 0, np.finfo(float).eps, filter_banks1)  # Numerical Stability
    filter_banks1 = 20 * np.log10(filter_banks1)  # dB
    if singleframeflag == 1:
        filter_banks1 = (filter_banks1).reshape((1,nfilt))
    
    
    num_ceps = 12
    cep_lifter=22
    mfcc1 = dct(filter_banks1, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc1.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc1 *= lift  #*
    filter_banks1 -= (np.mean(filter_banks1, axis=0) + 1e-8)
    
    mfcc1 -= (np.mean(mfcc1, axis=0) + 1e-8)
    if singleframeflag == 0:
        
        delta=np.zeros((num_frames,num_ceps))
        for i in range(1, num_frames-1):
            for j in range(num_ceps):
                delta[i][j]=(mfcc1[i+1][j]-mfcc1[i-1][j])/2
                
                
                
        delta_delta=np.zeros((num_frames,num_ceps))
        for i in range(1, num_frames-1):
            for j in range(num_ceps):
                delta_delta[i][j]=(delta[i+1][j]-delta[i-1][j])/2
        
        Final_MFCC=(np.vstack((mfcc1.T,delta.T,delta_delta.T))).T
    
    return mfcc1
    


def Frame_generation(sig, frame_size = 2/75, frame_stride = 0.01, sample_rate=48000,pre_emphasis = 0.97, empha_signal_flag=0, hamming_flag=0):
    if empha_signal_flag==1:
        emphasized_signal = np.append(sig[0], sig[1:] - pre_emphasis * sig[:-1])
    else:
        emphasized_signal = sig
        

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    if hamming_flag==1:
        frames *= np.hamming(frame_length)
    else:
        frames *= 1
    
    return frames, num_frames



