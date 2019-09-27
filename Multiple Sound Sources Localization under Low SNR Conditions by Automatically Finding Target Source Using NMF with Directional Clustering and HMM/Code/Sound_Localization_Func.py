# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:47:09 2019

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

def GCC_Phat(s1, s2, threshold=0.2):
    
    N = len(s2)
    pad1 = np.zeros(len(s1))
    pad2 = np.zeros(len(s2))
    s11 = np.hstack([s1,pad1])
    s12 = np.hstack([pad2,s2])
    f_s1 = fft(s11)
    f_s2 = fft(s12)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = abs(f_s)
    denom[denom < 1e-6] = 1e-6
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    gg1=np.abs(ifft(f_s))[0:]


    x=gg1[N-15:N+16]
    
        
    
    peaks, _ = find_peaks(x, prominence=threshold)
    
    return peaks, x


def Source_Coordinate_Generation(num_direction, x, peaks, delayMAX, fs, c, micdist, Ratio_origin):
    
    coordinate=np.zeros((num_direction,2))              
    for j in range(num_direction):
        sort=sorted(x[peaks], reverse=True)
        zzz=[]
        for i in range(num_direction):
            zzz.append(np.array(np.where(x == sort[i])))
        zzz=np.array(zzz).reshape(-1)
        
        #K=zzz[j]
        K=peaks[j]         
        delay=K-delayMAX        
        tau=delay/fs        
        a=tau*c/micdist
        
        if a>1:
            a=1
        elif a<-1:
            a=-1
        
        b=np.arccos(a)
        coordinate[j]=np.array([np.sin(b/2),np.cos(b/2)])
    print(coordinate)
    Ratio_origin.append(max(x[peaks])) 
    plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
    plt.gcf().set_size_inches(10.5, 1.5)
    plt.xticks(np.arange(-1, 32, 1.0))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()
    
    return coordinate, Ratio_origin

def Phasediff_Magnitude_KmeanClustering(Zxx, Zxx1, spectrogramR, spectrogramL, t, f, c, micdist, k, phasediff_flag=1):
    
    divide=Zxx/Zxx1
    phdiff=np.zeros((len(f), len(t)))
    b=np.zeros((len(f), len(t)))
    argument=np.zeros((len(f), len(t)))

    for i in range(len(f)-1):
        for j in range(len(t)):
            argument[i][j]=cmath.phase(divide[i][j])
            if f[i]==0:
                phdiff[i][j]=(cmath.phase(divide[i][j]) / (2*math.pi*(f[i]+f[i+1])/2))*c/micdist
            else:
                phdiff[i][j]=(cmath.phase(divide[i][j]) / (2*math.pi*(f[i]+f[i+1])/2))*c/micdist
            if phdiff[i][j]>1:
                phdiff[i][j]=1
            elif phdiff[i][j]<-1:
                phdiff[i][j]=-1
            b[i][j]=np.arccos(phdiff[i][j])
            
    Ph_x=[]
    for i in range(9):
        for j in range(len(t)):
            Ph_x.append(np.sin(b[i][j]/2))
            
    Ph_y=[]
    for i in range(9):
        for j in range(len(t)):
            Ph_y.append(np.cos(b[i][j]/2))
    
    Ph_R= np.array(Ph_x)
    Ph_L= np.array(Ph_y)
                  
    C1=np.vstack((Ph_R,Ph_L))
    C1=C1.T
    
    
    R=[]
    for i in range(len(f)):
        for j in range(len(t)):
            R.append(spectrogramR[i,j])
    L=[]
    for i in range(len(f)):
        for j in range(len(t)):
            L.append(spectrogramL[i,j])
            
    R= np.array(R)
    L= np.array(L)
    
    C=np.vstack((R,L))
    C=C.T
    
    sumlength= len(t) * len(f)
    
    for i in range(sumlength):
        s=np.linalg.norm(C[i,])
        C[i,]=C[i,]/s
    if phasediff_flag==1:
        for i in range(99):
            C[i,]=C1[i,]
                
    # Number of clusters
    #k=11
    kmeans = KMeans(n_clusters=k, random_state=8)
    # Fitting the input data
    kmeans = kmeans.fit(C)
    # Getting the cluster labels
    labels = kmeans.predict(C)
    # Centroid values
    centroids = kmeans.cluster_centers_
    
    return labels, L, R, kmeans, centroids, k, C


def Binary_Mask(Zxx, Zxx1, L, R, index, fs, EPSILON):

    
    spectrogramL_Cluster,spectrogramL_phase=librosa.magphase(Zxx)
    spectrogramR_Cluster,spectrogramR_phase=librosa.magphase(Zxx1)
    
           
    D=[]
    for i in index:
        D.append(L[i])
    D= (np.array(D)) 
               
    Real_index=[]
    for i in range(0, len(D)):
        Real_index.append(np.where(spectrogramL_Cluster == D[i]))
        E=(np.array(Real_index[i])).reshape(-1)
        spectrogramL_Cluster[E[0]][E[1]]=EPSILON
        
    D=[]
    for i in index:
        D.append(R[i])
    D= (np.array(D)) 
               
    Real_index=[]
    for i in range(0, len(D)):
        Real_index.append(np.where(spectrogramR_Cluster == D[i]))
        E=(np.array(Real_index[i])).reshape(-1)
        spectrogramR_Cluster[E[0]][E[1]]=EPSILON
    
    
    _, xrec_Cluster = signal.istft(spectrogramL_Cluster*spectrogramL_phase, fs)
    _, xrec1_Cluster = signal.istft(spectrogramR_Cluster*spectrogramR_phase, fs)
    
    return xrec_Cluster, xrec1_Cluster, spectrogramL_Cluster, spectrogramR_Cluster

def SNMF(spectrogramL_NMF, spectrogramR_NMF, spectrogramL_Cluster, spectrogramR_Cluster, spectrogramL_phase, spectrogramR_phase, fs):
    
    model = NMF(n_components=1, init='random',  solver='mu', beta_loss='kullback-leibler',random_state=0)
    W= model.fit_transform(spectrogramL_NMF)
    H = model.components_
    #########################
    
    
    model1 = NMF(n_components=2, init='random',  solver='mu', beta_loss='kullback-leibler',random_state=0)
    W1 = model1.fit_transform(spectrogramL_Cluster,W0=W)
    H1 = model1.components_
    
    
    V1=W1[:,0]
    V1 = np.array(V1)
    V1=V1.reshape((len(V1), -1))
    
    V2=W1[:,1]
    V2=V2.reshape((len(V2), -1))
    
    S1=H1[0,:]
    S1=S1.reshape((-1, len(S1)))
    S2=H1[1,:]
    S2=S2.reshape((-1, len(S2)))
    
    Q1=V1*S1
    Q2=V2*S2
    
    _, xrec_NMF = signal.istft(Q2*spectrogramL_phase, fs)
    ################################################################
    
    W= model.fit_transform(spectrogramR_NMF)
    H = model.components_
    
    W1 = model1.fit_transform(spectrogramR_Cluster,W0=W)
    H1 = model1.components_
    
    
    V1=W1[:,0]
    V1 = np.array(V1)
    V1=V1.reshape((len(V1), -1))
    
    V2=W1[:,1]
    V2=V2.reshape((len(V2), -1))
    
    S1=H1[0,:]
    S1=S1.reshape((-1, len(S1)))
    S2=H1[1,:]
    S2=S2.reshape((-1, len(S2)))
    
    Q1=V1*S1
    Q2=V2*S2
    
    _, xrec1_NMF = signal.istft(Q2*spectrogramR_phase, fs)
    
    return xrec_NMF, xrec1_NMF



def Final_Sound_Localizaion(x, peaks, delayMAX, fs, c, micdist, Deg, Ratio, Deg1, Ratio1):
    if len(peaks) > 0:
        b=np.zeros((len(peaks),))
        for j in range(len(peaks)):
            sort=sorted(x[peaks], reverse=True)
            zzz=[]
            for i in range(len(peaks)):
                zzz.append(np.array(np.where(x == sort[i])))
            zzz=np.array(zzz).reshape(-1)
            
            K=zzz[j]        
            delay=K-delayMAX        
            tau=delay/fs        
            a=tau*c/micdist
            
            if a>1:
                a=1
            elif a<-1:
                a=-1
            
            b[j]=np.arccos(a)*180/math.pi
        Deg.append(b)
        Ratio.append(x[peaks])
        Deg1.append(b[0])
        Ratio1.append(x[zzz[0]])
        
        plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
        plt.gcf().set_size_inches(10.5, 1.5)
        plt.xticks(np.arange(-1, 32, 1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid()
        plt.show()
        print('The Degree of Source is %3.5f' % b[0])
        print('The Ratio of Similarity is %1.5f' % x[zzz[0]])
        
    else:
        K=np.argmax(x)
        delay=K-delayMAX

        tau=delay/fs
        
        a=tau*c/micdist
        

        if a>1:
            a=1
        elif a<-1:
            a=-1
        
        b=np.arccos(a)*180/math.pi
        
        Deg.append(b)
        Ratio.append(x[K])
        Deg1.append(b)
        Ratio1.append(x[K])
        plt.plot(K, x[K], "xr"); plt.plot(x); plt.legend(['distance'])
        plt.gcf().set_size_inches(10.5, 1.5)
        plt.xticks(np.arange(-1, 32, 1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid()
        plt.show()            
        print('The Degree of Source is %3.5f' % b)
        print('The Ratio of Similarity is %1.5f' % x[K])
    
    return Deg, Ratio, Deg1, Ratio1