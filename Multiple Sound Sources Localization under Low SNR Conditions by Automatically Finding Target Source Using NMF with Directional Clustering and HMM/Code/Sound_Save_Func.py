# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:52:18 2019

@author: sonmookoh
"""
import os
import winsound
import scipy.io.wavfile as wav


def Soundsave_Soundplay(xrec, xrec1, s1, s2, fs):
    wav.write('11.wav', fs, (xrec))
    wav.write('12.wav', fs, (xrec1))
    wav.write('output_original.wav', fs, s1+s2)
    
    
    os.system("output_original.wav")
    winsound.PlaySound('soundL.wav', winsound.SND_FILENAME)