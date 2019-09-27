# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:50:34 2019

@author: sonmookoh
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
from scipy import signal
import librosa
import scipy.io
import scipy.io.wavfile as wav


from MFCC_Frame_Func import MFCC_FULL, MFCC_exceptFraming, Frame_generation
from Sound_Localization_Func import GCC_Phat, Source_Coordinate_Generation, Phasediff_Magnitude_KmeanClustering, Binary_Mask, SNMF, Final_Sound_Localizaion
from HMM_Func import getTransmatPrior, initByBakis, HMM_Audioread_feature
from Init_Func import Init_Shifting
from Graph_Func import Graph_Waveform_InitSources, Graph_Spectrogram, Graph_Kmeans, Graph_Kmeans_AfterClustering, Graph_Waveform_finalComparison
from Sound_Save_Func import Soundsave_Soundplay


result=[]
gainiter=31
for hj in range(1, gainiter):
    print(hj)
    hj=hj*0.1
    
    
    #def ProposedAlgorithm():
    proposed=1         # 1: Algorithm "ON"
    TargetKnownFlag=1
    
    OnlyNoise_flag=0
    
    RecordNoise_ratio=hj
    ####################
    
    # =============================================================================
    fs = 48000
    c=343.2
    micdist=0.1001
    delayMAX=int(fs*micdist/c)+1
    # =============================================================================
    #Parameters needed to train GMMHMM
    m_num_of_HMMStates = 16  # number of states
    m_num_of_mixtures = 3 # number of mixtures for each hidden state
    m_covarianceType = 'diag'  # covariance type
    m_n_iter = 200  # number of iterations
    m_bakisLevel = 2
    
    
    EPSILON = np.finfo(np.float32).eps
    
# =============================================================================
#     Leftsound=pd.read_csv(r'E:\Sound\Leftsound_30_50cm.csv')
#     Rightsound=pd.read_csv(r'E:\Sound\Rightsound_30_50cm.csv')
# =============================================================================
    Leftsound=pd.read_csv(r'Sound/Lsound_30_50cm_singleshot.csv')
    Rightsound=pd.read_csv(r'Sound/Rsound_30_50cm_singleshot.csv')
# =============================================================================
#     Leftsound=pd.read_csv(r'Sound/Lsound_30_50cm_repeater1.csv')
#     Rightsound=pd.read_csv(r'Sound/Rsound_30_50cm_repeater1.csv')     
# =============================================================================
    X1=np.array(Leftsound['Data:float'])
    X2=np.array(Rightsound['Data:float'])
    X1, X2= Init_Shifting(X1, X2)
    X1_clean=np.array(Leftsound['Data:float'])
    X2_clean=np.array(Rightsound['Data:float'])
    X1_clean, X2_clean= Init_Shifting(X1_clean, X2_clean)
    
# =============================================================================
#     white_Leftsound=pd.read_csv(r'Sound1\Leftsound_150_50cm_babbleLOUD.csv')
#     white_Rightsound=pd.read_csv(r'Sound1\Rightsound_150_50cm_babbleLOUD.csv')
# =============================================================================
    white_Leftsound=pd.read_csv(r'Sound1\Leftsound_150_50cm_white.csv')
    white_Rightsound=pd.read_csv(r'Sound1\Rightsound_150_50cm_white.csv')
# =============================================================================
#     white_Leftsound=pd.read_csv(r'Sound/Lsound_150_50cm_engineroom.csv')
#     white_Rightsound=pd.read_csv(r'Sound/Rsound_150_50cm_engineroom.csv')
# =============================================================================
    white1=np.array(white_Leftsound['Data:float'])
    white2=np.array(white_Rightsound['Data:float'])
    white1, white2= Init_Shifting(white1, white2)
    
    if OnlyNoise_flag==1:
        X1 = 0
        X2 = 0
    white1 = (RecordNoise_ratio*white1)
    white2 = (RecordNoise_ratio*white2)
    X1 += white1
    X2 += white2
    
    
    Graph_Waveform_InitSources(X1_clean, X2_clean, white1, white2, fs)
    Graph_Spectrogram(X1_clean, fs, ylim=20000)
    Graph_Spectrogram(white1, fs, ylim=20000)
        
        
    
    #####################################################################################################
    Final_MFCC1 =MFCC_FULL(X1_clean)
    Final_MFCC2 =MFCC_FULL(X2_clean)
    
    Frameleng=(len(X1_clean)-5)/(10*fs)  #0.05
    #Frameleng=0.025
    Frameleng1=Frameleng
    frames1, num_frames = Frame_generation(X1, frame_size = Frameleng, frame_stride = Frameleng1, empha_signal_flag=0, hamming_flag=0)
    frames2, _ = Frame_generation(X2, frame_size = Frameleng, frame_stride = Frameleng1, empha_signal_flag=0, hamming_flag=0)
    framesSUM, _ =Frame_generation(X2, frame_size = Frameleng, frame_stride = Frameleng1, empha_signal_flag=0, hamming_flag=0)
    framesClean1, _ =Frame_generation(X1_clean, frame_size = Frameleng, frame_stride = Frameleng1, empha_signal_flag=0, hamming_flag=0)   
    framesClean2, _ =Frame_generation(X2_clean, frame_size = Frameleng, frame_stride = Frameleng1, empha_signal_flag=0, hamming_flag=0)
    #####################################################################################################
    kkkk=np.concatenate([Final_MFCC1 , Final_MFCC2])
    lengt = [len(Final_MFCC1), len(Final_MFCC2)]
    features=HMM_Audioread_feature(Final_MFCC1, Final_MFCC2)
    trainData=features
    m_startprobPrior ,m_transmatPrior = initByBakis(m_num_of_HMMStates,m_bakisLevel)
    
    
    model  = hmm.GMMHMM(random_state=8, n_components = m_num_of_HMMStates, n_mix = m_num_of_mixtures, \
                               transmat_prior = m_transmatPrior, startprob_prior = m_startprobPrior, \
                               covariance_type = m_covarianceType, n_iter = m_n_iter, verbose=True)
    length = np.zeros([len(trainData), ], dtype=np.int)
    for m in range(len(trainData)):
        length[m] = trainData[m].shape[0]
    trainData = np.vstack(trainData)
    model.fit(trainData, lengths=length)  # get optimal parameters    
    model._check()
    model.monitor_
    model.monitor_.converged
    
    
    D=[]
    Ratio_origin=[]
    SNR=[]
    source_count=[]
    
    Deg=[]
    Deg1=[]
    Ratio=[]
    Ratio1=[]
    
    TrueDeg=[]
    TrueDeg1=[]
    TrueRatio=[]
    TrueRatio1=[]
    
    for hh in range(num_frames):
    
        print('=====================================START====================================')
        print('The %d Frame.' % hh)
        s1=frames2[hh]
        s2=frames1[hh]
        s3=framesSUM[hh]
        s4=framesClean2[hh]
        s5=framesClean1[hh]
        
        peaks, x = GCC_Phat(s4, s5, threshold=0.2)              
        TrueDeg, TrueRatio, TrueDeg1, TrueRatio1 = Final_Sound_Localizaion(x, peaks, delayMAX, fs, c, micdist, TrueDeg, TrueRatio, TrueDeg1, TrueRatio1)
    
    
    
        Signal1=sum((s4)**2)/len(s1)
        Noise1=sum((s3-s4)**2)/len(s1)
        ILD=(10*np.log10(Signal1/Noise1))
        SNR.append(ILD)
    
    
        N = len(s2)                          
        f, t, Zxx = signal.stft(s2, fs, nperseg=256)  
        f1, t1, Zxx1 = signal.stft(s1, fs, nperseg=256)    
        spectrogramL, spectrogramL_phase = librosa.magphase(Zxx)
        spectrogramR, spectrogramR_phase = librosa.magphase(Zxx1)
    
        peaks, x = GCC_Phat(s1, s2, threshold=0.14)    
        num_direction=len(peaks)
        source_count.append(num_direction)
        
        if num_direction == 0:
            Ratio_origin.append(0)
            plt.plot(x); plt.legend(['distance'])
            plt.gcf().set_size_inches(10.5, 1.5)
            plt.xticks(np.arange(-1, 32, 1.0))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.grid()
            plt.show()  
            print('No Source')
        else:#num_direction > 1
            coordinate, Ratio_origin = Source_Coordinate_Generation(num_direction, x, peaks, delayMAX, fs, c, micdist, Ratio_origin)     
        print('The number of Source is %d.' % num_direction)
    
        if proposed==0:
            num_direction=0
            
        if num_direction > 1:
            labels, L, R, kmeans, centroids, k, C = Phasediff_Magnitude_KmeanClustering(Zxx, Zxx1, spectrogramR, spectrogramL, t, f, c, micdist, k=3, phasediff_flag=1)
            labels1, _, _, kmeans1, centroids1, k1, C1 = Phasediff_Magnitude_KmeanClustering(Zxx, Zxx1, spectrogramR, spectrogramL, t, f, c, micdist, k=19, phasediff_flag=1)
    
            #Graph_Kmeans(centroids, k, labels, C)
            
            FirstS=[]
            FirstS_1=[]
            SecondS=[]
            SecondS_1=[]
            ThirdS=[]
            ThirdS_1=[]
            LastS=[]
            LastS_1=[]
            Firstsource1=[]
            Firstsource=[]          
            Secondsource1=[]
            Secondsource=[]
            Thirdsource1=[]
            Thirdsource=[]
            Lastsource1=[]
            Lastsource=[]
            GG=[]
            GG1=[]
            for jj in range(num_direction):
                Target_source=jj
                Label_target=kmeans.predict(coordinate[[Target_source]])
                index_Cluster=np.array(np.where(labels != Label_target)).T
                if num_direction == 2:                
                    if Target_source==0: 
                        Label_noise=kmeans1.predict(coordinate[[Target_source+1]])
                    else:
                        Label_noise=kmeans1.predict(coordinate[[Target_source-1]])                
                    index_NMF=np.array(np.where(labels1 != Label_noise)).T
                elif num_direction == 3:
                    if Target_source==0: 
                        Label_noise=kmeans1.predict(coordinate[[Target_source+1]])
                        Label_noise1=kmeans1.predict(coordinate[[Target_source+2]])
                    elif Target_source==1: 
                        Label_noise=kmeans1.predict(coordinate[[Target_source-1]])
                        Label_noise1=kmeans1.predict(coordinate[[Target_source+1]])
                    else:
                        Label_noise=kmeans1.predict(coordinate[[Target_source-2]])
                        Label_noise1=kmeans1.predict(coordinate[[Target_source-1]])
                    lllll=np.where(labels1 != Label_noise)
                    lllll1=np.where(labels1 != Label_target)
                    index_NMF=np.array(list(set(lllll[0])&set(lllll1[0])))
                                    
                #Graph_Kmeans_AfterClustering(centroids, Label_target, Label_noise, labels, C)
    # ===================================================================================================================  
    # =============================================================================
    #             print(Label_target)
    #             print(Label_noise)
    # =============================================================================
                xrec_Cluster, xrec1_Cluster, spectrogramL_Cluster, spectrogramR_Cluster=Binary_Mask(Zxx, Zxx1, L, R, index_Cluster, fs, EPSILON)
                _, _, spectrogramL_NMF, spectrogramR_NMF=Binary_Mask(Zxx, Zxx1, L, R, index_NMF, fs, EPSILON)
               
              
                xrec_NMF, xrec1_NMF = SNMF(spectrogramL_NMF, spectrogramR_NMF, spectrogramL_Cluster, spectrogramR_Cluster, spectrogramL_phase, spectrogramR_phase, fs)
    
                Final_MFCC8=MFCC_FULL(xrec1_NMF, frame_size = 0.025, frame_stride = 0.01)
                Final_MFCC9=MFCC_FULL(xrec_NMF, frame_size = 0.025, frame_stride = 0.01)
                if num_direction == 2:
                    if jj==0:                
                        Firstsource1.append(xrec1_NMF)          #Rightest
                        Firstsource.append(xrec_NMF)
                        FirstS.append(Final_MFCC8)
                        FirstS_1.append(Final_MFCC9)
                    else:                 
                        Lastsource1.append(xrec1_NMF)         #left
                        Lastsource.append(xrec_NMF)
                        LastS.append(Final_MFCC8)
                        LastS_1.append(Final_MFCC9)
                elif num_direction == 3:
                    if jj==0:                
                        Firstsource1.append(xrec1_NMF)          #Rightest
                        Firstsource.append(xrec_NMF)
                        FirstS.append(Final_MFCC8)
                        FirstS_1.append(Final_MFCC9)
                    elif jj==1:                
                        Secondsource1.append(xrec1_NMF)         #left
                        Secondsource.append(xrec_NMF)
                        SecondS.append(Final_MFCC8)
                        SecondS_1.append(Final_MFCC9)
                    else:
                        Lastsource1.append(xrec1_NMF)         #left
                        Lastsource.append(xrec_NMF)
                        LastS.append(Final_MFCC8)
                        LastS_1.append(Final_MFCC9)
                else:
                    if jj==0:                
                        Firstsource1.append(xrec1_NMF)          #Rightest
                        Firstsource.append(xrec_NMF)
                        FirstS.append(Final_MFCC8)
                        FirstS_1.append(Final_MFCC9)
                    elif jj==1:                
                        Secondsource1.append(xrec1_NMF)         #left
                        Secondsource.append(xrec_NMF)
                        SecondS.append(Final_MFCC8)
                        SecondS_1.append(Final_MFCC9)
                    elif jj==2:
                        Thirdsource1.append(xrec1_NMF)
                        Thirdsource.append(xrec_NMF)
                        ThirdS.append(Final_MFCC8)
                        ThirdS_1.append(Final_MFCC9)
                    else:
                        Lastsource1.append(xrec1_NMF)         #left
                        Lastsource.append(xrec_NMF)
                        LastS.append(Final_MFCC8)
                        LastS_1.append(Final_MFCC9)
                    
            
            
            
            if num_direction == 2:
                GG.append(model.score(FirstS[0]))
                GG.append(model.score(LastS[0]))
                GG1.append(model.score(FirstS_1[0]))
                GG1.append(model.score(LastS_1[0]))
                
                k=np.argmax(GG)
                k1=np.argmax(GG1)
                
    # =============================================================================
    #             pp=model.score(FirstS[0])
    #             pp1=model.score(LastS[0])
    #             pp_1=model.score(FirstS_1[0])
    #             pp1_1=model.score(LastS_1[0])
    # =============================================================================
                
                if TargetKnownFlag==0:
                    #if pp<=pp1 or pp_1<=pp1_1:
                    if k==1 or k1==1:
                        s1=Lastsource1[0]    #s1:rightchannel sound
                        s2=Lastsource[0]
                    else:
                        s1=Firstsource1[0]    #s1:rightchannel sound
                        s2=Firstsource[0]
                else:
                    s1=Lastsource1[0]    #s1:rightchannel sound
                    s2=Lastsource[0]
                    
            elif num_direction == 3:
                GG.append(model.score(FirstS[0]))
                GG.append(model.score(SecondS[0]))
                GG.append(model.score(LastS[0]))
                GG1.append(model.score(FirstS_1[0]))
                GG1.append(model.score(SecondS_1[0]))
                GG1.append(model.score(LastS_1[0]))
                
                k=np.argmax(GG)
                k1=np.argmax(GG1)
                if TargetKnownFlag==0:
                    #if pp<=pp1 or pp_1<=pp1_1:
                    if k==2 or k1==2:
                        s1=Lastsource1[0]    #s1:rightchannel sound
                        s2=Lastsource[0]
                    else:
                        s1=Firstsource1[0]    #s1:rightchannel sound
                        s2=Firstsource[0]
                else:
                    s1=Lastsource1[0]    #s1:rightchannel sound
                    s2=Lastsource[0]
            else:
                GG.append(model.score(FirstS[0]))
                GG.append(model.score(SecondS[0]))
                GG.append(model.score(ThirdS[0]))
                GG.append(model.score(LastS[0]))
                GG1.append(model.score(FirstS_1[0]))
                GG1.append(model.score(SecondS_1[0]))
                GG1.append(model.score(ThirdS_1[0]))
                GG1.append(model.score(LastS_1[0]))
                
                k=np.argmax(GG)
                k1=np.argmax(GG1)
                if TargetKnownFlag==0:
                    #if pp<=pp1 or pp_1<=pp1_1:
                    if k==3 or k1==3:
                        s1=Lastsource1[0]    #s1:rightchannel sound
                        s2=Lastsource[0]
                    else:
                        s1=Firstsource1[0]    #s1:rightchannel sound
                        s2=Firstsource[0]
                else:
                    s1=Lastsource1[0]    #s1:rightchannel sound
                    s2=Lastsource[0]
           
            peaks, x = GCC_Phat(s1, s2, threshold=0.2)              
            Deg, Ratio, Deg1, Ratio1 = Final_Sound_Localizaion(x, peaks, delayMAX, fs, c, micdist, Deg, Ratio, Deg1, Ratio1)
                        
        else:
            s1=frames2[hh]
            s2=frames1[hh]
            xrec1_NMF=s1
            xrec_NMF=s2
            peaks, x = GCC_Phat(s1, s2, threshold=0.2)       
            
            Deg, Ratio, Deg1, Ratio1 = Final_Sound_Localizaion(x, peaks, delayMAX, fs, c, micdist, Deg, Ratio, Deg1, Ratio1)
    # =============================================================================
    #         Firstsource1.append(xrec1_NMF)
    #         Firstsource.append(xrec_NMF)
    #         Secondsource1.append(xrec1_NMF)         #left
    #         Secondsource.append(xrec_NMF)
    #         Thirdsource1.append(xrec1_NMF)
    #         Thirdsource.append(xrec_NMF)
    #         Lastsource1.append(xrec1_NMF)
    #         Lastsource.append(xrec_NMF)
    # =============================================================================
            
            
        #Graph_Waveform_finalComparison(s1, s3, s2, s4, xrec1_NMF, xrec_NMF, fs)
        #Soundsave_Soundplay(xrec_NMF, xrec1_NMF, s1, s2, fs)
      
        print('=====================================END====================================')
    #########################################################################################################
            
    
    plt.plot(SNR)
    plt.gcf().set_size_inches(10.5, 10.5)
    # =============================================================================
    # plt.savefig('test2png.png', dpi=100)
    # =============================================================================
    plt.ylim(min(SNR)-5, max(SNR)+5)
    plt.xticks(np.arange(0, num_frames, 1.0))
    plt.yticks(np.arange(min(SNR)-5, max(SNR)+5, 1.0))
    plt.legend(['SNR'])
    plt.title('SNR per Each Frame')
    plt.ylabel('dB')
    plt.xlabel('Frame')
    plt.grid()
    plt.show()
    mean_SNR=np.mean(SNR)
    print('The Mean of SNR is %3.5f' % mean_SNR)
    
    
    
    frame = np.arange(num_frames)
    plt.plot(Deg1);plt.scatter(frame, Deg1, marker='*', c=Ratio1,s=100);
    plt.gcf().set_size_inches(10.5, 10.5)
    # =============================================================================
    # plt.savefig('test2png.png', dpi=100)
    # =============================================================================
    # =============================================================================
    # plt.ylim(-20,200)
    # =============================================================================
    plt.xticks(np.arange(0, num_frames, 1.0))
    plt.yticks(np.arange(0, 200, 30.0))
    plt.legend(['Degree','Ratio'])
    plt.title('Direction of Target Source per Each Frame')
    plt.ylabel('Degree')
    plt.xlabel('Frame')
    plt.grid()
    plt.show()
    
    
    plt.plot(frame, Ratio_origin, frame, Ratio1)
    plt.gcf().set_size_inches(10.5, 10.5)
    # =============================================================================
    # plt.savefig('test2png.png', dpi=100)
    # =============================================================================
    # =============================================================================
    # plt.ylim(0, 1.0)
    # =============================================================================
    plt.xticks(np.arange(0, num_frames, 1.0))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(['Ratio_origin','Ratio'])
    plt.title('Similarity of Each Channel per Each Frame')
    plt.ylabel('Ratio')
    plt.xlabel('Frame')
    plt.grid()
    plt.show()
    
    plt.plot(frame, source_count)
    plt.gcf().set_size_inches(10, 3)
    # =============================================================================
    # plt.savefig('test2png.png', dpi=100)
    # =============================================================================
    # =============================================================================
    # plt.ylim(0, 1.0)
    # =============================================================================
    plt.xticks(np.arange(0, num_frames, 1.0))
    plt.yticks(np.arange(0, 4, 1))
    plt.legend(['Ratio_origin','Ratio'])
    plt.title('Similarity of Each Channel per Each Frame')
    plt.ylabel('Ratio')
    plt.xlabel('Frame')
    plt.grid()
    plt.show()
    
    wow_index=np.array(np.where(np.array(source_count)>=2)).T
    wow_index=(np.where(np.array(source_count)>=2))[0]
    
    #result=np.concatenate([SNR , Deg1])
    
    error=[]
    SNRfinal=[]
    for kk in wow_index:
        if Deg1[kk]==TrueDeg1[kk]:
            error.append(0)
        else:
            error.append(1)
        SNRfinal.append(SNR[kk])
    
    Z=np.array(SNRfinal)    
    Z1=np.array(error)
    Z=Z.reshape((len(SNRfinal),1))
    Z1=Z1.reshape((len(error),1))
    result.append(np.concatenate((Z, Z1), axis=1))
rresult1=result[0]
for kj in range(gainiter-2):
    
    rresult1=np.concatenate([rresult1 , result[kj+1]])
    
rrresult=np.load('notproposed_repeater1_bubble2.npy')
rresult=np.load('proposed_repeater1_bubble2.npy')
rrresult_sort=sorted(rrresult[:,0], reverse=False)
zzz=[]
for i in range(len(rrresult_sort)):
    zzz.append(np.array(np.where(rrresult[:,0] == rrresult_sort[i])))
zzz=np.array(zzz).reshape(-1)
yyy=[]
for i in range(len(rrresult_sort)):
    yyy.append(rrresult[zzz[i]][1])
yyy=np.array(yyy).reshape((len(yyy),1))
# =============================================================================
# xxx=[]
# for i in range(len(rrresult_sort)):
#     xxx.append(rrresult[zzz[i]][0])
# xxx=np.array(xxx).reshape((len(xxx),1))
# =============================================================================
xxx=[]
for i in range(len(rrresult_sort)):
    xxx.append(round(rrresult[zzz[i]][0]))
xxx=np.array(xxx).reshape((len(xxx),1))



yyy1=[]
for i in range(len(rrresult_sort)):
    yyy1.append(rresult[zzz[i]][1])
yyy1=np.array(yyy1).reshape((len(yyy1),1))


plt.plot(xxx,yyy)
plt.gcf().set_size_inches(10.5, 10.5)
plt.grid()
plt.show()
plt.plot(xxx,yyy1)
plt.grid()
plt.show()

plt.plot(xxx,yyy,xxx,yyy1)
plt.gcf().set_size_inches(10.5, 10.5)
plt.xlim(-7, 4)
plt.xticks(np.arange(-7, 4, 1.0))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Ratio_origin','Ratio'])
plt.title('Similarity of Each Channel per Each Frame')
plt.ylabel('Ratio')
plt.xlabel('Frame')
plt.grid()
plt.show()

finalresult=np.concatenate((xxx, yyy), axis=1)
finalresult1=np.concatenate((xxx, yyy1), axis=1) #proposed

