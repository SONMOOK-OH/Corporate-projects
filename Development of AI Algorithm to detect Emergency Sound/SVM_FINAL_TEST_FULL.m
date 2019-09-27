%% Feature Extraction

% Clean-up MATLAB's environment
clear; close all; clc;
%global Sound;
Sound=1;
for Sound=1:1:6
    clearvars -except Sound
    if  Sound > 1
         pause(3);
    end  

   
    load('SVM_MODEL.mat')

    % Define variables v
    Tw = 25;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 13;                 % number of filterbank channels 
    C = 13;                 % number of cepstral coefficients
    L = 20;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 8000;              % upper frequency limit (Hz)

    MfccNum = 13;
    FrameNum = 37;
    Shift = 10;
    start=1;
    k=1;
    global pitc;

    pitc=0.1;



    if  Sound==1
        wav_file_raw(1,:) = 'Guntest.wav';
        pitc=0.001;
    elseif Sound == 2
        wav_file_raw(1,:) = 'Glassbreakingtest.wav';
        pitc=0.1;
    elseif Sound == 3
        wav_file_raw(1,:) = 'SCREAMtest.wav';
        pitc=0.1;
    elseif Sound == 4
        wav_file_raw(1,:) = 'Guntest2.wav';
        pitc=0.1;
    elseif Sound == 5
        wav_file_raw(1,:) = 'Glassbreakingtest2.wav';
        pitc=0.1;
    else
        wav_file_raw(1,:) = 'MyScream4.wav';
        pitc=0.1;

    end  

    [ speech , Fs ] = audioread( wav_file_raw(1,:) );
    l = numel(speech)/2;
    for k=1:1:l    

        if abs(speech(k,1)) > pitc
           start=k; 
           k=l;
           break
        end 


    end
    samples = [start,(0.4*Fs)+start-1];
    %  samples = [29001,(0.4*Fs)+29000];
    clear speech Fs
    [speech,Fs] = audioread( wav_file_raw(1,:),samples);
    sound(speech , Fs)
    filename = '1.wav';
    audiowrite(filename,speech,Fs);

    wav_file(1,:) = '1.wav';
    % wav_file(1,:) = 'N:\AI_MATLAB\Glassbreakingtest.wav';

    i=1;
        [ speech, fs ] = audioread(wav_file(1,:));
        [ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
        scream(i,:,:) = MFCCs(2:14,:)';


% Delta
    for i=2:FrameNum-1
        for j=1:MfccNum
            scream(:,i,j+MfccNum) = (scream(:,i+1,j) - scream(:,i-1,j))/2;
        end
    end

    % Delta-Delta
    for i=2:FrameNum-1
        for j=1:MfccNum
            scream(:,i,j+MfccNum*2) = (scream(:,i+1,j+MfccNum) - scream(:,i-1,j+MfccNum))/2;
        end
    end


    ExampleNum = 30;
    MfccNum = 39;
    FrameNum = 37;
    i=1;
    for j=1:MfccNum      
        scream_meanvar1(i,j) = mean(scream(i,:,j));
        scream_meanvar1(i,j+MfccNum) = var(scream(i,:,j));    
    end
    
    load('Best_feature.mat')
    X_test_w_best_features = scream_meanvar1(:,fs);

    K=predict(Md1,X_test_w_best_features);

    % Plot
    fs=44100;
    % Generate data needed for plotting 
    [ Nw, NF ] = size( frames );                % frame length and number of frames
    time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames 
    time = [ 0:length(speech)-1 ]/fs;           % time vector (s) for signal samples 
    logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
    logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
    logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range


    if  Sound==1
        figure('Position', [0 400 640 600], 'PaperPositionMode', 'auto', ... 
          'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 


    elseif Sound == 2
        figure('Position', [640 400 640 600], 'PaperPositionMode', 'auto', ... 
              'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 

    elseif Sound == 3
        figure('Position', [1280 400 640 600], 'PaperPositionMode', 'auto', ... 
          'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 


    elseif Sound == 4
        figure('Position', [0 400 640 600], 'PaperPositionMode', 'auto', ... 
              'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 


    elseif Sound == 5
        figure('Position', [640 400 640 600], 'PaperPositionMode', 'auto', ... 
              'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );
        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 

    else
        figure('Position', [1280 400 640 600], 'PaperPositionMode', 'auto', ... 
          'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

        subplot( 311 );
        plot( time, speech, 'k' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Amplitude' ); 
        title( 'Speech waveform'); 

        subplot( 312 );
        imagesc( time_frames, [1:M], logFBEs ); 
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Channel index' ); 
        title( 'Log (mel) filterbank energies'); 

        subplot( 313 );
        imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
        imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
        axis( 'xy' );
        xlim( [ min(time_frames) max(time_frames) ] );
        xlabel( 'Time (s)' ); 
        ylabel( 'Cepstrum index' );
        title( 'Mel frequency cepstrum' );

        % Set color map to grayscale
        colormap( 1-colormap('gray') ); 



    end  


% Generate plots


% % Print figure to pdf and png files
% print('-dpdf', sprintf('%s.pdf', mfilename)); 
% print('-dpng', sprintf('%s.png', mfilename)); 




    if K == 1
        disp('Glass Breaking')
        [ speech , Fs ] = audioread( 'OUTPUT_Glass.mp3');
        sound(speech , Fs)
                  
    elseif K == 2
        disp('Gunshot')
        [ speech , Fs ] = audioread( 'OUTPUT_GUN.mp3' );
        sound(speech , Fs)
        
    else
        disp('Scream')
        [ speech , Fs ] = audioread('OUTPUT_SCREAM.mp3' );
        sound(speech , Fs)
    end    
      
% for i=1:39
%     data(i) = scream(1,5,i);
% end
% t=1:39;
% plot(t,data);
end







