import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
import pyedflib 
import numpy as np
import datetime
import pandas as pd
import scipy 
import scipy.signal
import glob
import multiprocessing as mp
from pandas import read_csv
from numpy.linalg import lstsq
from sklearn import preprocessing 
from scipy.signal import butter, lfilter

def readEDFfile(file_name,epoch):

    if os.path.isfile(file_name + ".txt"):
        spike_annotations = read_csv(file_name + ".txt", sep='\t')
        spike_annotations = spike_annotations[~spike_annotations['Annotation'].str.contains("Recording")].values # Removes 'Started Recording' and 'Stopped Recording' rows
        i = 0
        for row in spike_annotations[:,1]:
            spike_annotations[i,1] = row[0:6] + "20" + row[6:-4]
            i += 1
    else:
        spike_annotations = []

    f = pyedflib.EdfReader(file_name + ".edf")

    eeg1 = f.readSignal(0)
    eeg2 = f.readSignal(1)
    emg = f.readSignal(2)
    fd = f.getFileDuration()
    f._close()

    start_date = datetime.datetime(f.getStartdatetime().year,f.getStartdatetime().month,f.getStartdatetime().day,
                                   f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second)
    file_end = start_date + datetime.timedelta(seconds = fd)

    step = datetime.timedelta(seconds=epoch)  

    time_stamps = []
    curr_spike = 0
    labels_EEG1 = []
    spikes_EEG1 = []
    labels_EEG2 = []
    spikes_EEG2 = []

    while start_date < file_end:

        if os.path.isfile(file_name + ".txt") and (start_date <= datetime.datetime.strptime(spike_annotations[curr_spike][1], "%m/%d/%Y %H:%M:%S") <= start_date + step):

            if "EEG1" in str(spike_annotations[curr_spike][4]):
                if "polyspike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Polyspike")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")
                elif "spike-wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Spike-Wave")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")
                elif "spike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Spike")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")
                elif "wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Wave")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")
                else:
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")

                if curr_spike < len(spike_annotations) - 1: curr_spike += 1

            elif "EEG2" in str(spike_annotations[curr_spike][4]):
                if "polyspike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Polyspike")
                elif "spike-wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Spike-Wave")
                elif "spike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Spike")
                elif "wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Wave")
                else:
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")

                if curr_spike < len(spike_annotations) - 1: curr_spike += 1                           

            elif "ALL" in str(spike_annotations[curr_spike][4]):
                if "polyspike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Polyspike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Polyspike")
                elif "spike-wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Spike-Wave")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Spike-Wave")
                elif "spike" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Spike")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Spike")
                elif "wave" in str(spike_annotations[curr_spike][5]).lower():
                    labels_EEG1.append("Label")
                    spikes_EEG1.append("Wave")
                    labels_EEG2.append("Label")
                    spikes_EEG2.append("Wave")
                else:
                    labels_EEG1.append("No label")
                    spikes_EEG1.append("Non-spike")
                    labels_EEG2.append("No label")
                    spikes_EEG2.append("Non-spike")

                if curr_spike < len(spike_annotations) - 1: curr_spike += 1         

            else:
                labels_EEG1.append("No label")
                spikes_EEG1.append("Non-spike")
                labels_EEG2.append("No label")
                spikes_EEG2.append("Non-spike")

                if curr_spike < len(spike_annotations) - 1: curr_spike += 1       


            while datetime.datetime.strptime(spike_annotations[curr_spike][1], "%m/%d/%Y %H:%M:%S") <= start_date + step:
                if curr_spike < len(spike_annotations) - 1: curr_spike += 1 
                else: break

        else:
            labels_EEG1.append("No label")
            spikes_EEG1.append("Non-spike")
            labels_EEG2.append("No label")
            spikes_EEG2.append("Non-spike")

        time_stamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))

        start_date += step

    return eeg1,eeg2,emg,time_stamps,labels_EEG1,spikes_EEG1,labels_EEG2,spikes_EEG2

def calculate_psd_and_f(signal,fs,epoch):
    epoch = epoch*fs
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    fr,p = scipy.signal.welch(new_signal,fs=fs,nperseg=epoch,scaling='spectrum')
    return fr,p

#Butter filter for obtain the signal only in a range of freqs. Used for energy and amplitude of each band
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a    

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Compute mean curve length, energy, and Teager energy (according to Gardner et al., 2006)
def CL_func(signal):
    CL = []
    
    for i in range(0,len(signal)):
        
        CL.append(abs(signal[i]-signal[i-1]))
    
    output = np.log(sum(CL)/len(signal))
    return output

def E_func(signal):
    E = []
    
    for i in range(0,len(signal)):
        E.append(signal[i]**2)
    
    output = np.log(sum(E)/len(signal))
    return output

def TE_func(signal):
    TE = []
    
    for i in range(2,len(signal)):
        TE.append((signal[i-1]**2) - (signal[i]*signal[i-2]))
    
    output = np.log(sum(TE)/len(signal))
    return output

def calculate_novelty(signal,epoch, var):
    curve_length = []
    mean_energy = []
    teager_energy = []
    
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    
    for i in range(0, len(new_signal)):
        lengthCurve = CL_func(new_signal[i,:])
        mean_E = E_func(new_signal[i,:])
        mean_TE = TE_func(new_signal[i,:])
        
        curve_length.append(lengthCurve)
        mean_energy.append(mean_E)
        teager_energy.append(mean_TE)
    
    return curve_length, mean_energy, teager_energy

# Compute Higuchi Fractal Dimension of a time series (signal); Kmax is an HFD parameter
def hfd_func(signal, Kmax):
    L = []
    output = []
    N = len(signal)
    
    for k in range(1,Kmax):
        Lk = []
        
        for m in range(0,k):
            Lmk = 0
            for i in range(1,int(np.floor((N - m)/k))):
                Lmk += abs(signal[m + i*k] - signal[m + i*k - k])
            Lmk = Lmk*(N - 1)/np.floor((N - m)/float(k))/k
            Lk.append(Lmk)
        
        L.append(np.log(np.mean(Lk)))
        output.append([np.log(float(1)/k), 1])
    
    (p, r1, r2, s)=lstsq(output, L,rcond=None)
    
    return p[0]

def calculate_hfd(signal,epoch,kmax):
    prob = []
    
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    
    for i in range(0, len(new_signal)):
        p = hfd_func(new_signal[i,:], kmax)
        prob.append(p)
    
    return prob

# Calculates phase coherence between two eeg channels
def phase_func(sig1,sig2,epoch,fs):
    
    max_fr = 100
    
    fr, cxy = scipy.signal.coherence(sig1, sig2, fs, nperseg=epoch/5)
    cxy = cxy[(fr>=5) & (fr <=max_fr)]
    meanCoherence = np.mean(cxy)
    sdCoherence = np.std(cxy)
    
    return meanCoherence, sdCoherence

def calculate_PC(sig1,sig2,epoch,fs):
    
    coherenceMean = []
    coherenceVar = []
    
    epoch = epoch*fs
    
    corr1 = sig1[:len(sig1)-(len(sig1)%epoch)]
    corr2 = sig2[:len(sig2)-(len(sig2)%epoch)]
    
    new1 = np.reshape(corr1,(len(corr1)//epoch,epoch))
    new2 = np.reshape(corr2,(len(corr2)//epoch,epoch))
    
    for i in range(0, len(new1)):
        meanC, sdC = phase_func(new1[i,:],new2[i,:], epoch, fs)
        coherenceMean.append(meanC)
        coherenceVar.append(sdC)
        
    return coherenceMean, coherenceVar

# This does all the heavy lifting

def extract_features(signal,signal_label,epoch,fs,is_emg=False):
    
    fr,p = calculate_psd_and_f(signal,fs,epoch)
    epoch = epoch*fs
    
    #Data from EEG
    
    if (is_emg == False):
        max_fr = 100
        
        ## Calculate the total power, total power per epoch, and extract the relevant frequencies.
        ## IMPORTANT NOTE: These are not the ACTUAL power values, they are standardized to account
        ## for individual variability, and are thus relative.
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T
           
        ## Calculate the relative power at the different brain waves:
        delta = np.sum(std_power[:,(freq>=0.5) & (freq <=4)],axis=1)
        
        thetacon = np.sum(std_power[:,(freq>=4) & (freq <=12)],axis=1)
        theta1 = np.sum(std_power[:,(freq>=6) & (freq <=9)],axis=1)
        theta2 = np.sum(std_power[:,(freq>=5.5) & (freq <=8.5)],axis=1)
        theta3 = np.sum(std_power[:,(freq>=7) & (freq <=10)],axis=1)
        
        beta = np.sum(std_power[:,(freq>=20) & (freq <=40)],axis=1)
                
        alpha = np.sum(std_power[:,(freq>=8) & (freq <=13)],axis=1)
        sigma = np.sum(std_power[:,(freq>=11) & (freq <=15)],axis=1)
        spindle = np.sum(std_power[:,(freq>=12) & (freq <=14)],axis=1)
        gamma= np.sum(std_power[:,(freq>=35) & (freq <=45)],axis=1)
        
        temp1= np.sum(std_power[:,(freq>=0.5) & (freq <=20)],axis=1)
        temp2= np.sum(std_power[:,(freq>=0.5) & (freq <=50)],axis=1)
        
        temp3= np.sum(std_power[:,(freq>=0.5) & (freq <=40)],axis=1)
        temp4= np.sum(std_power[:,(freq>=11) & (freq <=16)],axis=1)
        
        freq4565 = np.sum(std_power[:,(freq>=45) & (freq <=65)],axis=1)
        freq520 = np.sum(std_power[:,(freq>=5) & (freq <=20)],axis=1)
        freq85100 = np.sum(std_power[:,(freq>=85) & (freq <=100)],axis=1) 
        
        EEGrel1 = thetacon/delta;
        EEGrel2 = temp1/temp2;
        EEGrel3 = temp4/temp3;
        
        hann = np.hanning(12);
        
        spindelhan1=np.convolve(hann,EEGrel3,'same');
        
        spindelhan=np.transpose(spindelhan1);
        
        ## Calculate Higuchi Fractal Dimension
        HFD = calculate_hfd(signal,epoch,8)
        meanCL, meanE, meanTE = calculate_novelty(signal, epoch, np.var(signal))
        
        ## Calculate the 90% spectral edge:
        spectral90 = 0.9*(np.sum(sum_power,axis=1))
        s_edge = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_edge[row_ind,:]) if j>=spectral90[row_ind]][0] for row_ind in range(s_edge.shape[0])]
        spectral_edge = np.take(fr,l) # spectral edge 90%, the frequency below which power sums to 90% of the total power
        
         ## Calculate the 50% spectral mean:
        spectral50 = 0.5*(np.sum(sum_power,axis=1))
        s_mean = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_mean[row_ind,:]) if j>=spectral50[row_ind]][0] for row_ind in range(s_mean.shape[0])]
        spectral_mean50 = np.take(fr,l) 
                
    else:
        #for EMG
        max_fr = 100
        
        ## Calculate the total power, total power per epoch, and extract the relevant frequencies: 
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T
    
    ## Calculate the Root Mean Square of the signal
    signal = signal[0:p.shape[0]*epoch]
    s = np.reshape(signal,(p.shape[0],epoch))
    rms = np.sqrt(np.mean((s)**2,axis=1)) #root mean square
    ## Calculate amplitude and spectral variation:
    amplitude = np.mean(np.abs(s),axis=1)
    amplitude_m=np.median(np.abs(s),axis=1)
    signal_var = (np.sum((np.abs(s).T - np.mean(np.abs(s),axis=1)).T**2,axis=1)/(len(s[0,:])-1)) # The variation
    ## Calculate skewness and kurtosis
    m3 = np.mean((s-np.mean(s))**3,axis=1) #3rd moment
    m2 = np.mean((s-np.mean(s))**2,axis=1) #2nd moment
    m4 = np.mean((s-np.mean(s))**4,axis=1) #4th moment
    skew = m3/(m2**(3/2)) # skewness of the signal, which is a measure of symmetry
    kurt = m4/(m2**2) #kurtosis of the signal, which is a measure of tail magnitude
    
    ## Calculate more time features
    
    signalzero=preprocessing.maxabs_scale(s,axis=1)
    zerocross = (np.diff(np.sign(signalzero)) != 0).sum(axis=1)
        
    maxs = np.amax(s,axis=1)
    mins = np.amin(s,axis=1)
    
    peaktopeak= maxs - mins
    
    maxIndex = np.zeros(len(maxs))
    minIndex = np.zeros(len(mins))

    for i in range(0 , len(maxs)):
        currRow = list(s[i,:])
        maxIndex[i] = currRow.index(maxs[i])
        minIndex[i] = currRow.index(mins[i])

    p2pVAR = peaktopeak/signal_var
    p2pSD = peaktopeak/np.sqrt(signal_var)
    
    arv1 = ((np.abs(s)))

    arv = np.sum(arv1,axis=1)

    arv = arv / len(s)
                 
    #Energy and amplitude           
    freq4565comp = butter_bandpass_filter(s, 45, 65, fs, 5)
    freq4565energy = sum([x*2 for x in np.matrix.transpose(freq4565comp)])
    freq4565amp = np.mean(np.abs(freq4565comp),axis=1)
    
    freq520comp = butter_bandpass_filter(s, 5, 20, fs, 5)
    freq520energy = sum([x*2 for x in np.matrix.transpose(freq520comp)])
    freq520amp = np.mean(np.abs(freq520comp),axis=1)
    
    freq85100comp = butter_bandpass_filter(s, 85, 100, fs, 5)
    freq85100energy = sum([x*2 for x in np.matrix.transpose(freq85100comp)])
    freq85100amp = np.mean(np.abs(freq85100comp),axis=1)
       
    ## Calculate the spectral mean and the spectral entropy (essentially the spectral power distribution):
    spectral_mean = np.mean(std_power,axis=1)
    spectral_entropy = -(np.sum((std_power+0.01)*np.log(std_power+0.01),axis=1))/(np.log(len(std_power[0,:])))
        
    ## Create a matrix of all of the features per each epoch of the signal
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    epochs = np.arange(len(corr_signal)/epoch)+1
    
    if (is_emg == False):
        feature_matrix = np.column_stack((epochs,delta, thetacon, theta1, theta2, theta3, beta, alpha, sigma,
                                          spindle, gamma, freq4565, freq4565energy, freq4565amp, freq520, freq520energy, freq520amp,
                                          freq85100, freq85100energy, freq85100amp, HFD, meanCL, meanE, meanTE, EEGrel1, EEGrel2, 
                                          spindelhan, spectral_edge, spectral_mean50, zerocross, maxs, peaktopeak, p2pVAR, p2pSD,
                                          arv, rms, amplitude, amplitude_m, signal_var, skew, kurt, spectral_mean, spectral_entropy))
                 
        features = (['epochs','delta','thetacon','theta1','theta2', 'theta3','beta','alpha', 'sigma','spindle','gamma',  
                     'freq4565', 'freq4565energy', 'freq4565amp', 'freq520', 'freq520energy', 'freq520amp',
                     'freq85100', 'freq85100energy', 'freq85100amp', 'HFD', 'meanCL', 'meanE', 'meanTE', 'EEGrel1', 'EEGrel2', 
                     'spindelhan', 'spectral_edge', 'spectral_mean50', 'zerocross', 'maxs' , 'peaktopeak', 'p2pVAR', 'p2pSD',
                     'arv', 'rms', 'amplitude', 'amplitude_m', 'signal_var', 'skew', 'kurt', 'spectral_mean', 'spectral_entropy'])
        
    else:
        feature_matrix = np.column_stack((epochs,amplitude,signal_var,skew,kurt,rms,
                                     spectral_mean,spectral_entropy,amplitude_m))
        
        features = (['epochs','amplitude','signal_var','skew',
                          'kurt','rms','spectral_mean','spectral_entropy','amplitude_m'])

    feature_labels = []
    
    for i in range(len(features)):
        feature_labels.append('%s_%s' % (signal_label,features[i]))
    return feature_matrix,feature_labels,freq520comp,freq85100comp

# Calls extract_features on a signal to create a final matrix with all features
def CreateFeaturesDataFrame(eeg1,eeg2,emg,epoch,fs):  
    
    print("EEG 1...")
    eeg1_features,eeg1_feature_labels,eeg1_520,eeg1_85100 = extract_features(eeg1,'EEG1',epoch,fs)
    
    print("EEG 2...")
    eeg2_features,eeg2_feature_labels,eeg2_520,eeg2_85100 = extract_features(eeg2,'EEG2',epoch,fs)
    
    print("EMG...")
    emg_features,emg_feature_labels,emg_520,emg_85100 = extract_features(emg,'EMG',epoch,fs,is_emg=True)
    
    ### Calculate Power Spectral Coherence Ratio (PSCR)
    PSC1 = (eeg1_520*np.conjugate(eeg2_520))/len(eeg1_520)**2
    PSC2 = (eeg1_85100*np.conjugate(eeg2_85100))/len(eeg1_85100)**2
    PSC1std = np.std(np.abs(PSC1),axis=1)
    PSC2std = np.std(np.abs(PSC2),axis=1)
    PSCR = PSC1std/PSC2std
    
    ### Calculate Phase Coherence (PC)
    PCmean, PCvar = calculate_PC(eeg1,eeg2,epoch,fs)
    
    ### Creates feature matrix
    feature_matrix = np.column_stack((eeg1_features,eeg2_features,PSCR,PCmean,PCvar,emg_features[:,1:]))
    feature_labels = ['Epoch1'] + eeg1_feature_labels[1:] + ['Epoch2'] + eeg2_feature_labels[1:] + ['PSCR'] + ['PCmean'] + ['PCvar'] + emg_feature_labels[1:]
    data = pd.DataFrame(feature_matrix,columns=feature_labels)
    final_data = pd.DataFrame(data.iloc[:,1:])
    return final_data

def exportCSV(nameFS):
    name = nameFS[0]
    fs = nameFS[1]
    
    print(name)
    if name.endswith(".edf"):
        name = name[:-4]
    
    epoch = 10 # In seconds
    eeg1,eeg2,emg,time_stamps,labels_EEG1,spikes_EEG1,labels_EEG2,spikes_EEG2 = readEDFfile(name,epoch)
    data_frame = CreateFeaturesDataFrame(eeg1,eeg2,emg,epoch,fs)
    
    if len(data_frame) != len(time_stamps):
        time_stamps = time_stamps[:-1]
        labels_EEG1 = labels_EEG1[:-1]
        spikes_EEG1 = spikes_EEG1[:-1]
        labels_EEG2 = labels_EEG2[:-1]
        spikes_EEG2 = spikes_EEG2[:-1]

    ### Write features to .csv file
    data_frame['time_stamps'] = time_stamps
    data_frame['EEG1_label_present'] = labels_EEG1
    data_frame['EEG1_label'] = spikes_EEG1
    data_frame['EEG2_label_present'] = labels_EEG2
    data_frame['EEG2_label'] = spikes_EEG2

    data_frame.to_csv(name + "_export.csv")

os.chdir('G:/My Drive/Alzheimers Spike Analyses/toExtract')
        
files = glob.glob('*.edf')
files = list(zip(files,[400]*len(files)))
        
def main():
    pool = mp.Pool(2)
    pool.map(exportCSV, [nameFS for nameFS in files])
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    main()