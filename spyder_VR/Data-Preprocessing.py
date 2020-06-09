# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:22:33 2020

@author: Lei Xian
"""

import os
import shutil
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
import librosa
from pydub import AudioSegment
from scipy.io import wavfile
import plot_funcs
import auxiliary_funcs
import FolderMap

folderMap = FolderMap.FolderMap()

raw_data = folderMap.RAW_DATA 

def fetch_data(folder_path=raw_data): 
    audios = []
    for audio in os.listdir(folder_path):
        if audio.endswith('.wav'):
            audios.append(os.path.join(folder_path, audio))
    return audios

audibles = fetch_data()
print(audibles)


def get_audios_stats(audios):
    print(f"found {len(audios)} files.\n")
    for audio in audios:
        signal = AudioSegment.from_file(audio)  
        channels = signal.channels
        sample_rate = signal.frame_rate
        sample_width = signal.sample_width
        length_ms = len(signal)
        frame_width = signal.frame_width
        if channels != 1 or sample_rate != 8000:
            print(f'{audio}\nchannels: {channels}, sample_rate: {sample_rate}, \
            \nsample_width: {sample_width}, length: {length_ms}, frame_width: {frame_width}\n')
        
get_audios_stats(audibles)  

auxiliary_funcs.save_csv('0_speak.csv', audibles)


def plot_audio_frames(file):
    print(f"file: {file}")
    rate, data = wavfile.read(file)
    print(data.shape)
    plt.title(f"{os.path.split(file)[1]}: whole length")
    plt.plot(data, '-')
    
    #plt.title("{file}: {frame} frames")

    frame = 100
    plt.figure(figsize=(16, 4))
    plt.plot(data[:frame], '.'); 
    plt.plot(data[:frame], '-');

plot_audio_frames(audibles[1])


df = pd.read_csv('0_speak.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    fname = os.path.join(folderMap.RAW_DATA, f)
    rate, samples = wavfile.read(fname)
    print()
    df.at[f, 'length'] = samples.shape[0]/rate

labels = list(np.unique(df.label))
label_distr = df.groupby(['label'])['length'].mean()


## pie plot
fig, axis = plt.subplots()
axis.set_title('Class Distribution', y=1.08)
axis.pie(label_distr, labels=label_distr.index, autopct='%1.1f%%', shadow=False, startangle=90)
axis.axis('equal')
plt.show()

signals, fft, fbank, mfccs = {}, {}, {}, {}

df.reset_index(inplace=True)

resample = False


## applying mask to filter away the high freq
## fill dicts each with a signle wavfile signal
for label in labels:
    wav_file = df[df.label == label].iloc[0, 0] 
    print('wavfiles/' + wav_file)
    signal, rate = librosa.load('wavfiles/'+wav_file, sr=44100)

    if resample:
        print(signal)
        mask = envelope(signal, rate, 0.0005)
        print(f"mask: {mask[:10]}")
        signal = signal[mask]

    signals[label] = signal
    fft[label] = auxiliary_funcs.calc_fft(signal, rate)
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[label] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[label] = mel


## plots    
plot_funcs.plot_signals(signals)
plt.show()

plot_funcs.plot_fft(fft)
plt.show()

plot_funcs.plot_fbank(fbank)
plt.show()

plot_funcs.plot_mfccs(mfccs)
plt.show()


if resample:
    resample_folder = folderMap.RESAMPLES
    
    if os.path.exists(resample_folder):
        shutil.rmtree(resample_folder)
    
    if not os.path.exists(resample_folder):
        os.makedirs(resample_folder)
    
    
    ## downsampling and save wavfiles
    for f in tqdm(df.fname):
        file = f'{{folderMap.RAW_DATA}}/{f}'
        file_re = f'{resample_folder}/{f}'
        print(file)
        signal, rate = librosa.load(file, sr=16000)
        mask = auxiliary_funcs.envelope(signal, rate, 0.0005)
        wavfile.write(filename=file_re, rate=rate, data=signal[mask])
    