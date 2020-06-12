# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:44:02 2020

"""

import matplotlib.pyplot as plt


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Samples', size=16)
    i = 0
    for x in range(4):
        for y in range(3):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i = i + 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(4):
        for y in range(3):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i = i + 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    
    i = 0
    
    for x in range(4):
        for y in range(3):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    
    i = 0
    for x in range(4):
        for y in range(3):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


















