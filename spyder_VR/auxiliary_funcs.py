# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:12:45 2020

"""

import pandas as pd
import numpy as np
import math
import csv
import os
from pydub import AudioSegment


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


def save_csv(file_name, files):
    with open(file_name, mode='w') as csv_file:
        header = ['fname', 'label']
        csv_writer = csv.DictWriter(csv_file, fieldnames=header)

        csv_writer.writeheader()
        if len(files) > 0:
            for file in files:
                fname = os.path.split(file)[1]
                label = fname.split('.')[0].split('_')[1]
                csv_writer.writerow({'fname': fname, 'label': label})
        else:
            print("nothing to write to csv.")


def trim_audio(audio_path, dest_folder):
    signal = AudioSegment.from_file(audio_path)
    length = len(signal)
    segm_qty = 5 
    segm_len = math.floor(length/segm_qty)
#    segm_len = 200
#    segm_qty = math.floor(length/segm_len)
    for i in range(segm_qty):
        audio_name = f"{i}_{os.path.split(audio_path)[1]}"
        export_path = os.path.join(dest_folder, audio_name)
        signal_segm = signal[segm_len*i : segm_len*(i+1)]
        signal_segm.export(out_f=export_path, format='wav')
        print(export_path)