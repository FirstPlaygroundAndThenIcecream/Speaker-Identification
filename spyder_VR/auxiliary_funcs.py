# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:12:45 2020

@author: Lei Xian
"""

import pandas as pd
import numpy as np
import csv
import os


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