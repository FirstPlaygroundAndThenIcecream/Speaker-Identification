# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:09:01 2020

@author: xianl_pmrkzzf
"""

import os


class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=8000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
