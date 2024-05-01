import os 
import pandas as pd

from torch.utils.data import Dataset
import torchaudio


import torch
import matplotlib.pyplot as plt

import librosa
import soundfile as sf
from IPython.display import Audio

def resample(signal, sr, target_sample_rate):
    ''' resamples the audio signal to the target sample rate '''
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def mix_down(signal):
   '''mixes down audio signal from stereo to mono'''

   if signal.shape[0] > 1: 
        signal = torch.mean(signal, dim=0, keepdim=True)
   return signal

def cut_signal(signal, num_samples):
    '''cut the length os the signal to a fixed number of samples'''
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal

def right_pad_signal(signal, num_samples):
    '''right pad the signal if longer than the desired number of samples'''
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal 
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def processing_pipeline(signal, params):
    '''applies the processing pipeline to the audio signal
    
    params: dictionary containing the following
    - sr: sample rate
    - target_sample_rate: target sample rate
    - num_samples: number of samples to cut the signal to
    '''
    sr = params['sr']
    target_sample_rate = params['target_sample_rate']
    num_samples = params['num_samples']

    signal = resample(signal, sr, target_sample_rate)
    signal = mix_down(signal)
    signal = cut_signal(signal, num_samples)
    signal = right_pad_signal(signal, num_samples)
    return signal