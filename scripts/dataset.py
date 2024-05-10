import os
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, data_path, final_sample_rate, num_samples_limit, transformation, device):
        
        self.data_path = data_path
        
        self.final_sample_rate = final_sample_rate
        self.num_samples_limit = num_samples_limit
        self.device = device
        self.transformation = transformation.to(self.device)
        self.df = pd.read_csv(data_path)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        audio_path = self.get_audio_path(index)
        label = self.get_label(index)
        signal, sr = torchaudio.load(audio_path)
        
        signal = signal.to(self.device)
        signal = self.to_mono_if_necessary(signal)
        signal = self.resample_if_necessary(signal, sr)
        signal = self.cut_down_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        
        signal[signal == 0] = 0.000000001
        signal = torch.log(signal)
        
        return signal, label
    
    def resample_if_necessary(self, signal, original_sr):
        if original_sr != self.final_sample_rate:
            resampler = torchaudio.transforms.Resample(original_sr, self.final_sample_rate).to(self.device)
            signal = resampler(signal)
        
        return signal
    
    def to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True).to(self.device)
        return signal
    
    def cut_down_if_necessary(self, signal):
        return signal[:, :self.num_samples_limit].to(self.device)
    
    def right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples_limit:
            num_missing_samples = self.num_samples_limit - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding).to(self.device)
        return signal
        
    def get_audio_path(self, index):
        return self.df.iloc[index].filename
    
    def get_label(self, index):
        return self.df.iloc[index].category_num
    
    def plot_spectrogram(self, index, log=True):
        spec = self.__getitem__(index)[0]
        spec = spec.cpu().numpy().reshape(spec.shape[1], -1)
        frame_rate = self.final_sample_rate / self.transformation.hop_length
        seconds = np.arange(spec.shape[1]) / frame_rate
        
        if not log:
            spec = np.exp(spec)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(spec, cmap='inferno', origin='lower', aspect='auto')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel')
        plt.colorbar(label='Intensity (dB)')
        plt.xticks(np.linspace(0, spec.shape[1], 12)[:-1], np.linspace(0, seconds[-1], 12).round(2)[:-1])
        plt.show()