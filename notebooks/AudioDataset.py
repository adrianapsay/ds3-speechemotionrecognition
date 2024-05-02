import os 
import pandas as pd

from torch.utils.data import Dataset
import torchaudio


import torch
import matplotlib.pyplot as plt

import librosa
import soundfile as sf
from IPython.display import Audio

from audio_processing_functions import *


class AudioEmotionDataset(Dataset):
    # audio_dir example: '../data/Crema'
    def __init__(self, 
                 audio_dir, 
                 transformation, 
                 target_sample_rate, 
                 num_samples, 
                 device):
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.transformation = transformation.to(self.device)
        self.num_samples = num_samples
        
        wav_file_paths = os.listdir(audio_dir)
        
        # create data frame out of wav_files
        # emotion is listed and can be used by label
        # idk what the notsure column is again
        emotions = pd.DataFrame(wav_file_paths, columns=['filename'])
        emotions['filename'] = emotions['filename'].str.split('_')
        emotions = pd.DataFrame(emotions['filename'].tolist(), columns=['id', 'notsure', 'emotion', 'version'])
        emotions['filename'] = wav_file_paths
        
        # emotions data frame with file path
        self.emotions = emotions
        
        # could pass this in on as a fixed map if we know that the labels will not change in each dataset
        emotion_map = dict()

        label_num = 0
        for emotion in self.emotions['emotion'].unique():
            emotion_map[emotion] = label_num
            label_num += 1
        
        # save the map for interpretation
        self.emotion_map = {'HAP': 0, 'NEU': 1, 'ANG': 2, 'FEA': 3, 'DIS': 4, 'SAD': 5}
        
        # retain a column with emotions letter code for interpretability
        # map emotions column to number for PyTorch model 
        self.emotions['emotion_code'] = self.emotions['emotion']
        self.emotions['emotion'] = self.emotions['emotion'].map(emotion_map)
    
    def __len__(self):
        # return len of dataframe
        return len(self.emotions)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        
        # load audio file with torch audio
        # i think 2 channels is a stereo audio
        # signal -> (num_channels, samples) --> (2, 16000) --> (1, 16000)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        
        # resize the array so that they all match in size
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        # transform the signal with a mel spectogram 
        # that is passed in 
        signal = self.transformation(signal)
        
        signal = torch.log(signal + 0.00001)
        return signal, label
        
    def _get_audio_sample_path(self, index):
        # get the fold (idk what that is do we have that?)
        # index 4 corresponds to the file name
        path = os.path.join(self.audio_dir, self.emotions.iloc[index, 4])
        
        return path 
    
    def _get_audio_sample_label(self, index):
        # index 2 of columns corresponds to the emotion label
        return self.emotions.iloc[index, 2]
    
    def _resample_if_necessary(self, signal, sr):
        return resample(signal, sr, self.target_sample_rate)
    
    def _mix_down_if_necessary(self, signal):
        # if we have a signal with multiple channels 
        # we will need to mix the signal down from stereo (or whatever)
        # and make it mono
        return mix_down(signal)
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50,000)
        
        # if the signal has more samples than the expected number
        # of samples, we need to cut it down
        return cut_signal(signal, self.num_samples)
    
    def _right_pad_if_necessary(self, signal):
        return right_pad_signal(signal, self.num_samples)