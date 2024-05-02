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

class CNNNetwork(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        in_kernels = 1
        out_kernels = 16
        for i in range(self.num_layers):
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_kernels, out_channels=out_kernels, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layers.append(conv)
            
            in_kernels = out_kernels
            out_kernels *= 2
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(5760, 6)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        out = self.layers[0](input_data)
        
        for layer in self.layers[1:]:
            out = layer(out)
        
        out = self.flatten(out)
        out = self.linear(out)
        predictions = self.softmax(out)
        
        return predictions