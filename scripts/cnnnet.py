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
    def __init__(self, num_layers, learning_rate, loss_fn, device, data_loader):
        super().__init__()
        
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.device = device
        self.data_loader = data_loader
        self.layers = nn.ModuleList()
        self.epoch_losses = []
        
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
        self.linear = nn.LazyLinear(6)
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, input_data):
        out = self.layers[0](input_data)
        
        for layer in self.layers[1:]:
            out = layer(out)
        
        out = self.flatten(out)
        out = self.linear(out)
        predictions = self.softmax(out)
        
        return predictions
    
    def train(self, num_epochs):
        self.epoch_losses = []
        
        for i in range(1, num_epochs+1):
            print(f'Epoch {i} / {num_epochs} started')
            self.epoch_losses.append(self.__train_single_epoch())
            print(f'Epoch {i} / {num_epochs} finished')
            print()
            
        return epoch_losses[-1]
            
    def predict(self, inp, target, class_mapping):
        self.eval()
    
        with torch.no_grad():
            prediction_probs = self(inp)
            predicted_index = prediction_probs[0].argmax()
            prediction = class_mapping[predicted_index]
            expected = class_mapping[target]
    
        return prediction, expected
    
    def __train_single_epoch(self):
        for inp, target in self.data_loader:
            inp, target = inp.to(self.device), target.to(self.device)

            pred = self(inp)
            loss = self.loss_fn(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'Loss = {loss.item()}')
        return float(loss.item())
    
    def get_epoch_losses(self):
        return self.epoch_losses