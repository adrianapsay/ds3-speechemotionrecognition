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
    def __init__(self, num_layers, learning_rate, loss_fn, device, train_data_loader, validate_data_loader):
        super().__init__()
        
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.device = device
        self.train_data_loader = train_data_loader
        self.validate_data_loader = validate_data_loader
        self.layers = nn.ModuleList()
        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        
        self.layers = nn.ModuleList()
        in_kernels = 1
        out_kernels = 16
        for i in range(self.num_layers):
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_kernels, out_channels=out_kernels, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(0.00165)
            )
            self.layers.append(conv)
            
            in_kernels = out_kernels
            out_kernels *= 2
        
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(6)
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5)
    
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
        early_stopper = EarlyStopper(patience=6, min_delta=0)
        
        for i in range(1, num_epochs+1):
            print(f'Epoch {i} / {num_epochs} started')
            train_loss = self.__train_single_epoch()
            self.epoch_train_losses.append(train_loss)
            
            valid_loss = self.__validate_single_epoch()
            self.epoch_valid_losses.append(valid_loss)
            
            if early_stopper.early_stop(valid_loss):             
                print(f'Training stopped at Epoch {i} due to early stopping critera')
                break
            
            self.scheduler.step()
            print(f'Epoch {i} / {num_epochs} finished')
            print()
            
        return self.epoch_valid_losses[-1]
    
    def __train_single_epoch(self):
        for inp, target in self.train_data_loader:
            inp, target = inp.to(self.device), target.to(self.device)

            pred = self(inp)
            loss = self.loss_fn(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'Train Loss = {loss.item()}')
        return float(loss.item())
    
    def __validate_single_epoch(self):
        total_loss, n = 0, 0
        
        with torch.no_grad():
            for inp, target in self.validate_data_loader:
                n += 1
                inp, target = inp.to(self.device), target.to(self.device)

                pred = self(inp)
                total_loss += self.loss_fn(pred, target).item()
            
        print(f'Validation Loss = {total_loss / n}')
        return total_loss / n
    
    def predict(self, inp, target, class_mapping):
        #self.eval()
    
        with torch.no_grad():
            prediction_probs = self(inp)
            predicted_index = prediction_probs[0].argmax()
            prediction = class_mapping[predicted_index]
            expected = class_mapping[target]
    
        return prediction, expected
    
    def get_train_epoch_losses(self):
        return self.epoch_train_losses
    
    def get_valid_epoch_losses(self):
        return self.epoch_valid_losses
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False