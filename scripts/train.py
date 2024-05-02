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

def create_data_loader(train_data, batch_size):
    data_loader = DataLoader(train_data, batch_size = batch_size)
    return data_loader

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for inp, target in data_loader:
        inp, target = inp.to(device), target.to(device)
        
        pred = model(inp)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Loss = {loss.item()}')

def train(model, data_loader, loss, optimizer, device, num_epochs):
    print('Started training\n')
    for i in range(1, num_epochs+1):
        print(f'Epoch {i} / {num_epochs} started')
        train_single_epoch(model, data_loader, loss, optimizer, device)
        print(f'Epoch {i} / {num_epochs} finished')
        print()