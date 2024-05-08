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