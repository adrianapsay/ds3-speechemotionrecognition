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

def predict(model, inp, target, class_mapping):
    model.eval()
    
    with torch.no_grad():
        prediction_probs = model(inp)
        predicted_index = prediction_probs[0].argmax()
        prediction = class_mapping[predicted_index]
        expected = class_mapping[target]
    
    return prediction, expected