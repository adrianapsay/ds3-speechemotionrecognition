from AudioDataset import *
from AudioModel import *
import wandb

import torch 
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm
 # https://github.com/musikalkemist/pytorchforaudio/blob/main/09%20Training%20urban%20sound%20classifier/train.py

# BATCH_SIZE = 128
# EPOCHS = 10
# LEARNING_RATE = 0.001

# AUDIO_DIR = '../data/Crema'
# SAMPLE_RATE = 16000
# NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


# another word for loss_fn is criterion i believe
def train(model, data_loader, loss_fn, optimiser, device, epochs):
    # Tell wandb to track thte model's gradients, weights, and loss
    wandb.watch(model, loss_fn, log='all', log_freq=10)
    
    total_batches = len(data_loader) * epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}")
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)
            
            example_ct += len(input)
            batch_ct += 1

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        print(f"loss: {loss.item()}")
        print("---------------------------")
    print("Finished training")
    

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    
    
# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"
#     print(f"Using {device}")

#     # instantiating our dataset object and create data loader
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=1024,
#         hop_length=512,
#         n_mels=64
#     )

#     usd = UrbanSoundDataset(AUDIO_DIR,
#                             mel_spectrogram,
#                             SAMPLE_RATE,
#                             NUM_SAMPLES,
#                             device)
    
#     train_dataloader = create_data_loader(usd, BATCH_SIZE)

#     # construct model and assign it to device
#     cnn = CNNNetwork().to(device)
#     print(cnn)

#     # initialise loss funtion + optimiser
#     loss_fn = nn.CrossEntropyLoss()
#     optimiser = torch.optim.Adam(cnn.parameters(),
#                                  lr=LEARNING_RATE)

#     # train model
#     train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

#     # save model
#     torch.save(cnn.state_dict(), "feedforwardnet.pth")
#     print("Trained feed forward net saved at feedforwardnet.pth")