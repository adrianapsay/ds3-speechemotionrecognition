import streamlit as st 
import torch 

# python package to manage binary data and treat it as a file
# without even needing to write it to disk
from io import BytesIO 
#from AudioModel import * # CNNNetwork()
from AudioDataset import * # AudioEmotionDataset()

from cnnnet import * # import Vrisan's model 

# Pytorch Ploting Functions required imports 
from IPython.display import Audio
from matplotlib.patches import Rectangle
from torchaudio.utils import download_asset

# audio recorder project
# https://pypi.org/project/audio-recorder-streamlit/
from audio_recorder_streamlit import audio_recorder


# https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None: 
        fig, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

    st.pyplot(fig)
    
def plot_spectrogram(specgram, title=None, ylabel="freq_bin (decibels)"):
    # Creating a figure and an axis object
    fig, ax = plt.subplots(1, 1)
    
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Convert power spectrogram to decibel units
    db_specgram = librosa.power_to_db(specgram) # np.log(specgram + 1) 

    # Plot the masked spectrogram
    im = ax.imshow(db_specgram, origin="lower", aspect="auto", interpolation="nearest", cmap='inferno')
    fig.colorbar(im, ax=ax)  # Adding a colorbar

    # Display the plot in Streamlit
    st.pyplot(fig)

# TODO: research what a filter bank is? do we need this? 
def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin") # only for mel spectrograms? 

    st.pyplot(fig)

# TODO: add this to the Readme.md file to run our app 
# streamlit run app.py --server.enableXsrfProtection false
# this prevents the error: AxiosError: Request failed with status code 403
def main():
    st.title('Audio Emotion Classifier')
    
    # option for user to upload a file from their computer
    audio_file = st.file_uploader("Upload an audio file", type=['wav'])

    # option for user to record audio
    audio_bytes = audio_recorder()

    if audio_bytes:
        # creates a bytesio object that behaves like a file, but is stored within memory
        audio_file = BytesIO(audio_bytes)

    if audio_file is not None:

        # displays an audio player so that the user can listen to the audio file
        st.audio(audio_file, format="audio/wav")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load file as pytorch tensor
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(device)

        # hyperparameters for the model and processing pipeline
        hyperparameters = dict(
        target_sample_rate=16000,
        num_samples=60000, # 22050
        n_fft=1024, 
        hop_length=512, 
        n_mels=64, 
        sr = sr
        )


        # process the signal so it's in desired format

        # imported from audio_processing_functions.py
        signal = processing_pipeline(signal, hyperparameters)

        waveform_signal = signal # Save the waveform signal for later
        print(signal.shape)

        plot_waveform(signal, sr, title="Waveform")


        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=hyperparameters['target_sample_rate'],
            n_fft=hyperparameters['n_fft'],
            hop_length=hyperparameters['hop_length'],
            n_mels=hyperparameters['n_mels']
        )

        # apply transformation to the signal
        signal = mel_spectrogram(signal)

        # save the mel_spectrogram so that we can show the
        # user what it looks like

        # turn the spectrogram into 2 dimensional image for matplotlib 
        spectrogram_output = signal.squeeze() 

        print(spectrogram_output.shape)
        # TODO: try plotting with librosa instead
        plot_spectrogram(spectrogram_output, title="Mel Spectrogram")

        # convert to decibels (kinda)
        # is there a way we can improve on this?
        signal = torch.log(signal + 0.00001)

        # add a channel dimension so that model thinks it's a batch
        signal = signal.unsqueeze(0) 

        # Load the model
        model = CNNNetwork(num_layers=4).to(device)
        model.load_state_dict(torch.load('cnn.pth', map_location=torch.device("cpu")))
        model.eval()
        # Make a prediction

        emotion_map = ['HAP', 'NEU', 'ANG', 'FEA', 'DIS', 'SAD']

        with torch.no_grad():
            predictions = model(signal)
            #st.write('Predictions:', predictions)
            predicted_index = predictions.argmax(1).item()
            st.write(f'Predicted class index: {emotion_map[predicted_index]}')

            predictions = predictions.squeeze().numpy()
            #st.write(predictions)

            df = pd.DataFrame({'emotion': emotion_map, 'probability': predictions})
            st.header('Predicted Emotion Probabilities')
            st.bar_chart(data=df, x='emotion', y='probability', use_container_width=True)


    

if __name__ == "__main__":
    main()
