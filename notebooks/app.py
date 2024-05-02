import streamlit as st
import torch
from io import BytesIO
#from your_pytorch_model_module import classify_audio  # Ensure this is your PyTorch model function
from AudioModel import * # CNNNetwork()
from AudioDataset import * # AudioEmotionDataset()


# streamlit run app.py --server.enableXsrfProtection false
# this prevents the error: AxiosError: Request failed with status code 403
def main():
    st.title('Audio Emotion Classifier')
    
    # Upload audio file or record audio
    audio_file = st.file_uploader("Upload an audio file", type=['wav'])
    record = st.checkbox("Or record audio")

    if record:
        audio_data = st.audio_recorder("Record your audio", type="wav", time_limit=10)
        if audio_data:
            audio_bytes = audio_data.read()
            audio_file = BytesIO(audio_bytes)

    if audio_file is not None:
        # Assuming classify_audio expects a file-like object
        #label = classify_audio(audio_file)
        #st.write(f"The predicted category is: {label}")
        st.write("Audio file uploaded")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using {device}")
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(device)

        hyperparameters = dict(
        target_sample_rate=16000,
        num_samples=22050, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64, 
        sr = sr
        )

        # process the signal so it's in desired format
        signal = processing_pipeline(signal, hyperparameters)

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
        spectrogram_output = signal 

        # convert to decibels (kinda)
        signal = torch.log(signal + 0.00001)

        # add a channel dimension so that model thinks it's a batch
        signal = signal.unsqueeze(0) 

        # Load the model
        model = CNNNetwork().to(device)
        model.load_state_dict(torch.load('feedforwardnet.pth', map_location=torch.device("cpu")))
        model.eval()
        # Make a prediction

        emotion_map = ['HAP', 'NEU', 'ANG', 'FEA', 'DIS', 'SAD']

        with torch.no_grad():
            predictions = model(signal)
            st.write('Predictions:', predictions)
            predicted_index = predictions.argmax(1).item()
            st.write(f'Predicted class index: {emotion_map[predicted_index]}')

            predictions = predictions.squeeze().numpy()
            st.write(predictions)

            df = pd.DataFrame({'emotion': emotion_map, 'probability': predictions})
            st.bar_chart(data=df, x='emotion', y='probability', use_container_width=True)


    

if __name__ == "__main__":
    main()
