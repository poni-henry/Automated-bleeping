import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa
import os
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from PIL import Image

import shutil
import soundfile as sf

# Create a tkinter window instance
window = tk.Tk()
window.title("Audio Recording")
window.geometry("400x200")

# Create a label to display processing message
processing_status = tk.StringVar()
processing_status.set("")

recording_status = tk.StringVar()
recording_status.set("Press Start to begin recording.")

status_label = tk.Label(window, textvariable=recording_status)
status_label.pack(pady=10)


# Load the model
model = keras.models.load_model('swear_detection_model.h5')

# Set the audio parameters
fs = 16000
# Define the sampling rate
sr = 44100 
# Variable to store the recorded audio
audio = None

my_list = []


# Function to start recording
def start_recording():
    global audio
    audio = None
    recording_status.set("Recording started. Do not say the word 'shit'.")
    duration = 1.0  # Set the recording duration to 1 second
    frames = int(duration * fs)
    audio = sd.rec(frames, samplerate=fs, channels=1)

# Function to stop recording and make prediction
def stop_prediction():
    global audio
    recording_status.set("Recording stopped.")
    sd.stop()
    # Save audio to file
    sf.write('recorded_audio.wav', audio[:, 0], fs)
    print("Audio saved to file.")
    my_list.append('recorded_audio.wav')  # Add the recorded audio file to the list
    window.after(0, lambda: processing_status.set("Processing audio..."))
    window.update()  # Update the window to immediately show the processing message
    check_for_swear_word()

def check_for_swear_word():
    def obtain_specgram(filepath, window_length, overlap, plot=False, save=False, save_dir=None):
        # This function returns the normalized power spectrogram of an audio
        audio, fs = librosa.load(filepath)
        window_length = int(window_length/1000*fs)
        overlap = int(overlap * window_length)
        nfft = 2**(int(np.log2(window_length)) + 1)
        S = librosa.stft(audio, n_fft=nfft, hop_length=window_length-overlap, win_length=window_length)
        S = np.abs(S)**2
        S = librosa.power_to_db(S, ref=np.max(S))
        S = np.clip(S, -40, -3)
        if save:
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            librosa.display.specshow(S, sr=fs, hop_length=window_length-overlap, n_fft=nfft,
                                    win_length=window_length, y_axis='log', x_axis='time')
            plt.savefig(save_dir)
            plt.close()
        if plot:
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(S, sr=fs, hop_length=window_length-overlap, n_fft=nfft,
                                    win_length=window_length, y_axis='log', x_axis='time')
            plt.title("Log-frequency power spectrogram")
            plt.show()
        return S
    
    dataset_folder = pathlib.Path("audio_code")
    specgram_folder = pathlib.Path('generated_spectrograms/recorded_audio')

    shutil.rmtree(specgram_folder, ignore_errors=True)
    os.makedirs(specgram_folder, exist_ok=True)

    window_length = 40   # 40ms
    overlap = 0.5        # 50%

    for audio_file in my_list:
        audio_path = pathlib.Path(audio_file)
        audio_name = audio_path.stem
        print(audio_name)

    save_dir = specgram_folder / audio_name
    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir / f"{audio_name}.png"
    print(save_path)
    obtain_specgram(audio_file, window_length, overlap, plot=False, save=True, save_dir=save_path)
    class_names = ['Swear word', 'Non swear word']
    
    # Set the path to the spectrogram image
    image_dir = 'generated_spectrograms\recorded_audio\recorded_audio\recorded_audio.png'
    image_name = 'recorded_audio.png'
    image_path = os.path.join(save_dir, f"{audio_name}.png")


    # Load the spectrogram image
    spectrogram_image = Image.open(image_path)
    # Resize the spectrogram image to match the expected input shape of the model
    desired_shape = (256, 256)
    spectrogram_image = spectrogram_image.resize(desired_shape)

    # Convert the image to a numpy array
    spectrogram = np.array(spectrogram_image)

    # If the image has an alpha channel, remove it
    if spectrogram.shape[2] == 4:
        spectrogram = spectrogram[:, :, :3]

    # Reshape the spectrogram to match the input shape expected by the model
    spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2])

    # Make predictions
    predictions = model.predict(spectrogram)
    probabilities = tf.keras.layers.Softmax()(predictions)
    predicted_class_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_index]

    processing_status.set("Prediction: " + predicted_class)
    window.update()  # Update the window to immediately show the prediction result

    print("Prediction: ", predicted_class)
    if predicted_class == "Swear word":
        bleep_sound, _ = librosa.load('bleep.wav', sr=sr, duration=1.0)
        sf.write("recorded_audio_with_bleeps.wav", bleep_sound, fs)
        # Play the bleep sound
        sd.play(bleep_sound, fs)
    else:
        # Play the recorded audio
        sd.play(audio, fs)

# Create a label to display processing status
processing_label = tk.Label(window, textvariable=processing_status)
processing_label.pack(pady=10)

# Create Start and Stop buttons
start_button = tk.Button(window, text="Start Recording", command=start_recording)
start_button.pack(side="left", padx=10, pady=10)

stop_button = tk.Button(window, text="Make Prediction", command=stop_prediction)
stop_button.pack(side="left", padx=10, pady=10)

# Run the GUI window
window.mainloop()
