import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa
import os
import soundfile as sf
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Set audio parameters
sample_rate = 44100
target_shape = (256, 256)

# Load the model
model = keras.models.load_model('swear_detection_model.h5')

# Load the bleep sound
bleep_sound, bleep_sr = librosa.load('bleep.wav', sr=sample_rate)

def obtain_spectrogram(segment, window_length, overlap, plot=False, save=False, save_dir=None):
    # Compute the spectrogram of the segment
    audio = segment.astype(np.float32)
    window_length = int(window_length / 1000 * sample_rate)
    overlap = int(overlap * window_length)
    nfft = 2 ** (int(np.log2(window_length)) + 1)
    spectrogram = librosa.stft(audio, n_fft=nfft, hop_length=window_length - overlap, win_length=window_length)
    spectrogram = np.abs(spectrogram) ** 2
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max(spectrogram))
    spectrogram = np.clip(spectrogram, -40, -3)

    # Resize spectrogram to match the input shape expected by the model
    spectrogram = resize(spectrogram, target_shape)
    spectrogram = np.expand_dims(spectrogram, axis=-1)

    if save:
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        librosa.display.specshow(spectrogram[:, :, 0], sr=sample_rate, hop_length=window_length - overlap, n_fft=nfft,
                                 win_length=window_length, shading='auto')
        plt.savefig(save_dir)
        plt.close()

    if plot:
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogram[:, :, 0], sr=sample_rate, hop_length=window_length - overlap, n_fft=nfft,
                                 win_length=window_length, shading='auto')
        plt.title("Log-frequency power spectrogram")
        plt.show()

    return spectrogram[:, :, 0]

# Function to process audio segments and locate swear words
def process_audio_segments(audio_segments):
    swear_word_locations = []
    segment_duration = len(audio_segments[0]) / sample_rate

    for i, segment in enumerate(audio_segments):
        segment_time = i * segment_duration

        # Obtain spectrogram for the segment
        window_length = 40   # 40ms
        overlap = 0.5        # 50%
        spectrogram = obtain_spectrogram(segment, window_length, overlap)
        # Normalize spectrogram
        spectrogram /= np.max(np.abs(spectrogram))

        # Reshape spectrogram to match the input shape expected by the model
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        # Duplicate the single-channel spectrogram along the channel axis
        spectrogram = np.concatenate([spectrogram] * 3, axis=-1)

        # Make prediction using the model
        prediction = model.predict(np.expand_dims(spectrogram, axis=0))
        predicted_class_index = np.argmax(prediction)

        # If predicted class is a swear word, record the location
        if predicted_class_index == 0:
            swear_word_locations.append(segment_time)

    return swear_word_locations

# Function to replace swear word locations with bleep sound
def replace_swear_words(audio, locations, bleep_sound):
    for location in locations:
        start_sample = int(location * sample_rate)
        end_sample = start_sample + len(bleep_sound)
        audio[start_sample:end_sample] = bleep_sound

    return audio

def start_recording():
    duration = 10  # Recording duration in seconds

    print("Recording started. Speak into the microphone...")

    # Start recording
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished

    print("Recording finished.")

    # Save the audio recording to a file
    output_file = "recording.wav"
    sf.write(output_file, audio, sample_rate)
    print(f"Audio recording saved as {output_file}.")

    # Process the recorded audio
    audio = audio.flatten()
    audio = audio.astype(np.float32)

    # Limit audio duration to 100 seconds
    audio = audio[:5 * sample_rate]

    # Split the audio into segments
    segment_duration = 1.0  # Duration of each segment in seconds
    segment_samples = int(segment_duration * sample_rate)
    num_segments = len(audio) // segment_samples

    # Process audio segments and locate swear words
    audio_segments = np.array_split(audio[:num_segments * segment_samples], num_segments)
    swear_word_locations = process_audio_segments(audio_segments)

    if len(swear_word_locations) > 0:
        print("Swear word detected. Replacing with bleep sounds.")

        # Generate bleep sound for each segment
        bleep_full = np.tile(bleep_sound, len(audio_segments))
        bleep_full = bleep_full[:len(audio)]

        # Replace swear word locations with bleep sound
        audio = replace_swear_words(audio, swear_word_locations, bleep_full)

        # Save the modified audio with bleep sounds
        modified_file = "recording_with_bleep.wav"
        sf.write(modified_file, audio, sample_rate)
        print(f"Modified audio saved as {modified_file}.")

        # Play back the modified audio with bleep sounds
        os.system(f"afplay {modified_file}")
    else:
        # Save the original audio
        original_file = "recording_original.wav"
        sf.write(original_file, audio, sample_rate)
        print(f"No swear words detected. Original audio saved as {original_file}.")

        # Play back the original audio
        os.system(f"afplay {original_file}")

# Create a tkinter window
window = tk.Tk()
window.title("Swear Word Detection")

# Create a label
label = tk.Label(window, text="Click the 'Record' button to start recording.")
label.pack(pady=20)

# Create a record button
record_button = tk.Button(window, text="Record", command=start_recording)
record_button.pack(pady=10)

# Run the tkinter event loop
window.mainloop()
