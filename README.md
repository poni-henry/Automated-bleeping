# Automated-bleeping-of-Swear-words

## Overview
The developed system is a software tool that utilizes deep learning techniques to automatically detects and bleeps out one English swear word in audio recordings.  
The tool employs spectrogram representations of the audio, which is then  processed by a deep learning model trained to identify swear words.  
In the case a swear word is detected, it will be replaced with a bleep sound, and the processed audio will be playedback to the speaker and saved as a new file. If no swear words are found, the original audio recording will be played back to the listener without any modifications.

## Objectives of project
1. To develop a deep learning model that can accurately detect swear words in audio recordings.
2. To implement the model in a software tool that can process audio files and automatically bleep out
detected swear words.
3. To evaluate the performance of the tool on a diverse range of audio recordings and assess its
effectiveness in accurately detecting and bleeping out swear words.

## Tools used to develop system
- Python
- Google collab 
- VS Code

## How to use the software tool
- Download the files in this repository
- Launch the GUI: Run the Python script that contains the GUI code. In command prompt navigate to
folder containing the python script. Type python software_tool_for_swear_word_bleeping.py to lauch the  GUI. A window titled "Audio Recording" will appear.
- Start Recording: Click on the "Start Recording" button. The GUI will display a message indicating that recording has started. Speak into the microphone while avoiding the use of the word "shit" (the sswear word being bleeped).
- Stop Recording and Make Prediction: Click on the "Make Prediction" button. The GUI will display
a message indicating that recording has stopped.The recorded audio will be saved to a file named
"recorded_audio.wav".
- Processing and Prediction Result: The GUI will display a message indicating that the audio is being
processed. The system will convert the recorded audio into a spectrogram image and save it. The
saved spectrogram image will be loaded and resized to match the model's input shape. The system
will use the trained model to make a prediction on whether the recorded audio contains a swear word
or not.
- The GUI will display the prediction result as either "Swear word" or "Non-swear word".
Bleep Sound (if applicable):  
If the prediction result is "Swear word", a bleep sound will be played to censor the offensive content.  
If the prediction result is "Non-swear word", the recorded audio will be played as it is.  
- Repeat: You can repeat the process by clicking on the "Start Recording" button again to record a new
audio sample. The GUI will update accordingly to reflect the recording status and prediction result for
each new audio sample.

## Conclusion
The project successfully developed a software tool for the automated bleeping of English swear words in
audio recordings using deep learning techniques. The trained model demonstrated high accuracy in detecting
and bleeping out swear words and can be trained to detect a wider range swear words. The tool's performance
was evaluated in detecting swear words in sentences and further evaluation and improvement can be done by
testing it with various scenarios and potentially refining the model with additional data.

## Further work
- Refine the testing_on_sentences.py code to correctly replace swear word i swear word location
- Train model to detect more swear words