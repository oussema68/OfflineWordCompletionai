import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
from pocketsphinx import AudioFile, get_model_path
import speech_recognition as sr
import multiprocessing
import os

# Importing the offline and online noise filtering functions
def offline_noise_filter(data, window_size):
    """
    Applies offline noise filtering by convolving the audio data with a moving average filter.
    This smoothens the data over a fixed window size.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def online_noise_filter(new_data, prev_filtered_data, window_size):
    """
    Applies online noise filtering using a moving average filter.
    This updates the filtered data in real-time as new audio data is captured.
    """
    updated_data = np.append(prev_filtered_data, new_data)
    updated_data = updated_data[-window_size:]
    return np.convolve(updated_data, np.ones(window_size)/window_size, mode='valid')

# Create a session state to persistently store variables
session_state = st.session_state

def create_directory(directory):
    """
    Creates a directory if it does not exist.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")

def create_file(filename):
    """
    Creates a new file. Raises an error if the file already exists.
    """
    try:
        with open(filename, 'x'):  # 'x' mode creates a new file; raises an error if the file already exists
            print(f"File '{filename}' created successfully.")
    except Exception as e:
        print(f"Error creating file '{filename}': {e}")

def capture_audio(duration):
    """
    Captures audio for the specified duration using sounddevice and returns the recorded data.
    """
    st.text("Recording audio... Click the button below to stop recording.")
    audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype=np.int16)
    sd.wait()  # Wait until the recording is done
    st.text("Audio recording complete.")
    return audio_data

# Queue for handling multiprocessing tasks like transcription
result_queue = multiprocessing.Queue()

def play_audio(audio_data, audio_count):
    """
    Plays the recorded audio after applying offline and online noise filtering. Saves the audio to a file
    and transcribes it.
    """
    st.text("Playing recorded audio...")

    # Apply offline noise filtering to the audio data
    window_size_offline = 3
    audio_data_filtered = offline_noise_filter(audio_data[:, 0], window_size_offline)
    audio_data_filtered = np.column_stack((audio_data_filtered, audio_data_filtered))  # Duplicate for stereo

    # Apply online noise filtering to the audio data
    window_size_online = 3
    prev_filtered_data_online = audio_data_filtered[-window_size_online:, 0]
    for i in range(len(audio_data_filtered)):
        audio_data_filtered[i] = online_noise_filter(audio_data_filtered[i, 0], prev_filtered_data_online, window_size_online)
        prev_filtered_data_online = audio_data_filtered[i, 0]

    # Create 'audios' directory if it doesn't exist
    create_directory('audios')

    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save the filtered audio data to a WAV file
    wav_filename = f'audios/{audio_count}.wav'
    create_file(wav_filename)
    wav.write(wav_filename, 44100, audio_data_filtered)

    # Display the saved audio file in the app for playback
    st.audio(wav_filename, format='audio/wav')

    # Transcribe the audio using speech recognition
    result = transcribe_audio(wav_filename)

    # Display the transcription result
    if result:
        st.text(f"Transcription: {result}")
    else:
        st.text("Could not understand audio.")

def transcribe_audio(audio_path, num_runs=3):
    """
    Transcribes the given audio file using the PocketSphinx speech recognition engine. 
    The transcription is attempted multiple times to get a consistent result.
    """
    recognizer = sr.Recognizer()

    results = []  # Store transcription results from multiple runs

    for _ in range(num_runs):
        try:
            # Load the audio file for transcription
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)

                # Use pocketsphinx for offline transcription
                result = recognizer.recognize_sphinx(audio_data)
                results.append(result)

        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Error in offline transcription: {e}")

    # Return the most common transcription result
    valid_results = [result for result in results if result is not None]
    if valid_results:
        most_common_result = max(set(valid_results), key=valid_results.count)
        return most_common_result
    else:
        return None

def main():
    """
    Main function for the Streamlit app. Allows users to record audio, play it back, and display transcriptions.
    """
    st.title("Audio Recorder and Player")

    # User selects the duration of the audio recording
    duration = st.slider("Select recording duration (seconds):", 1, 10, 5)

    # Start recording when the user clicks the button
    if st.button("Record Audio"):
        if 'audio_count' not in session_state:
            session_state.audio_count = 1
        else:
            session_state.audio_count += 1

        # Capture audio and then play it back
        audio_data = capture_audio(duration)
        play_audio(audio_data, session_state.audio_count)

# Entry point of the script
if __name__ == "__main__":
    main()
