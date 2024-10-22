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
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def online_noise_filter(new_data, prev_filtered_data, window_size):
    updated_data = np.append(prev_filtered_data, new_data)
    updated_data = updated_data[-window_size:]
    return np.convolve(updated_data, np.ones(window_size)/window_size, mode='valid')

# Create a session state to persistently store variables
session_state = st.session_state

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")

def create_file(filename):
    try:
        with open(filename, 'x'):  # 'x' mode creates a new file; raises an error if the file already exists
            print(f"File '{filename}' created successfully.")
    except Exception as e:
        print(f"Error creating file '{filename}': {e}")

def capture_audio(duration):
    st.text("Recording audio... Click the button below to stop recording.")
    audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype=np.int16)
    sd.wait()
    st.text("Audio recording complete.")
    return audio_data

result_queue = multiprocessing.Queue()

def play_audio(audio_data, audio_count):
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
    
    # WAV file
    wav_filename = f'audios/{audio_count}.wav'
    create_file(wav_filename)
    wav.write(wav_filename, 44100, audio_data_filtered)

    # Display the WAV audio
    st.audio(wav_filename, format='audio/wav')

    # Transcribe the audio in a separate process
    #process = multiprocessing.Process(target=transcribe_worker, args=(wav_filename, result_queue))
   # process.start()
    #st.text("ebda nayek")

    # Wait for the transcription process to finish, with a timeout
  #  process.join(timeout=30)  # You can adjust the timeout as needed
   # st.text("join zok omok")

    # If the process is still alive, terminate it
    #if process.is_alive():
     #   process.terminate()
   # st.text("not alive nor dead")

    # Retrieve the transcription result
    result = transcribe_audio(wav_filename)

    if result:
        st.text(f"Transcription: {result}")
    else:
        st.text("Could not understand audio.")

def transcribe_audio(audio_path, num_runs=3):
    recognizer = sr.Recognizer()

    results = []

    for _ in range(num_runs):
        try:
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)

                # Use pocketsphinx for offline transcription
                result = recognizer.recognize_sphinx(audio_data)
                results.append(result)

        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Error in offline transcription: {e}")

    # Filter out None results and return the most common transcription
    valid_results = [result for result in results if result is not None]
    if valid_results:
        most_common_result = max(set(valid_results), key=valid_results.count)
        return most_common_result
    else:
        return None

def main():
    st.title("Audio Recorder and Player")

    duration = st.slider("Select recording duration (seconds):", 1, 10, 5)

    if st.button("Record Audio"):
        if 'audio_count' not in session_state:
            session_state.audio_count = 1
        else:
            session_state.audio_count += 1

        audio_data = capture_audio(duration)
        play_audio(audio_data, session_state.audio_count)

if __name__ == "__main__":
    main()
