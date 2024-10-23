# OfflineWordCompletionai

## Overview

`OfflineWordCompletionai` is a project that combines natural language processing (NLP) and audio processing to create a versatile tool for text analysis and audio interaction. It leverages various libraries to perform tasks such as sentiment analysis, collocation extraction, and audio recording with noise filtering.

## System Requirements

- Python 3.6+
- A functioning microphone for audio recording.
- GPU support (optional but recommended for faster processing of NLP tasks with GPT-2).


## Features

- **NLP Pipeline**:
  - Preprocess text for analysis (tokenization, lemmatization, POS tagging).
  - Generate summaries based on sentiment scores.
  - Create dialogues based on collocations.
  - Chatbot response generation using the GPT-2 model.

- **Audio Processing**:
  - Record audio with adjustable duration.
  - Apply offline and online noise filtering to improve audio quality.
  - Transcribe recorded audio using the PocketSphinx speech recognition system.


## Project Structure

- `stt.py`: Contains the natural language processing pipeline, including sentiment analysis, dialogue generation, and GPT-2 response generation.
- `vctest.py`: Handles the audio recording, noise filtering, and transcription functionalities.
- `requirements.txt`: Specifies the necessary dependencies for the project.
- `README.md`: Project overview and usage instructions.

## Installation

To run this project, you need to have Python 3.6 or higher. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

- **NLP Pipeline:** You can run the NLP pipeline by using the stt.py script. It takes user input and processes it for various analyses, including sentiment and collocation extraction.

- **Audio Recorder:** To use the audio recording functionality, run the vctest.py script. This will allow you to record audio, apply noise filtering, and transcribe the recorded audio.

## Example

- **NLP Pipeline Usage:**
```bash
# Example of using the NLP pipeline
text = "Your sample text here."
results = run_nlp_pipeline(text)
print(results)

```


- **Audio Recording Usage:**

```bash
# Run the Streamlit app for audio recording
if __name__ == "__main__":
    main()
```

### **Testing**
Add a section for testing or running the project locally:

```markdown
## Testing

To test the NLP pipeline or the audio functionality locally:

1. Run the `stt.py` file for text processing features.
2. Run the `vctest.py` file for audio recording and processing.

For example, to test the audio recording:

```bash
python vctest.py
```

### **Contributing Guidelines**
Add more detailed instructions for how contributors can add to the project, format code, or raise issues.


## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.


## License

**This project is licensed under the MIT License. See the LICENSE file for details.**

## Known Issues

- Transcription accuracy may vary depending on the audio quality.
- GPT-2 chatbot response times may be slower on CPUs; using a GPU is recommended.
