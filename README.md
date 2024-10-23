# OfflineWordCompletionai

## Overview

`OfflineWordCompletionai` is a project that combines natural language processing (NLP) and audio processing to create a versatile tool for text analysis and audio interaction. It leverages various libraries to perform tasks such as sentiment analysis, collocation extraction, and audio recording with noise filtering.

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
## Contributions

## License