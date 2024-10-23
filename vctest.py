import torch  # Importing PyTorch for deep learning and tensor computations
from typing import Dict, List, Tuple  # Importing type hints for better code clarity
import nltk  # Natural Language Toolkit for NLP tasks

# Uncomment the lines below if you need to download these NLTK resources
# nltk.download('punkt')  # Tokenizer
# nltk.download('wordnet')  # WordNet, used for lemmatization
# nltk.download('vader_lexicon')  # Lexicon for sentiment analysis
# nltk.download('averaged_perceptron_tagger')  # POS tagger

import json  # Importing for JSON handling (though not used directly in this code)
from nltk.tokenize import word_tokenize  # For tokenizing text into words
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures  # For finding bigram collocations
from nltk.sentiment import SentimentIntensityAnalyzer  # For sentiment analysis
from nltk.stem import WordNetLemmatizer  # For lemmatizing words (reducing them to base forms)
from nltk.corpus import wordnet  # WordNet corpus for lemmatization
from textblob import TextBlob  # For simpler sentiment analysis

# Import pre-trained GPT-2 model and tokenizer from Hugging Face's transformers library
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')  # GPT-2 tokenizer with left padding
model = GPT2LMHeadModel.from_pretrained(model_name)  # GPT-2 language model

# Check if a GPU is available and set the device to either 'cuda' (GPU) or 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # Send the GPT-2 model to the selected device (GPU/CPU)


def preprocess(text):
    """
    Preprocesses the input text by tokenizing, lemmatizing, and performing POS tagging.
    """
    # Tokenize the text using GPT-2's tokenizer
    tokens = tokenizer.tokenize(text)

    # Lemmatize the tokens (convert words to their base form)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    # Perform POS (Part-of-Speech) tagging on tokens
    pos_tags = nltk.pos_tag(tokens)

    return lemmatized_tokens, pos_tags


def find_collocations(text: str) -> List[Tuple[str, str]]:
    """
    Finds the top 10 bigram collocations (word pairs that frequently occur together) in the input text.
    """
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Create a BigramCollocationFinder to find word pairs (bigrams)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)

    # Find and return the top 10 bigram collocations based on Pointwise Mutual Information (PMI)
    collocations = finder.nbest(bigram_measures.pmi, 10)
    return collocations


def sentiment_analysis(text: str) -> Tuple[float, float]:
    """
    Performs sentiment analysis on the input text, returning its polarity and subjectivity.
    """
    # Use TextBlob to analyze the sentiment of the text
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment  # Polarity (-1 to 1) and subjectivity (0 to 1)
    return polarity, subjectivity


def run_nlp_pipeline(text: str) -> Dict[str, any]:
    """
    Runs the NLP pipeline: Preprocessing, collocations, and sentiment analysis on the input text.
    """
    results = {}

    # Preprocessing: Tokenization, Lemmatization, and POS tagging
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]
    pos_tags = nltk.pos_tag(tokens)
    results['preprocessing'] = {
        'lemmatized_tokens': lemmatized_tokens,
        'pos_tags': pos_tags
    }

    # Collocations: Finding the top 10 bigram collocations in the text
    tokens = word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    collocations = finder.nbest(bigram_measures.pmi, 10)
    results['collocations'] = {'top_10_collocations': collocations}

    # Sentiment analysis: Analyze the polarity and subjectivity of the text
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment
    results['sentiment_analysis'] = {'polarity': polarity, 'subjectivity': subjectivity}

    return results


def generate_summary(text, summary_length=3):
    """
    Summarizes the input text by selecting the top sentences based on sentiment scores.
    """
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Initialize a SentimentIntensityAnalyzer (VADER)
    sid = SentimentIntensityAnalyzer()
    
    # Compute the sentiment score (polarity) for each sentence
    sent_polarities = []
    for sent in sentences:
        sentiment_score = sid.polarity_scores(sent)
        polarity = sentiment_score['compound']  # Compound sentiment score
        sent_polarities.append(polarity)
    
    # Select top sentences with the highest sentiment polarities
    summary_indices = sorted(range(len(sentences)), key=lambda i: sent_polarities[i], reverse=True)[:summary_length]
    summary = [sentences[i] for i in summary_indices]
    return " ".join(summary)


def generate_dialogue(text, character_name="Alice"):
    """
    Generates a dialogue between the specified character and another speaker based on collocations in the text.
    """
    preprocessed = preprocess(text)  # Preprocess the text
    tokens = word_tokenize(text)  # Tokenize the text into words
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    collocations = finder.nbest(bigram_measures.raw_freq, 10)  # Find collocations
    
    # Create a dialogue based on the collocations
    dialogue = []
    for colloc in collocations:
        dialogue.append(f"{character_name}: What do you think about {colloc[0]}?")
        dialogue.append(f"Speaker: I think {colloc[1]}")
    
    return "\n".join(dialogue)


def generate_response(text):
    """
    Generates a text-based response using the GPT-2 model.
    """
    preprocessed_text = preprocess(text)  # Preprocess the input text

    # Encode the input text as tokens for GPT-2 model
    input_tokens = tokenizer.encode(preprocessed_text[0], add_special_tokens=True, return_tensors='pt').to(device)

    # Generate the output using GPT-2 with a maximum length of 100 tokens
    output = model.generate(input_tokens, max_length=100, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    
    # Decode the output tokens into human-readable text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text


def chatbot_response(input_text):
    """
    Generates a chatbot response using the GPT-2 model.
    """
    return generate_response(input_text)


# Example usage of the chatbot
user_input = input("User: ")
while user_input != "bye":
    response = chatbot_response(user_input)  # Generate a response
    print("ChatBot:", response)  # Print the chatbot response
    user_input = input("User: ")  # Get the next user input
