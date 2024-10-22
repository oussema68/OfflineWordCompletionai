import torch
from typing import Dict, List, Tuple
import nltk
#nltk.download('punkt')

#nltk.download('wordnet')

#nltk.download('vader_lexicon')
#nltk.download('averaged_perceptron_tagger')

import json
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob

# Load the pre-trained model and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set device (CPU or GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def preprocess(text):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)

    return lemmatized_tokens, pos_tags


def find_collocations(text: str) -> List[Tuple[str, str]]:
    # Tokenize the text
    tokens = word_tokenize(text)

    # Create BigramCollocationFinder
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)

    # Find the top 10 bigram collocations
    collocations = finder.nbest(bigram_measures.pmi, 10)
    return collocations


def sentiment_analysis(text: str) -> Tuple[float, float]:
    # Perform sentiment analysis
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment
    return polarity, subjectivity


def run_nlp_pipeline(text: str) -> Dict[str, any]:
    results = {}

    # Preprocessing
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]
    pos_tags = nltk.pos_tag(tokens)
    results['preprocessing'] = {
        'lemmatized_tokens': lemmatized_tokens,
        'pos_tags': pos_tags
    }

    # Collocations
    tokens = word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    collocations = finder.nbest(bigram_measures.pmi, 10)
    results['collocations'] = {'top_10_collocations': collocations}

    # Sentiment analysis
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment
    results['sentiment_analysis'] = {'polarity': polarity, 'subjectivity': subjectivity}

    return results


# The rest of the code remains unchanged







def generate_summary(text, summary_length=3):
    """
    Summarizes the input text by returning the top `summary_length` sentences based on the sum of the
    sentiment polarities of each sentence.
    """
    sentences = nltk.sent_tokenize(text)
    sid = SentimentIntensityAnalyzer()
    sent_polarities = []
    for sent in sentences:
        sentiment_score = sid.polarity_scores(sent)
        polarity = sentiment_score['compound']
        sent_polarities.append(polarity)
    summary_indices = sorted(range(len(sentences)), key=lambda i: sent_polarities[i], reverse=True)[:summary_length]
    summary = [sentences[i] for i in summary_indices]
    return " ".join(summary)

def generate_dialogue(text, character_name="Alice"):
    """
    Generates a dialogue between the input character name and another speaker based on the top
    collocations in the input text.
    """
    preprocessed = preprocess(text)
    tokens = word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    collocations = finder.nbest(bigram_measures.raw_freq, 10)
    dialogue = []
    for colloc in collocations:
        dialogue.append(f"{character_name}: What do you think about {colloc[0]}?")
        dialogue.append(f"Speaker: I think {colloc[1]}")
    return "\n".join(dialogue)





def generate_response(text):
    # Preprocess input text
    preprocessed_text = preprocess(text)

    # Tokenize input text
    input_tokens = tokenizer.encode(preprocessed_text[0], add_special_tokens=True, return_tensors='pt').to(device)

    # Generate output
    output = model.generate(input_tokens, max_length=100, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text

def chatbot_response(input_text):
    return generate_response(input_text)

# Example usage
user_input = input("User: ")
while user_input != "bye":
    response = chatbot_response(user_input)
    print("ChatBot:", response)
    user_input = input("User: ")

