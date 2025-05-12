import re, sys
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# Set seed for consistent results
DetectorFactory.seed = 0

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Turn on tqdm pandas integration
tqdm.pandas()


def get_token(x):
    return ' '.join([token.lower for token in nlp(x)])


def get_pos(x):
    return ' '.join([token.pos_ for token in nlp(x)])


def get_tag(x):
    return ' '.join([token.tag_ for token in nlp(x)])


def is_english(text: str) -> bool:
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def filter_english(input_csv: str, output_csv: str, text_col: str = 'sentence'):
    # Load
    df = pd.read_csv(input_csv, dtype=str)

    # Ensure column exists
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")

    # Run detection
    mask = df[text_col].fillna('').progress_apply(is_english)

    # Filter and save
    df[mask].to_csv(output_csv, index=False, header=True)


def get_features_fast(input_csv: str, output_csv: str, batch_size=1000):
    # df = pd.read_csv(input_csv, encoding='latin-1')
    df = pd.read_csv(input_csv, encoding='latin-1', dtype=str).fillna("")
    texts = df['sentence'].fillna('').tolist()

    # preallocate lists
    word_list = [None] * len(texts)
    pos_list = [None] * len(texts)
    tag_list = [None] * len(texts)

    # single pass through spaCy
    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=batch_size, disable=["ner", "parser"]),
                            total=len(texts))):
        word_list[i] = [token.lower_ for token in doc]
        pos_list[i] = [tok.pos_ for tok in doc]
        tag_list[i] = [tok.tag_ for tok in doc]

    df['word'] = word_list
    df['pos'] = pos_list
    df['tag'] = tag_list

    # explode all three in lockstep
    df = df.explode(['word', 'pos', 'tag'])
    sentence = np.array(df['sentence']).tolist()
    word = np.array(df['word']).tolist()
    
    local = []
    for i in tqdm(range(len(sentence))):
        flag = 0
        w = word[i]
        slice = [part for part in re.split(r'[.,?!;]+', sentence[i]) if part != '']
        
        for j in range(len(slice)):
            if str(w) in slice[j] and flag == 0:
                local.extend([slice[j]])
                flag = 1
                break
        if flag == 0:
            local.extend([''])
    df['local'] = local
    
    df.to_csv(output_csv, index=False,
                    columns=['sentence','word','pos','tag', 'local'])


if __name__ == '__main__':
    tweet_data_csv = '../shared_data/cleaned_tweets.csv'
    filtered_csv = '../shared_data/en_filtered_tweets.csv'
    preprocessed_csv = './data/tweets_preprocessed.csv'
    
    print("Starting preprocessing...")
    print("Filtering English tweets...")
    filter_english(tweet_data_csv, filtered_csv)
    print("Extracting features...")
    get_features_fast(filtered_csv, preprocessed_csv)
    print("Preprocessing completed.")
