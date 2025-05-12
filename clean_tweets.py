import csv, emoji, html, ftfy, re
import pandas as pd
from tqdm import tqdm

# Precompile regexes
BR_RE       = re.compile(r'[\r\n]+')                      # line breaks
URL_RE      = re.compile(r'https?://\S+')                 # URLs
HASHTAG_RE  = re.compile(r'#\w+')                         # hashtags
MENTION_RE  = re.compile(r'@\w+')                         # mentions
HTML_RE     = re.compile(r'&\w+;')                        # HTML entities

# Keep letters, digits, whitespace, and these punct: . , - ? ! ; :
PUNCT_RE    = re.compile(r'[^A-Za-z0-9\s\.\,\-\?\!\;\:\'\/]')

# Collapse 2+ non-word non-space chars into single ';'
MULTIPUNCT  = re.compile(r'(?:[,\.\-\?\!;](?:[^A-Za-z]+)?|[^\w\s]\s*){2,}')

# Strip leading/trailing junk except alnum or !?. 
TRIM_EDGE   = re.compile(r'^[^A-Za-z0-9]+|[^A-Za-z0-9!?\.]+$')

# Substitutions dictionary
SUBS_DICT = {
    "ain't": "is not", "aren't": "are not","can't": "can not", "cannot": "can not", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
    "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
    "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
    "you've": "you have", "rt": "retweet", "genai": "generative ai", "aiart": "ai art", "aigenerated": "ai generated",
    "aiphotography": "ai photography", "musicai": "music ai", "promts": "prompts", "&amp;": "and",
    " w ": " with ", " w/ ": " with ", " w/o ": " without ",
}

SUBS_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in SUBS_DICT) + r')\b',
    flags=re.IGNORECASE
)

# Function to clean tweets
def clean_tweet(text: str) -> str:
    t = text or ""
    t = t.lower()
    
    # collapse internal newlines & strip wrapping quotes
    t = BR_RE.sub('; ', t).strip('"')
    
    # fix encoding
    t = ftfy.fix_text(t)
    
    # apply substitutions
    t = SUBS_RE.sub(lambda m: SUBS_DICT[m.group(1).lower()], t)
    
    # unescape HTML entities
    t = html.unescape(t)
    
    # remove URLs, hashtags, mentions, emojis
    t = URL_RE.sub('', t)
    t = HASHTAG_RE.sub('', t)
    t = MENTION_RE.sub('', t)
    t = emoji.replace_emoji(t, replace='')
    
    # ensure space around allowed punctuation
    t = re.sub(r'(?<!\s)([\,\.\-\?\!\;\:\/])', r' \1 ', t)
    
    # drop all other punct
    t = PUNCT_RE.sub('', t)
    
    # drop bad single‐char tokens
    parts = t.split()
    t = " ".join(w for w in parts if not (len(w)==1 and not w.isalnum() and w not in ".,-?!;:"))
    
    # trim unwanted chars at edges
    t = TRIM_EDGE.sub('', t)
    
    # collapse odd punctuation runs to ';'
    t = MULTIPUNCT.sub('; ', t)
    
    # collapse whitespace, lowercase
    t = re.sub(r'\s+', ' ', t).strip().lower()
    
    return t


if __name__ == "__main__":
    input_path = "./shared_data/merged_tweets.csv"
    output_path = "./shared_data/cleaned_tweets.csv"
    text_col = "text"
    
    with open(input_path, 'r', encoding='latin-1', newline='') as f:
        df = pd.DataFrame(list(csv.DictReader(f)))

    cleaned = []
    for raw in tqdm(df[text_col].astype(str), desc="Cleaning"):
        cleaned.append(clean_tweet(raw))

    out = pd.DataFrame({'sentence': cleaned})
    
    # Drop empties & duplicates
    out = out.loc[out['sentence'].str.strip().ne('')].drop_duplicates('sentence')

    # Write data
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write("sentence\n")
        out.to_csv(
            f,
            index=False,
            header=False,
            quoting=csv.QUOTE_ALL,
        )