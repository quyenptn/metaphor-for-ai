import os
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define AI-related terms
ai_terms = {
    "ai", "chatgpt", "genai", "artificial intelligence",
    "machine", "llm", "deep learning", "machine learning",
    "natural language processing", "generativeai"
}

# Define intelligence-related words (expand as needed)
intelligence_words = {
    "noun": {"intelligence", "logic", "reason", "brain", "understanding"},
    "verb": {"think", "understand", "analyze", "reason", "comprehend", "learn"},
    "adj": {"smart", "intelligent", "dumb", "logical", "brilliant", "clever"}
}
all_intel_words = set().union(*intelligence_words.values())

# OPTIONAL: Save list of intelligence words for inspection
with open("debug_intelligence_words.txt", "w") as f:
    for word in sorted(all_intel_words):
        f.write(word + "\n")

# Load cleaned tweet data
try:
    df = pd.read_csv("linguistic-project/cleaned_unique_tweets.csv")
except pd.errors.ParserError:
    df = pd.read_csv("linguistic-project/cleaned_unique_tweets.csv", engine="python", on_bad_lines="skip")

# Clean text
print("Cleaning and preparing text...")
df = df.dropna(subset=["clean_text"])
df["text_clean"] = df["clean_text"].str.lower().str.replace(r"http\S+|@\S+|#\S+", "", regex=True)

# Prepare texts
all_texts = df["text_clean"].tolist()
results = []

print(f"Processing {len(all_texts)} tweets with spaCy...")

for doc in tqdm(nlp.pipe(all_texts, batch_size=50, disable=["ner"]), total=len(all_texts)):
    for token in doc:
        token_text = token.text.lower()
        token_lemma = token.lemma_.lower()
        head_text = token.head.text.lower()
        head_lemma = token.head.lemma_.lower()

        # --- Type-II: Subject-Verb-Object (SVO) ---
        if token_text in intelligence_words["verb"] and head_text in ai_terms:
            results.append({
                "lemma": token_lemma,
                "ai_term": head_text,
                "pos": token.pos_,
                "context": doc.text,
                "dep": token.dep_,
                "metaphor_type": "SVO"
            })

        elif token_text in ai_terms and token.dep_ == "nsubj":
            if head_lemma in intelligence_words["verb"]:
                results.append({
                    "lemma": head_lemma,
                    "ai_term": token_text,
                    "pos": token.head.pos_,
                    "context": doc.text,
                    "dep": token.dep_,
                    "metaphor_type": "SVO"
                })

        # -- Filter 1: AI is subject of intelligence verb (SVO pattern) --
        if token_text in ai_terms and token.dep_ == "nsubj":
            if head_lemma in intelligence_words["verb"] and head_lemma not in {"click", "read", "see", "watch", "learn"}:
                results.append({
                    "lemma": head_lemma,
                    "ai_term": token_text,
                    "pos": token.head.pos_,
                    "context": doc.text,
                    "dep": token.dep_,
                    "metaphor_type": "SVO"
                })

        # -- Filter 2: Verb takes AI as object or related (avoid imperative junk) --
        elif token_text in intelligence_words["verb"] and head_text in ai_terms:
            if token.dep_ in {"xcomp", "ROOT", "ccomp"} and token_lemma not in {"click", "learn", "read", "see", "watch"}:
                results.append({
                    "lemma": token_lemma,
                    "ai_term": head_text,
                    "pos": token.pos_,
                    "context": doc.text,
                    "dep": token.dep_,
                    "metaphor_type": "SVO"
                })

        # --- Type-I: Nominal (AI is smart) ---
        elif token_text in intelligence_words["adj"] and head_text in ai_terms and token.dep_ in {"acomp", "attr"}:
            results.append({
                "lemma": token_text,
                "ai_term": head_text,
                "pos": token.pos_,
                "context": doc.text,
                "dep": token.dep_,
                "metaphor_type": "Nominal"
            })

        # --- Type-III: Adjective-Noun (smart AI) ---
        elif token_text in intelligence_words["adj"] and head_text in ai_terms and token.dep_ == "amod":
            results.append({
                "lemma": token_text,
                "ai_term": head_text,
                "pos": token.pos_,
                "context": doc.text,
                "dep": token.dep_,
                "metaphor_type": "Adj-Noun"
            })

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv("output_intelligence_dependency_newest.csv", index=False)
print(f"\n✅ Saved {len(results)} matches to 'output_intelligence_dependency_newest.csv'")

# Frequency count
lemma_counter = Counter([r["lemma"].lower() for r in results])
type_counter = Counter([r["metaphor_type"] for r in results])

freq_lemmas = pd.DataFrame(lemma_counter.items(), columns=["lemma", "frequency"]).sort_values(by="frequency", ascending=False)
freq_lemmas.to_csv("intelligence_lemma_frequency_newest.csv", index=False)

freq_types = pd.DataFrame(type_counter.items(), columns=["metaphor_type", "count"]).sort_values(by="count", ascending=False)
freq_types.to_csv("metaphor_type_counts_newest.csv", index=False)

print("\n📊 Frequencies saved to 'intelligence_lemma_frequency_newest.csv' and 'metaphor_type_counts_newest.csv'")
