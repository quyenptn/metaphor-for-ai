import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # Define AI‐related keywords
    ai_keywords = [
        "ai", "chatgpt", "artificial intelligence",
        "machine", "llm", "deep learning", "machine learning",
        "natural language processing", "generative ai", "prompt"
    ]

    # Load model and pre‐compute keyword embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_embs = model.encode(ai_keywords, convert_to_numpy=True)

    # Read input CSV
    df = pd.read_csv('./predict/predict.csv', dtype={"sentence": str, "word": str, "label": int})

    # For each row, decide label1
    label1 = []
    score = []
    for _, row in tqdm(df.iterrows()):
        if row["predict"] == 0:
            label1.append(0)
            score.append(0)
        else:
            # embed the candidate word (lowercased)
            emb = model.encode(row["word"].lower(), convert_to_numpy=True)
            # compute similarities against all keywords
            sims = [
                cosine_sim(emb, kw_emb)
                for kw_emb in kw_embs
            ]
            score.append(max(sims))
            # assign 1 if any similarity exceeds threshold
            label1.append(1 if max(sims) >= 0.7 else 0)

    # 5) Write out new CSV
    df["predict1"] = label1
    df["score"] = score
    df.to_csv('./predict/predict1.csv', index=False)