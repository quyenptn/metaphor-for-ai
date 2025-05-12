# Sentiment Analysis Script for Detected AI Metaphor Contexts
import pandas as pd
from textblob import TextBlob

# Load the CSV file containing metaphor contexts
input_file = "output_intelligence_dependency_newest.csv"
df = pd.read_csv(input_file)

# Check if 'context' column exists
if "context" not in df.columns:
    raise ValueError("The file does not contain a 'context' column.")

# Define a function to compute polarity and sentiment label
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return polarity, "positive"
    elif polarity < -0.1:
        return polarity, "negative"
    else:
        return polarity, "neutral"

# Apply sentiment analysis
df[["sentiment_polarity", "sentiment_label"]] = df["context"].apply(lambda x: pd.Series(get_sentiment(x)))

# Save results to a new CSV file
output_file = "ai_metaphor_intelligence_sentiment.csv"
df.to_csv(output_file, index=False)

print(f"✅ Sentiment analysis completed. Results saved to '{output_file}'")
