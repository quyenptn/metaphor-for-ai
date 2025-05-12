# metaphor-for-ai

## Installation

1. Install `python@3.11`

2. Create a Python virtual environment:
    ```
    python -m venv .venv

    source .venv/bin/activate
    ```

3. Install dependencies
    ```
    pip install -r requirements.txt

    python -m spacy download en_core_web_sm
    ```

## Clean Tweets data
    ```
    python clean_tweets.py
    ```

## Top-down Approach
    ```
    cd top-down

    python parsing_emotion.py

    python parsing_metaphor_intelligence.py
    ```

## Bottom-up Approach
    ```
    cd bottom-up

    python data_preprocessing.py

    python predict_metaphor.py

    python predict_ai.py
    ```
