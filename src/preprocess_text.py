import os
import pandas as pd
import re

IN1 = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/text/dreaddit-train.csv"
IN2 = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/text/dreaddit-test.csv"
OUT = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts/text_features.csv"

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # ✅ Load CSV instead of Excel
    df_train = pd.read_csv(IN1)
    df_test = pd.read_csv(IN2)

    # Combine datasets
    df = pd.concat([df_train, df_test], ignore_index=True)

    df.columns = [c.lower() for c in df.columns]

    # Detect text column
    text_col = "text"
    for col in df.columns:
        if "text" in col:
            text_col = col
            break

    print(f"✅ Using text column:", text_col)

    # Clean text
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    # Sentiment
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df[text_col].apply(lambda x: sid.polarity_scores(x)["compound"])

    df.to_csv(OUT, index=False)
    print("✅ text_features saved at:", OUT)
    print(df.head())

if __name__ == "__main__":
    main()
