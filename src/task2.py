#!/usr/bin/env python
"""
topic_nmf.py
------------
Light-weight topic modelling on financial headlines using TF-IDF + Non-negative
Matrix Factorisation (NMF).  Requires only scikit-learn, pandas, nltk.

1.  Cleans & tokenises headlines
2.  Builds bigram-aware TF-IDF matrix
3.  Fits an NMF model
4.  Prints top words/phrases in each topic
"""

import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

###############################################################################
# Configuration
###############################################################################
CSV_PATH     = r"C:\Users\btulu\OneDrive - University of Florida\Class\UF_Class\Summer_2025\10Academy\week1\project\Predicting-Price-Moves-with-News-Sentiment\data\raw_analyst_ratings.csv"
N_TOPICS     = 6        # try 5–10
TOP_WORDS    = 10       # words/phrases to show per topic
BIGRAMS_ONLY = False    # True = only keep bigrams (e.g. "price_target")

###############################################################################
# Helper: minimal text cleaning
###############################################################################
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)           # strip URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

###############################################################################
# 1. Load + preprocess headlines
###############################################################################
df = pd.read_csv(CSV_PATH)
df["clean"] = df["headline"].astype(str).map(clean)

###############################################################################
# 2. TF-IDF vectorisation (include bigrams to capture phrases)
###############################################################################
ngram_range = (2, 2) if BIGRAMS_ONLY else (1, 2)        # unigrams+-bigrams
tfidf = TfidfVectorizer(
    max_df=0.6, min_df=2,
    ngram_range=ngram_range,
    norm="l2",            # cosine length-normalised
)

X = tfidf.fit_transform(df["clean"])

###############################################################################
# 3. Fit NMF
###############################################################################
nmf = NMF(
    n_components=N_TOPICS,
    init="nndsvd",
    random_state=42,
    max_iter=400,
)
W = nmf.fit_transform(X)    # document-topic matrix (unused here)
H = nmf.components_         # topic-term matrix

terms = tfidf.get_feature_names_out()

###############################################################################
# 4. Display topics
###############################################################################
print(f"\n=== Top {TOP_WORDS} words/phrases for each of {N_TOPICS} topics ===")
for topic_idx, row in enumerate(H):
    best = row.argsort()[::-1][:TOP_WORDS]
    words = " | ".join(terms[i] for i in best)
    print(f"Topic {topic_idx+1}: {words}")
