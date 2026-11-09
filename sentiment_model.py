# sentiment_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_sentiment_model():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = df[['target', 'text']]
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

    vectorizer = CountVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(df['text'])
    y = df['target']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/sentiment_vectorizer.pkl')

    print("✅ Sentiment model trained and saved.")

# 🔥 THIS LINE MUST BE INCLUDED!
train_sentiment_model()
