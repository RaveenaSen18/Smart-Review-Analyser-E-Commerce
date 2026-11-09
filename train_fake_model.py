import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load your dataset (replace with Reviews_clean.csv if needed)
df = pd.read_csv("Reviews.csv")  # should have a "Text" column

# Create fake labels using heuristics (temporary for training)
def label_fake(review):
    text = str(review).lower()
    if len(text.split()) < 4 or text.count("good") > 3 or "!!!" in text:
        return 1  # fake
    return 0  # genuine

df["Fake"] = df["Text"].apply(label_fake)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Fake"], test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "models/fake_review_model.pkl")

print("✅ Fake review model trained & saved at models/fake_review_model.pkl")
print("Accuracy on test:", pipeline.score(X_test, y_test))
