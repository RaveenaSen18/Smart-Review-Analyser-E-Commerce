from textblob import TextBlob
import joblib
import os

# Load the trained fake review ML model
model_path = "models/fake_review_model.pkl"
fake_model = joblib.load(model_path) if os.path.exists(model_path) else None

# --- Sentiment Analyzer ---
def get_sentiment(review: str):
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- Fake Review Detector (ML + rules for explanation) ---
def is_fake_review(review: str):
    reasons = []

    # ML prediction (if model exists)
    if fake_model:
        pred = fake_model.predict([review])[0]
    else:
        pred = 0  # fallback genuine if no model

    review_lower = review.lower()

    # Heuristic explanations (WHY it may be fake)
    generic_phrases = ["good product", "nice", "amazing", "excellent", "best product"]
    if any(phrase in review_lower for phrase in generic_phrases):
        reasons.append("Too short or generic (not detailed).")

    promo_words = ["highly recommend", "must buy", "life changing", "everyone should buy"]
    if any(word in review_lower for word in promo_words):
        reasons.append("Over-promotional language detected.")

    if review.count("!") > 3 or review.isupper():
        reasons.append("Excessive exclamations / ALL CAPS detected.")

    sentiment = get_sentiment(review)
    if ("fake" in review_lower or "scam" in review_lower) and sentiment == "Positive":
        reasons.append("Contradiction between sentiment and words (mentions scam/fake but marked positive).")

    # Final decision
    if pred == 1 or reasons:
        return True, "; ".join(reasons) if reasons else "ML model flagged this as suspicious."
    else:
        return False, "Looks like a genuine review."

# --- Trust Score ---
def get_trust_score(is_fake: bool):
    return 40 if is_fake else 90
