import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------- HELPER FUNCTIONS ----------------
def fake_review_with_reason(review):
    """Return whether a review is fake and explain why"""
    reasons = []
    review_lower = review.lower()
    fake = False
    if len(review.split()) < 3:
        reasons.append("Too short")
    if any(phrase in review_lower for phrase in ["amazing", "best purchase", "perfect", "highly recommend"]):
        reasons.append("Contains generic promotional phrases")
        fake = True
    if any(word in review_lower for word in ["terrible", "worst", "broke"]):
        fake = True
        reasons.append("Contains negative fake keywords")
    explanation = ", ".join(reasons) if fake else "Appears genuine"
    return fake, explanation

def get_trust_score(fake):
    """Simple trust score"""
    return 30 if fake else 80

def classify_tp_tn_fp_fn(actual, predicted):
    """Return TP/TN/FP/FN"""
    actual = actual.lower()
    predicted = predicted.lower()
    if actual == "fake" and predicted == "fake":
        return "TP"
    elif actual == "genuine" and predicted == "genuine":
        return "TN"
    elif actual == "genuine" and predicted == "fake":
        return "FP"
    elif actual == "fake" and predicted == "genuine":
        return "FN"
    else:
        return "Unknown"

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Confusion Matrix 10 Reviews", layout="wide")
st.title("🛠 Confusion Matrix Analyzer for 10 Reviews")

# ---------------- PREDEFINED 10 REVIEWS ----------------
reviews = [
    "This product is amazing, best purchase ever!",
    "Terrible quality, broke on first use!",
    "I liked the product, works perfectly as expected.",
    "Good value for money, very satisfied with it.",
    "Perfect item, highly recommend to everyone!",
    "Worst purchase, do not buy this product.",
    "Decent quality, but not great for the price.",
    "Excellent service and the product meets expectations.",
    "Amazing quality, I will buy again!",
    "Not bad, works fine but could be better."
]

actual_labels = [
    "Fake", "Fake", "Genuine", "Genuine", "Fake",
    "Fake", "Genuine", "Genuine", "Fake", "Genuine"
]

# ---------------- PROCESS REVIEWS ----------------
results = []
y_true = []
y_pred = []

for review, actual_label in zip(reviews, actual_labels):
    fake, reason = fake_review_with_reason(review)
    trust = get_trust_score(fake)
    predicted_label = "Fake" if fake else "Genuine"
    metric = classify_tp_tn_fp_fn(actual_label, predicted_label)

    results.append({
        "Review": review,
        "Actual Label": actual_label,
        "Predicted Label": predicted_label,
        "Explanation": reason,
        "Trust Score": trust,
        "Metric": metric
    })

    y_true.append("Fake" if actual_label.lower() == "fake" else "Genuine")
    y_pred.append(predicted_label)

df_results = pd.DataFrame(results)

# ---------------- SHOW RESULTS ----------------
st.markdown("### 📋 Review Predictions with TP/TN/FP/FN")
st.dataframe(df_results)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred, labels=["Fake", "Genuine"])
tp, fn = cm[0,0], cm[0,1]
fp, tn = cm[1,0], cm[1,1]

st.markdown("### 🎯 Confusion Matrix Counts")
st.write(f"**True Positives (TP):** {tp}")
st.write(f"**True Negatives (TN):** {tn}")
st.write(f"**False Positives (FP):** {fp}")
st.write(f"**False Negatives (FN):** {fn}")

# ---------------- COLORED CONFUSION MATRIX ----------------
st.markdown("### 🖼️ Confusion Matrix Heatmap")
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Fake","Genuine"], yticklabels=["Fake","Genuine"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
st.pyplot(plt)

# ---------------- DOWNLOAD CSV ----------------
st.download_button(
    "⬇️ Download Predictions CSV",
    data=df_results.to_csv(index=False).encode('utf-8'),
    file_name="10_reviews_confusion_matrix.csv"
)
