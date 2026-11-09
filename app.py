import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import get_sentiment, is_fake_review, get_trust_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🛍️ Smart Review Analyzer", layout="wide", initial_sidebar_state="expanded")

# ---------------- CUSTOM BLUE-PINK THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #89f7fe, #fbd3e9);
    font-family: "Segoe UI", sans-serif;
    color: #2c2c2c;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #4facfe, #fbc2eb);
    color: white;
}
h1,h2,h3,h4,h5,h6 {
    color: #1f3c88;
    font-weight: 700;
}
div.stButton > button {
    background: linear-gradient(90deg, #4facfe, #fbc2eb);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #43e97b, #38f9d7);
}
.metric-card {
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    font-weight:bold;
}
.stPlotlyChart, .stVegaLiteChart {
    border-radius: 15px;
    padding: 10px;
    background: #ffffffcc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
textarea {
    background-color: #ffffffdd !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;font-size:42px'>🛍️ Smart Product Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#333;'>Detect fake reviews, analyze sentiment & visualize trust scores</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Filters & Options")
show_fake = st.sidebar.checkbox("Show Only Fake Reviews", value=False)
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"])

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Bulk Dashboard", "📝 Real-Time Analyzer"])

# ---------------- HELPER FUNCTIONS ----------------
def get_sentiment_simple(review):
    """Return sentiment of the review using utils.get_sentiment"""
    return get_sentiment(review)

def fake_review_with_reason(review):
    """Return whether a review is fake and explain why"""
    fake, _ = is_fake_review(review)
    reasons = []
    if len(review.split()) < 3:
        reasons.append("Too short")
    if any(phrase in review.lower() for phrase in ["amazing product","best purchase","perfect"]):
        reasons.append("Contains generic promotional phrases")
    if fake:
        explanation = ", ".join(reasons) if reasons else "Model flagged as fake"
    else:
        explanation = "Appears genuine" + (f" | {', '.join(reasons)}" if reasons else "")
    return fake, explanation

# ---------------- BULK DASHBOARD ----------------
with tab1:
    uploaded_file = st.file_uploader("📂 Upload CSV (`Reviews.csv`)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        review_col = next((col for col in df.columns if col.strip().lower() in ["review","text","comment"]), None)

        if not review_col:
            st.error("❌ Review column not found. Must have 'Review', 'Text', or 'Comment'.")
        else:
            # Process reviews
            results = []
            for review in df[review_col]:
                sentiment = get_sentiment_simple(review)
                fake, reason = fake_review_with_reason(review)
                trust = get_trust_score(fake)
                results.append({
                    "Review": review,
                    "Sentiment": sentiment,
                    "Fake?": "Yes" if fake else "No",
                    "Explanation": reason,
                    "Trust Score": trust
                })

            result_df = pd.DataFrame(results)

            # Apply filters
            if show_fake:
                result_df = result_df[result_df["Fake?"]=="Yes"]
            if sentiment_filter != "All":
                result_df = result_df[result_df["Sentiment"]==sentiment_filter]

            # ---------------- METRICS ----------------
            total = len(result_df)
            fake_count = (result_df["Fake?"]=="Yes").sum()
            genuine_count = total - fake_count

            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card' style='background:#4facfe;'><h3>Total Reviews</h3><h2>{total}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card' style='background:#43e97b;'><h3>Genuine Reviews</h3><h2>{genuine_count}</h2></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card' style='background:#ff6a88;'><h3>Fake Reviews</h3><h2>{fake_count}</h2></div>", unsafe_allow_html=True)

            st.markdown("---")

            # ---------------- CHARTS ----------------
            col1,col2 = st.columns(2)

            with col1:
                fig_sent = px.histogram(result_df, x="Sentiment", color="Sentiment", text_auto=True,
                                        color_discrete_map={"Positive":"green","Negative":"red","Neutral":"orange"},
                                        title="Sentiment Distribution")
                st.plotly_chart(fig_sent, use_container_width=True)

                fig_fake = px.pie(result_df, names="Fake?", hole=0.4, color="Fake?",
                                  color_discrete_map={"Yes":"#ff6a88","No":"#43e97b"}, title="Fake vs Genuine")
                st.plotly_chart(fig_fake, use_container_width=True)

            with col2:
                fig_trust = px.histogram(result_df, x="Trust Score", nbins=20, color="Trust Score",
                                         title="Trust Score Distribution")
                st.plotly_chart(fig_trust, use_container_width=True)

                for label, color_map in [("Yes","Reds"), ("No","Greens")]:
                    text = " ".join(result_df[result_df["Fake?"]==label]["Review"].astype(str))
                    if text.strip():
                        wc = WordCloud(width=400,height=200,background_color='white',colormap=color_map,collocations=False).generate(text)
                        plt.figure(figsize=(6,3))
                        plt.imshow(wc, interpolation='bilinear')
                        plt.axis('off')
                        st.markdown(f"**{'Fake' if label=='Yes' else 'Genuine'} Reviews Word Cloud**")
                        st.pyplot(plt)

            # ---------------- CONFUSION MATRIX ----------------
            st.markdown("---")
            st.markdown("### 🎯 Confusion Matrix and Performance Metrics")

            actual_col = next((col for col in df.columns if col.strip().lower() in ["label", "actual", "is_fake", "groundtruth"]), None)

            if actual_col:
                y_true = df[actual_col].astype(str).str.strip().str.lower().map(lambda x: "yes" if "fake" in x else "no")
                y_pred = result_df["Fake?"].str.lower()

                cm = confusion_matrix(y_true, y_pred, labels=["yes","no"])
                tn, fp, fn, tp = cm.ravel()

                st.write(f"**True Positives (TP):** {tp} — Correctly predicted fake reviews")
                st.write(f"**True Negatives (TN):** {tn} — Correctly predicted genuine reviews")
                st.write(f"**False Positives (FP):** {fp} — Genuine reviews incorrectly flagged as fake")
                st.write(f"**False Negatives (FN):** {fn} — Fake reviews missed by the model")

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Genuine"])
                fig_cm, ax = plt.subplots(figsize=(5,4))
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                plt.title("Confusion Matrix")
                st.pyplot(fig_cm)
            else:
                st.info("ℹ️ Add a 'Label' column (with values like 'Fake' or 'Genuine') to compute the confusion matrix.")

            st.markdown("---")
            st.download_button("⬇️ Download Processed CSV", data=result_df.to_csv(index=False).encode('utf-8'), file_name="processed_reviews.csv")

# ---------------- REAL-TIME ANALYZER ----------------
with tab2:
    st.markdown("### ✍️ Enter a review and click **Analyze** to see insights")
    review_input = st.text_area("Type or paste a review here:")
    analyze_button = st.button("🔍 Analyze Review")

    if analyze_button and review_input.strip():
        sentiment = get_sentiment_simple(review_input)
        fake, reason = fake_review_with_reason(review_input)
        trust = get_trust_score(fake)

        emojis = {"positive":"✅😊","negative":"❌😡","neutral":"⚪😐"}

        st.markdown(f"<div class='metric-card' style='background:#4facfe;'><h2 style='color:white'>{sentiment} {emojis.get(sentiment.lower())}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card' style='background:{'#ff6a88' if fake else '#43e97b'};'><h3>Fake Review?: {'❌ Yes' if fake else '✅ No'}</h3><p>{reason}</p></div>", unsafe_allow_html=True)

        # Gauge for trust score
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=trust,
            delta={'reference':50,'increasing':{'color':'green'},'decreasing':{'color':'red'}},
            title={'text':'Trust Score'},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'green' if trust>70 else 'orange' if trust>40 else 'red'},
                'steps':[
                    {'range':[0,40],'color':'#ff6a88'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'#43e97b'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Detailed explanation
        if fake:
            st.warning(f"⚠️ This review seems suspicious. Reason: {reason}")
        else:
            st.success(f"✅ This review seems genuine. Reason: {reason}")
