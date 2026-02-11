# ===========================
# Complete Upgraded Fake News Detection Streamlit App
# Steps 1–4 included
# ===========================

import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib

# ===========================
# Load model & TF-IDF
# ===========================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ===========================
# Function to explain predictions
# ===========================
def explain_prediction(text, model, vectorizer, top_n=10):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    indices = vec.nonzero()[1]
    word_scores = {feature_names[i]: coef[i] * vec[0,i] for i in indices}
    top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return top_words

# ===========================
# Streamlit interface
# ===========================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection System")
st.write("Paste a news article or upload a CSV file to see if the news is Real or Fake.")

# ===========================
# User input - text area
# ===========================
user_input = st.text_area("Enter a news article here:", height=200)

# Slider for number of top words
top_n = st.slider("Number of top influencing words to display:", 5, 20, 10)

# ===========================
# File upload for batch predictions
# ===========================
uploaded_file = st.file_uploader("Or upload a CSV with column 'text'", type=["csv"])

# ===========================
# Prediction function
# ===========================
def predict_news(texts):
    X_input = tfidf.transform(texts)
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)
    return preds, probs

# ===========================
# If user inputs text manually
# ===========================
if st.button("Predict Article") and user_input.strip():
    preds, probs = predict_news([user_input])
    pred = preds[0]
    prob = probs[0][pred]

    # Prediction + Confidence
    result_text = "REAL" if pred == 1 else "FAKE"
    st.subheader(f"Prediction: {result_text}")
    st.write(f"Confidence: {prob*100:.2f}%")
    st.progress(int(prob*100))

    if pred == 1:
        st.success("The model is confident this news is REAL.")
    else:
        st.error("The model is confident this news is FAKE.")

    # Explain prediction - top words
    reasons = explain_prediction(user_input, model, tfidf, top_n)
    df_words = pd.DataFrame(reasons, columns=["Word", "Score"])
    df_words["Color"] = df_words["Score"].apply(lambda x: "green" if x > 0 else "red")

    # Colored bar chart
    fig_bar = px.bar(
        df_words,
        x="Word",
        y="Score",
        color="Color",
        color_discrete_map={"green": "green", "red": "red"},
        title="Top Influencing Words",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Interactive probability chart (Step 4)
    labels = ["REAL", "FAKE"]
    values = [probs[0][1], probs[0][0]]  # REAL prob first
    fig_prob = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["green", "red"],
        text=[f"{v*100:.2f}%" for v in values],
        textposition="auto"
    ))
    fig_prob.update_layout(title_text="Prediction Probabilities", yaxis=dict(range=[0,1]))
    st.plotly_chart(fig_prob, use_container_width=True)

    # Reasoning
    if pred == 0:
        st.info("Reason: These keywords indicate potential misinformation or fake content.")
    else:
        st.success("Reason: These keywords indicate credible or real content.")

# ===========================
# If user uploads CSV file
# ===========================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        st.write("Predicting news articles in uploaded CSV...")
        preds, probs = predict_news(df['text'].astype(str))
        df['prediction'] = ["REAL" if p==1 else "FAKE" for p in preds]
        df['confidence'] = [round(p.max()*100,2) for p in probs]

        st.dataframe(df.head(10))

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

        # Interactive probability chart for batch (first 10)
        fig_batch = go.Figure()
        for i in range(min(10, len(df))):
            fig_batch.add_trace(go.Bar(
                x=["REAL","FAKE"],
                y=[probs[i][1], probs[i][0]],
                name=f"Article {i+1}",
                text=[f"{probs[i][1]*100:.2f}%","{:.2f}%".format(probs[i][0]*100)],
                textposition="auto"
            ))
        fig_batch.update_layout(
            barmode='group',
            title="Prediction Probabilities for First 10 Articles"
        )

        st.plotly_chart(fig_batch, use_container_width=True)

