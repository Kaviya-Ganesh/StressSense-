import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

nltk.download("vader_lexicon", quiet=True)

MODEL_DIR = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/models"

ensemble = joblib.load(f"{MODEL_DIR}/ensemble.pkl")
rf = joblib.load(f"{MODEL_DIR}/rf.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
imputer = joblib.load(f"{MODEL_DIR}/imputer.pkl")
feature_order = joblib.load(f"{MODEL_DIR}/feature_order.pkl")

sid = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def sentiment_feature(text):
    text = clean_text(text)
    return sid.polarity_scores(text)["compound"]

def prepare_input(lifestyle_values, text):
    sentiment = sentiment_feature(text)
    data = lifestyle_values + [sentiment]
    df = pd.DataFrame([data], columns=feature_order)
    df = imputer.transform(df)
    df = scaler.transform(df)
    return df

st.set_page_config(page_title="StressSense", layout="wide")

# ---- HEADER ---- #
st.markdown("""
<div style="text-align:center;">
<h1>🧠 StressSense</h1>
<p style="font-size:18px;">AI-powered Stress Assessment Using Lifestyle + Emotional Expression</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("**Age**", 15, 60, 20)
    cgpa = st.number_input("**CGPA (0–10)**", 0.0, 10.0, 7.0)
    depression = st.selectbox("**Do you feel depressed?**", [0, 1])
    anxiety = st.selectbox("**Do you feel anxious?**", [0, 1])

with col2:
    panic = st.selectbox("**Do you experience panic attacks?**", [0, 1])
    treatment = st.selectbox("**Have you consulted a specialist?**", [0, 1])
    text_input = st.text_area("**Describe how you're feeling today** (1–3 sentences)")

if st.button("🔍 Analyze Stress", use_container_width=True):

    vals = [age, cgpa, depression, anxiety, panic, treatment]
    X = prepare_input(vals, text_input)

    probs = ensemble.predict_proba(X)[0]
    pred = np.argmax(probs)
    conf = max(probs)

    level = ["😌 Low Stress", "😟 Moderate Stress", "🔥 High Stress"][pred]
    color = ["#2ecc71", "#f1c40f", "#e74c3c"][pred]

    # ---- RESULT CARD ---- #
    st.markdown(f"""
    <div style="background:{color}; padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:white;">{level}</h2>
        <p style="color:white; font-size:18px;">Confidence: <b>{conf*100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ---- SUGGESTIONS ---- #
    st.write("### 💡 Recommended Well-being Action")
    if pred == 0:
        st.success("You're doing well! Maintain consistent sleep, mindful breaks, and hydration. 🌿")
    elif pred == 1:
        st.warning("Try practicing breathing exercises, short breaks while studying, and light walks. 🧘‍♀️")
    else:
        st.error("Consider talking with a counselor, reducing workload, and practicing emotional journaling. ❤️‍🩹")

    # ---- FEATURE IMPORTANCE ---- #
    st.write("---")
    st.write("### 🔍 Top Stress Influencing Factors (Model Explanation)")

    importances = rf.feature_importances_
    fi = pd.DataFrame({"feature": feature_order, "importance": importances})
    fi = fi.sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(fi["feature"], fi["importance"])
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    st.pyplot(fig)
