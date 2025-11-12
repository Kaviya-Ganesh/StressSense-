import streamlit as st
import joblib
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objects as go

# ----------------- LOAD MODEL & SCALER -----------------
model = joblib.load("artifacts/final_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
feature_order = joblib.load("artifacts/feature_order.pkl")

nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="StressSense", page_icon="🌈", layout="wide")

st.markdown("""
    <style>
        body { background-color: #1e1e1e; color: #f1f1f1; }
        .title {
            text-align: center; 
            font-size: 3em; 
            font-weight: 700; 
            color: #8e97fd; 
            font-family: 'Poppins', sans-serif;
            animation: fadeIn 1.5s ease-in;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        h3 { color: #d1d1ff; font-family: 'Poppins'; }
        .stButton>button {
            background-color: #8e97fd; 
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 16px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #b5aafc;
            color: #000;
            transform: scale(1.03);
        }
        .result-box {
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            font-size: 1.3em;
            margin-top: 20px;
        }
        .low { background-color: #b8f2e6; color: #055160; }
        .moderate { background-color: #fff3cd; color: #664d03; }
        .high { background-color: #f5c6cb; color: #58151c; }
        .suggest-box {
            border-radius: 10px;
            padding: 15px;
            font-size: 1.1em;
            background-color: #e0e7ff;
            color: #2e2e2e;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- TITLE -----------------
st.markdown("<div class='title'>🌈 StressSense – Balance Your Mind</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;'>Lifestyle + Emotions = Mindful Well-being</p>", unsafe_allow_html=True)

# ----------------- USER INPUTS -----------------
st.markdown("### 🧘‍♀️ Enter your details below:")

st.subheader("🛏️ Sleep Habits")
col1, col2 = st.columns(2)
with col1:
    sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
    sleep_quality = st.selectbox("Sleep Quality (1=Poor, 5=Excellent)", [1, 2, 3, 4, 5])
with col2:
    study_hours = st.slider("Average Study Hours (per day)", 0, 15, 5)
    screen_time = st.slider("Screen Time (hours/day)", 0, 15, 6)

st.subheader("📚 Study & Academic Pressure")
col3, col4 = st.columns(2)
with col3:
    academic_pressure = st.selectbox("Academic Pressure (1=Low, 5=High)", [1, 2, 3, 4, 5])
    exam_stress = st.selectbox("Exam Stress (1=Low, 5=High)", [1, 2, 3, 4, 5])
with col4:
    study_satisfaction = st.selectbox("Satisfaction with Study Routine (1=Low, 5=High)", [1, 2, 3, 4, 5])
    procrastination = st.selectbox("Do you procrastinate often?", [0, 1])

st.subheader("🧍‍♀️ Health & Lifestyle")
col5, col6 = st.columns(2)
with col5:
    exercise_frequency = st.selectbox("Exercise Frequency (per week)", [0, 1, 2, 3, 4, 5, 6, 7])
    water_intake = st.slider("Water Intake (liters/day)", 0.0, 5.0, 2.0)
    social_activity = st.selectbox("Do you socialize with friends/family?", [0, 1])
with col6:
    diet_quality = st.selectbox("Diet Quality (1=Poor, 5=Excellent)", [1, 2, 3, 4, 5])
    bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.0)
    health_issues = st.selectbox("Any current health issues?", [0, 1])

st.subheader("💭 Emotional & Mental State")
col7, col8 = st.columns(2)
with col7:
    depression = st.selectbox("Do you feel depressed?", [0, 1])
    anxiety = st.selectbox("Do you feel anxious?", [0, 1])
    panic_attack = st.selectbox("Do you experience panic attacks?", [0, 1])
with col8:
    specialist = st.selectbox("Have you consulted a specialist?", [0, 1])
    motivation_level = st.selectbox("Motivation Level (1=Low, 5=High)", [1, 2, 3, 4, 5])
    concentration_level = st.selectbox("Concentration Ability (1=Low, 5=High)", [1, 2, 3, 4, 5])

st.subheader("✍️ Describe how you’re feeling today (1–3 sentences):")
text_input = st.text_area("Example: I felt relaxed after studying, went out for a walk, and had a peaceful evening.")

# ----------------- ANALYSIS -----------------
if st.button("🌸 Analyze Stress"):
    sentiment_score = sid.polarity_scores(text_input)["compound"]

    X = pd.DataFrame([{
        "sleep_duration": sleep_duration,
        "sleep_quality": sleep_quality,
        "study_hours": study_hours,
        "screen_time": screen_time,
        "academic_pressure": academic_pressure,
        "exam_stress": exam_stress,
        "study_satisfaction": study_satisfaction,
        "procrastination": procrastination,
        "exercise_frequency": exercise_frequency,
        "water_intake": water_intake,
        "social_activity": social_activity,
        "diet_quality": diet_quality,
        "bmi": bmi,
        "health_issues": health_issues,
        "depression": depression,
        "anxiety": anxiety,
        "panic_attack": panic_attack,
        "specialist": specialist,
        "motivation_level": motivation_level,
        "concentration_level": concentration_level,
        "sentiment_score": sentiment_score,
        "age": 20,
        "cgpa": 7.5,
        "stress_label": 0  # dummy alignment
    }])

    # Align features
    expected_features = feature_order
    for c in expected_features:
        if c not in X.columns:
            X[c] = 0
    X = X[expected_features]

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[0]
    pred = np.argmax(probs)
    conf = np.max(probs) * 100

    stress_labels = ["Low Stress", "Moderate Stress", "High Stress"]
    stress_colors = ["#b8f2e6", "#fff3cd", "#f5c6cb"]
    stress_emojis = ["😊", "😟", "😣"]

    st.markdown(f"""
        <div class='result-box {["low","moderate","high"][pred]}'>
            {stress_emojis[pred]} <b>{stress_labels[pred]}</b><br>
            Confidence: {conf:.1f}% | Sentiment Score: {sentiment_score:.2f}
        </div>
    """, unsafe_allow_html=True)

    if pred == 0:
        msg = "You're in a good zone 🌿 Keep your balanced habits and mindful routine."
    elif pred == 1:
        msg = "Take breaks, meditate, and focus on your breathing to center yourself 🌸."
    else:
        msg = "You may be overwhelmed — please reach out to a counselor or someone you trust ❤️."
    
    st.markdown(f"<div class='suggest-box'>💡 <b>Recommended Well-being Tip:</b> {msg}</div>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Stress Level"},
        gauge={
            'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': ['Low', 'Moderate', 'High']},
            'bar': {'color': stress_colors[pred]},
            'steps': [
                {'range': [0, 1], 'color': "#b8f2e6"},
                {'range': [1, 2], 'color': "#fff3cd"},
                {'range': [2, 3], 'color': "#f5c6cb"},
            ]
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
