🌿 StressSense – Smart Student Stress Detection System

OVERVIEW:

  StressSense is an intelligent machine learning–based web app designed to predict a student’s stress level using lifestyle habits and emotional indicators.
It combines both survey-based behavioral features and text sentiment analysis to provide a holistic mental health insight.

FEATURES:

  Predicts Low / Moderate / High Stress levels
  Uses 25+ lifestyle and psychological indicators
  Analyzes text sentiment (VADER Sentiment Analyzer)
  Balanced training using SMOTE for fair classification
  Beautiful, animated Streamlit UI
  Model explainability with top stress factor visualization

TECH STACK:

  Python, Pandas, NumPy, Scikit-learn, XGBoost
  NLTK (VADER) – for sentiment scoring

PROJECT STRUCTURE:

StressSense/
│
├── data/
│   ├── student_stress_extended.csv
│   └── text_features.csv
│
├── src/
│   ├── preprocess_lifestyle.py
│   ├── preprocess_text.py
│   ├── train.py
│   └── generate_student_stress_extended.py
│
├── artifacts/
│   ├── final_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
└── app/
    └── app.py

HOW TO RUN:

1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train model (optional if artifacts exist)
python src/train.py
3️⃣ Run the app
streamlit run app/app.py

DATASET:

  Student Lifestyle Survey Data (Extended 25-Feature Version)
  Includes factors like sleep, study hours, screen time, exercise, anxiety, depression, motivation, and sentiment.

FUTURE ENHANCEMENTS:

  Integration with real-time emotion tracking
  Personalized stress management recommendations
  Mobile version with journaling & daily logs


Streamlit – interactive web app

Matplotlib – feature importance plots
