import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# ---------------- CONFIG ----------------
DATA_PATH = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/student_stress_extended.csv"
ARTIFACT_DIR = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ---------------- CLEAN & PREPARE ----------------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

expected_features = [
    "sleep_duration", "sleep_quality", "study_hours", "screen_time",
    "academic_pressure", "exam_stress", "study_satisfaction", "procrastination",
    "exercise_frequency", "water_intake", "social_activity", "diet_quality",
    "bmi", "health_issues", "depression", "anxiety", "panic_attack",
    "specialist", "motivation_level", "concentration_level", "sentiment_score",
    "age", "cgpa"
]

# Check if stress label exists
if "stress_label" not in df.columns:
    if "stress_level" in df.columns:
        df.rename(columns={"stress_level": "stress_label"}, inplace=True)
    else:
        raise ValueError("Dataset must contain a 'stress_label' column!")

# Keep only relevant columns
cols = expected_features + ["stress_label"]
df = df[cols]

# Handle missing values
df = df.fillna(df.median())

# ---------------- FEATURE & TARGET SPLIT ----------------
X = df.drop(columns=["stress_label"])
y = df["stress_label"]

# ---------------- SCALE & BALANCE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔄 Applying SMOTE balancing...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# ---------------- TRAIN MODEL ----------------
print("🚀 Training Random Forest...")
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), multi_class="ovr")

print("\n✅ Model Performance:")
print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, AUC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ARTIFACTS ----------------
joblib.dump(model, os.path.join(ARTIFACT_DIR, "final_model.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
joblib.dump(expected_features, os.path.join(ARTIFACT_DIR, "feature_order.pkl"))

print("\n📦 Artifacts saved to:", ARTIFACT_DIR)
print("✅ Model training complete.")
