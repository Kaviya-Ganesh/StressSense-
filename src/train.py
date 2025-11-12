import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, classification_report
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Paths
LIFESTYLE = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts/lifestyle_features.csv"
TEXT = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts/text_features.csv"
MODEL_DIR = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/models"

def parse_cgpa(val):
    """
    Convert CGPA strings to numeric:
    - '3.00 - 3.49' -> midpoint
    - '3.8' -> 3.8
    - Otherwise NaN
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # find two numbers (range)
    m = re.findall(r"\d+\.?\d*", s)
    if len(m) == 2:
        a, b = float(m[0]), float(m[1])
        return (a + b) / 2.0
    if len(m) == 1:
        return float(m[0])
    return np.nan

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("🔹 Loading datasets...")
    df_life = pd.read_csv(LIFESTYLE)
    df_text = pd.read_csv(TEXT)

    print("Lifestyle shape:", df_life.shape)
    print("Text shape:", df_text.shape)

    # --- Clean/convert lifestyle to numeric ---
    if "cgpa" in df_life.columns:
        df_life["cgpa"] = df_life["cgpa"].apply(parse_cgpa)

    # Keep only columns we expect to be numeric (present in this dataset)
    candidate_feats = ["age", "cgpa", "depression", "anxiety", "panic_attack", "treatment"]
    present_feats = [c for c in candidate_feats if c in df_life.columns]

    if "stress_label" not in df_life.columns:
        raise ValueError("stress_label column not found in lifestyle_features.csv")

    # Align by row count with text features
    # We only need sentiment_score from text features
    if "sentiment_score" not in df_text.columns:
        raise ValueError("sentiment_score column not found in text_features.csv. "
                         "Re-run preprocess_text.py.")

    min_len = min(len(df_life), len(df_text))
    df_life = df_life.iloc[:min_len].reset_index(drop=True)
    df_text = df_text.iloc[:min_len].reset_index(drop=True)

    # Build final feature frame
    X = df_life[present_feats].copy()
    X["sentiment_score"] = pd.to_numeric(df_text["sentiment_score"], errors="coerce")

    y = df_life["stress_label"].astype(int)

    # Coerce all to numeric and impute
    X = X.apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Check class distribution
    classes, counts = np.unique(y, return_counts=True)
    print(f"📦 Class distribution before SMOTE: {dict(zip(classes, counts))}")

    # If only 1 class present, raise error with guidance
    if len(classes) < 2:
        raise ValueError("Only one class present in stress_label. "
                         "Cannot train a classifier. Check your preprocessing.")

    # Balance with SMOTE (needs at least 2 classes)
    print("🔄 Applying SMOTE balancing...")
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_bal)),
            eval_metric="mlogloss",
            learning_rate=0.08,
            max_depth=5,
            n_estimators=250,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    }

    # Train + evaluate
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_test, y_pred)
        try:
            # multi-class AUC with OVR using predict_proba
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            auc = np.nan

        print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, AUC: {auc:.3f}")

    # Ensemble
    print("\n🤝 Building Ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models["Logistic Regression"]),
            ('rf', models["Random Forest"]),
            ('xgb', models["XGBoost"])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(models["Logistic Regression"], os.path.join(MODEL_DIR, "logreg.pkl"))
    joblib.dump(models["Random Forest"], os.path.join(MODEL_DIR, "rf.pkl"))
    joblib.dump(models["XGBoost"], os.path.join(MODEL_DIR, "xgb.pkl"))
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
    joblib.dump(present_feats + ["sentiment_score"], os.path.join(MODEL_DIR, "feature_order.pkl"))

    print("\n✅ All models and preprocessing objects saved to:", MODEL_DIR)

if __name__ == "__main__":
    main()
