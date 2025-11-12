import os
import pandas as pd
import numpy as np

IN = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/lifestyle/Student Mental health.csv"
OUT = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts/lifestyle_features.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(IN)

    df.columns = [c.strip().lower() for c in df.columns]

    # Rename to simpler names
    rename = {
        "choose your gender": "gender",
        "age": "age",
        "what is your course?": "course",
        "your current year of study": "year",
        "what is your cgpa?": "cgpa",
        "do you have depression?": "depression",
        "do you have anxiety?": "anxiety",
        "do you have panic attack?": "panic_attack",
        "did you seek any specialist for a treatment?": "treatment"
    }

    df.rename(columns=rename, inplace=True)

    # Convert Yes/No to 1/0
    for col in ["depression","anxiety","panic_attack","treatment"]:
        df[col] = df[col].replace({"Yes":1, "No":0, "yes":1, "no":0}).astype(int)

    # Create composite stress score
    df["stress_level"] = df["depression"] + df["anxiety"] + df["panic_attack"]

    # Convert to classes
    # 0 = Low, 1 = Moderate, 2 = High
    df["stress_label"] = df["stress_level"].map({0:0, 1:1, 2:2, 3:2})

    # Keep useful numeric features only
    final = df[["age","cgpa","depression","anxiety","panic_attack","treatment","stress_label"]]
    final.to_csv(OUT, index=False)

    print("✅ Saved lifestyle_features.csv:", final.shape)
    print(final.head())

if __name__ == "__main__":
    main()
