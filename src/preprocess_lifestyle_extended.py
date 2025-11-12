import os
import pandas as pd
from sklearn.impute import SimpleImputer

IN = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/lifestyle/Student_Stress_Extended.csv"
OUT = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/artifacts/lifestyle_features.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(IN)

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle missing numeric values
    num_cols = df.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Normalize stress label (categorical)
    if "stress_level" in df.columns:
        df["stress_label"] = pd.cut(df["stress_level"], bins=[-1,3,6,10],
                                    labels=[0,1,2]).astype(int)
    else:
        raise ValueError("No stress_level column found!")

    df.to_csv(OUT, index=False)
    print("✅ Extended lifestyle features saved:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()
