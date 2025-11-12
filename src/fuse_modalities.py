"""
Fuse modalities (wearable + lifestyle + text) → artifacts/fused_dataset.csv
- Gracefully handles missing wearable/text (uses what’s available)
- Aligns row counts by truncation
"""

import os, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

ART = r"C:/StressSense/artifacts"
OUT = os.path.join(ART, "fused_dataset.csv")

def load_or_empty(path):
    return pd.read_csv(path) if (os.path.exists(path) and os.path.getsize(path)>0) else pd.DataFrame()

def main():
    wesad = load_or_empty(os.path.join(ART, "wesad_features.csv"))
    life  = load_or_empty(os.path.join(ART, "lifestyle_features.csv"))
    text  = load_or_empty(os.path.join(ART, "text_features.csv"))

    parts = []
    # keep only numeric from each
    if not life.empty:
        parts.append(life.select_dtypes(include=[np.number]))
    if not text.empty:
        parts.append(text.select_dtypes(include=[np.number]).drop(columns=[c for c in ["stress_label"] if c in text.columns], errors="ignore"))
    if not wesad.empty:
        parts.append(wesad.select_dtypes(include=[np.number]))

    if not parts:
        raise RuntimeError("No modality features available. Run preprocess scripts first.")

    # equalize rows to min length
    min_len = min(len(p) for p in parts if len(p)>0)
    parts = [p.iloc[:min_len].reset_index(drop=True) for p in parts]

    fused = pd.concat(parts, axis=1)

    # final label (prefer lifestyle's stress_label)
    if "stress_label" in life.columns:
        y = life["stress_label"].iloc[:min_len].reset_index(drop=True)
        fused["stress_label"] = y
    else:
        # fallback: if text has it
        if "stress_label" in text.columns and len(text)>=min_len:
            fused["stress_label"] = text["stress_label"].iloc[:min_len].reset_index(drop=True)
        else:
            fused["stress_label"] = 0

    # scale numeric except label
    num_cols = fused.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "stress_label"]
    scaler = MinMaxScaler()
    fused[num_cols] = scaler.fit_transform(fused[num_cols])

    os.makedirs(ART, exist_ok=True)
    fused.to_csv(OUT, index=False)
    print(f"✅ Saved fused → {OUT}  shape={fused.shape}")
    print(fused.head())

if __name__ == "__main__":
    main()
