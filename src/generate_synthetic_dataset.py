import pandas as pd
import numpy as np

np.random.seed(42)

# Define feature names
columns = [
    "sleep_duration", "sleep_quality", "study_hours", "screen_time",
    "academic_pressure", "exam_stress", "study_satisfaction", "procrastination",
    "exercise_frequency", "water_intake", "social_activity", "diet_quality",
    "bmi", "health_issues", "depression", "anxiety", "panic_attack",
    "specialist", "motivation_level", "concentration_level", "sentiment_score",
    "age", "cgpa", "stress_label"
]

n = 100

# Generate semi-realistic data
data = {
    "sleep_duration": np.random.normal(6.5, 1.2, n).clip(3, 9),
    "sleep_quality": np.random.randint(1, 6, n),
    "study_hours": np.random.randint(1, 10, n),
    "screen_time": np.random.randint(2, 10, n),
    "academic_pressure": np.random.randint(1, 6, n),
    "exam_stress": np.random.randint(1, 6, n),
    "study_satisfaction": np.random.randint(1, 6, n),
    "procrastination": np.random.randint(0, 2, n),
    "exercise_frequency": np.random.randint(0, 7, n),
    "water_intake": np.round(np.random.uniform(1.0, 4.0, n), 1),
    "social_activity": np.random.randint(0, 2, n),
    "diet_quality": np.random.randint(1, 6, n),
    "bmi": np.round(np.random.uniform(18.0, 30.0, n), 1),
    "health_issues": np.random.randint(0, 2, n),
    "depression": np.random.randint(0, 2, n),
    "anxiety": np.random.randint(0, 2, n),
    "panic_attack": np.random.randint(0, 2, n),
    "specialist": np.random.randint(0, 2, n),
    "motivation_level": np.random.randint(1, 6, n),
    "concentration_level": np.random.randint(1, 6, n),
    "sentiment_score": np.round(np.random.uniform(-1, 1, n), 2),
    "age": np.random.randint(18, 25, n),
    "cgpa": np.round(np.random.uniform(5.0, 10.0, n), 2)
}

df = pd.DataFrame(data)

# Create stress score combining meaningful factors
stress_score = (
    (6 - df["sleep_quality"]) * 0.8 +
    (df["academic_pressure"] + df["exam_stress"]) * 0.7 +
    (1 - df["study_satisfaction"] / 5) * 3 +
    df["procrastination"] * 2 +
    (5 - df["exercise_frequency"] / 2) * 0.5 +
    df["health_issues"] * 2 +
    df["depression"] * 3 +
    df["anxiety"] * 2 +
    df["panic_attack"] * 2 +
    (1 - df["sentiment_score"]) * 2
)

# Normalize and bin into 3 categories
stress_scaled = (stress_score - stress_score.min()) / (stress_score.max() - stress_score.min()) * 10
df["stress_label"] = pd.cut(stress_scaled, bins=[-1, 3.3, 6.6, 10], labels=[0, 1, 2]).astype(int)

# Save dataset
out_path = r"C:/Users/kaviy/OneDrive/Desktop/StressSense/data/student_stress_extended.csv"
df.to_csv(out_path, index=False)
print(f"✅ Smart dataset created successfully at:\n{out_path}")
print("\nSample preview:")
print(df.head())
