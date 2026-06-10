import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# =========================
# 1. LOAD DATA (FIXED PATH)
# =========================
df = pd.read_csv(r"D:\gym-intelligence system\megaGymDataset.csv")

# =========================
# 2. CLEAN DATA
# =========================
print("Columns:\n", df.columns)

# Keep only useful columns
df = df[["Type", "BodyPart", "Equipment", "Level"]]

# Drop missing values
df = df.dropna()

# =========================
# 3. CONVERT TARGET
# =========================
df["Level"] = df["Level"].map({
    "Beginner": 0,
    "Intermediate": 1,
    "Expert": 2
})

df = df.dropna()

# =========================
# 4. FEATURES & TARGET
# =========================
X = df[["Type", "BodyPart", "Equipment"]]
y = df["Level"]

# =========================
# 5. ENCODE TEXT
# =========================
X = pd.get_dummies(X)

# =========================
# 6. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# =========================
# 8. EVALUATE
# =========================
accuracy = model.score(X_test, y_test)
print("\nAccuracy:", accuracy)

# =========================
# 9. TEST PREDICTION
# =========================
sample = X_test.iloc[0]
prediction = model.predict([sample])[0]

label_map = {0: "Beginner", 1: "Intermediate", 2: "Expert"}

print("\nPredicted Level:", label_map[prediction])