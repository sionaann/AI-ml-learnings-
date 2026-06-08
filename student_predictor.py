import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Data
data = [
    {"hours": 1, "sleep": 5, "attendance": 50, "marks": 30},
    {"hours": 2, "sleep": 6, "attendance": 60, "marks": 40},
    {"hours": 4, "sleep": 6, "attendance": 65, "marks": 55},
    {"hours": 6, "sleep": 7, "attendance": 75, "marks": 70},
    {"hours": 8, "sleep": 8, "attendance": 90, "marks": 90},
]

df = pd.DataFrame(data)

# Features & target
x_raw = df[["hours","sleep","attendance"]].values
y = df["marks"].values

# Normalize features (zero mean, unit variance)
x_mean = x_raw.mean(axis=0)
x_std = x_raw.std(axis=0)
x = (x_raw - x_mean) / x_std

# Initialize
w = np.zeros(3)
b = 0
lr = 0.01

losses = []

# Training
for epoch in range(2000):
    y_pred = np.dot(x, w) + b
    
    error = y_pred - y
    
    loss = np.mean(error**2)
    losses.append(loss)
    
    dw = np.dot(x.T, error) / len(y)
    db = np.mean(error)
    
    w -= lr * dw
    b -= lr * db

# Final predictions
y_pred = np.dot(x, w) + b

print("Weights:", w)
print("Bias:", b)
print("Predictions:", y_pred)

# New student prediction (normalize using same mean/std)
new_student = (np.array([5,7,80]) - x_mean) / x_std
predicted_marks = np.dot(new_student, w) + b

print("Predicted marks:", predicted_marks)

if predicted_marks >= 50:
    print("Pass")
else:
    print("Fail")

# 📈 Plot learning graph
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()