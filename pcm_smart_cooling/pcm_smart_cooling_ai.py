# 📌 PCM-Based Smart Cooling System (AI Decision)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ==============================
# STEP 1: Create Data
# ==============================
np.random.seed(42)  # reproducibility

# Simulate ambient and PCM temperatures
ambient_temp = np.random.uniform(25, 45, 250)  # °C
pcm_temp = ambient_temp - np.random.uniform(3, 10, 250)  # PCM cooler than ambient
efficiency = (ambient_temp - pcm_temp) / ambient_temp

# Create DataFrame
df = pd.DataFrame({
    "ambient_temp": ambient_temp,
    "pcm_temp": pcm_temp,
    "efficiency": efficiency
})

# AI decision: activate cooling if efficiency < 0.25
df['activate_cooling'] = df['efficiency'] < 0.25

print("Sample data:")
print(df.head())

# ==============================
# STEP 2: AI Model
# ==============================
X = df[['ambient_temp', 'pcm_temp']]
y = df['activate_cooling']

model = DecisionTreeClassifier()
model.fit(X, y)

# Test prediction
sample = pd.DataFrame({"ambient_temp": [40], "pcm_temp": [32]})
prediction = model.predict(sample)

print("\nShould cooling be activated for sample?", bool(prediction[0]))
