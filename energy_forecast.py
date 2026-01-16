# 📌 Energy Consumption Forecasting (Time Series)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================
# STEP 1: Create Dataset
# ==============================
np.random.seed(42)

days = np.arange(365)
energy = 50 + 10 * np.sin(days / 30) + np.random.normal(0, 3, 365)

df = pd.DataFrame({
    "day": days,
    "energy_consumption": energy
})

print("Sample data:")
print(df.head())

# ==============================
# STEP 2: ML Model
# ==============================
X = df[['day']]
y = df['energy_consumption']

model = LinearRegression()
model.fit(X, y)

# Predict for the next 10 days
future_days = pd.DataFrame({"day": np.arange(365, 375)})
predictions = model.predict(future_days)

print("\nPredicted energy consumption for next 10 days:")
print(predictions)

# Optional: visualize
plt.plot(df['day'], df['energy_consumption'], label='Actual')
plt.plot(future_days['day'], predictions, label='Predicted', linestyle='--')
plt.xlabel("Day")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()
