import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Create synthetic PCM dataset
np.random.seed(42)

temperature = np.random.uniform(30, 60, 300)
thermal_energy = temperature * 2.2 + np.random.normal(0, 5, 300)
electric_energy = thermal_energy * 0.35

df = pd.DataFrame({
    "temperature_C": temperature,
    "thermal_energy_J": thermal_energy,
    "electric_energy_Wh": electric_energy
})

# ML model
X = df[['temperature_C', 'thermal_energy_J']]
y = df['electric_energy_Wh']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Model R2 score:", r2_score(y_test, predictions))
