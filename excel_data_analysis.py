# 📌 Data Analytics Project (Excel + Python)

import pandas as pd

# ==============================
# STEP 1: Read Excel Data
# ==============================
# Make sure energy_data.xlsx is uploaded in your GitHub repo or local folder
file_path = "energy_data.xlsx"
df = pd.read_excel(file_path)

print("Sample data:")
print(df.head())

# ==============================
# STEP 2: Basic Analysis
# ==============================
# Group by month and calculate mean energy consumption
monthly_avg = df.groupby("month").mean()
print("\nAverage energy consumption per month:")
print(monthly_avg)

# Optional: export results
monthly_avg.to_excel("monthly_avg_energy.xlsx")
