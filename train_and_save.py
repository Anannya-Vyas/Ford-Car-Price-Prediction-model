"""
train_and_save.py
─────────────────
Run this ONCE before launching the Streamlit app.
It trains the Linear Regression model on ford.csv and saves three files:
  • ford_model.pkl    – trained model
  • ford_scaler.pkl   – fitted StandardScaler
  • ford_columns.pkl  – ordered list of OHE column names

Usage:
    python train_and_save.py

Make sure ford.csv is in the same directory (download from Kaggle:
https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading ford.csv …")
df = pd.read_csv("ford.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── 2. Split features / target ────────────────────────────────────────────────
X = df.drop(columns=["price"])
y = df["price"]

# ── 3. One-hot encode ─────────────────────────────────────────────────────────
X_ohe = pd.get_dummies(X, columns=["model", "transmission", "fuelType"], drop_first=True)
X_ohe = X_ohe.astype(int)

# ── 4. Scale numerical features ───────────────────────────────────────────────
num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
scaler = StandardScaler()
X_ohe[num_cols] = scaler.fit_transform(X_ohe[num_cols])

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_ohe, y, test_size=0.20, random_state=42
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("Training Linear Regression model …")
model = LinearRegression()
model.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
n, p = X_test.shape
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"\n  R²          : {r2:.4f}")
print(f"  Adjusted R² : {adj_r2:.4f}")
print(f"  MAE         : £{mae:,.0f}")

# ── 8. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model,          "ford_model.pkl")
joblib.dump(scaler,         "ford_scaler.pkl")
joblib.dump(list(X_ohe.columns), "ford_columns.pkl")

print("\n✅ Saved: ford_model.pkl, ford_scaler.pkl, ford_columns.pkl")
print("You can now run:  streamlit run app.py")
