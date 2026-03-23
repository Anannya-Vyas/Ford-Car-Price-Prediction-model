# 🚗 Ford Price Intelligence — Streamlit App

A clean Streamlit web app that predicts the resale price of any Ford car using a Linear Regression model trained on the Kaggle Ford Car Price dataset.

---

## 📁 Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend — run this to launch the app |
| `train_and_save.py` | Trains the model and saves `.pkl` artifacts |
| `requirements.txt` | Python dependencies |
| `ford.csv` | Dataset (you download this from Kaggle — see below) |

---

## 🚀 Setup & Run (Step-by-Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Get the dataset
Download `ford.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction

Place `ford.csv` in the **same folder** as `app.py`.

### Step 3 — Train the model
```bash
python train_and_save.py
```
This will print R², Adjusted R², and MAE scores, then save:
- `ford_model.pkl`
- `ford_scaler.pkl`
- `ford_columns.pkl`

### Step 4 — Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🧠 How It Works

1. User selects car model, transmission, fuel type, year, mileage, tax, MPG, and engine size.
2. The inputs are one-hot encoded (same as training) and numericals are scaled.
3. The Linear Regression model predicts the price in £.

---

## 📦 Deploy to Streamlit Cloud (Free)

1. Push all files (including the `.pkl` files) to a **GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — done!

> ⚠️ Make sure the three `.pkl` files and `requirements.txt` are committed to your repo.
