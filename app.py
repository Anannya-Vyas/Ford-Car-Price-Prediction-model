import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ford Price Intelligence",
    page_icon="🚗",
    layout="centered",
)

# ── Minimal dark styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #c9a84c; }
    .result-box {
        background: #111;
        border: 1px solid #c9a84c44;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-price { font-size: 2.5rem; font-weight: 700; color: #c9a84c; }
    .result-label { font-size: 0.85rem; color: #888; letter-spacing: 2px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ── Load model & artifacts ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("ford_model.pkl")
    scaler  = joblib.load("ford_scaler.pkl")
    columns = joblib.load("ford_columns.pkl")   # list of OHE column names
    return model, scaler, columns

try:
    model, scaler, ohe_columns = load_artifacts()
    artifacts_ok = True
except FileNotFoundError:
    artifacts_ok = False

# ── Constants ────────────────────────────────────────────────────────────────
FORD_MODELS = [
    "Fiesta", "Focus", "Kuga", "EcoSport", "C-MAX", "Puma", "Mondeo",
    "Ka+", "S-MAX", "B-MAX", "Galaxy", "Edge", "Mustang", "Ranger",
    "KA", "Grand C-MAX", "Tourneo Custom", "Tourneo Connect",
    "Grand Tourneo Connect", "Fusion", "Streetka", "Escort", "Transit Tourneo",
]
TRANSMISSIONS = ["Manual", "Automatic", "Semi-Auto", "Other"]
FUEL_TYPES    = ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🚗 FORD — Price Intelligence")
st.caption("Predict the resale price of a Ford vehicle using a trained Linear Regression model.")

if not artifacts_ok:
    st.warning(
        "⚠️ Model files not found. Run **`train_and_save.py`** first to generate "
        "`ford_model.pkl`, `ford_scaler.pkl`, and `ford_columns.pkl`."
    )

st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
with st.form("predictor_form"):
    col1, col2 = st.columns(2)

    with col1:
        car_model    = st.selectbox("Model",        FORD_MODELS)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        fuel_type    = st.selectbox("Fuel Type",    FUEL_TYPES)
        year         = st.number_input("Year", min_value=1990, max_value=2024, value=2018, step=1)

    with col2:
        mileage     = st.number_input("Mileage (miles)", min_value=0,    max_value=300_000, value=15_000,  step=500)
        tax         = st.number_input("Road Tax (£)",    min_value=0,    max_value=600,     value=150,     step=10)
        mpg         = st.number_input("MPG",             min_value=1.0,  max_value=200.0,   value=55.0,    step=0.5)
        engine_size = st.number_input("Engine Size (L)", min_value=0.0,  max_value=7.0,     value=1.5,     step=0.1, format="%.1f")

    submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    if not artifacts_ok:
        st.error("Cannot predict — model files are missing. See warning above.")
    else:
        # Build raw feature row (mirrors train_and_save.py preprocessing)
        raw = pd.DataFrame([{
            "model":        car_model,
            "year":         year,
            "transmission": transmission,
            "mileage":      mileage,
            "fuelType":     fuel_type,
            "tax":          tax,
            "mpg":          mpg,
            "engineSize":   engine_size,
        }])

        # One-hot encode
        raw_ohe = pd.get_dummies(raw, columns=["model", "transmission", "fuelType"], drop_first=True)

        # Align columns to training schema
        raw_ohe = raw_ohe.reindex(columns=ohe_columns, fill_value=0).astype(int)

        # Scale numerical columns
        num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
        raw_ohe[num_cols] = scaler.transform(raw_ohe[num_cols])

        prediction = model.predict(raw_ohe)[0]
        prediction = max(0, prediction)   # floor at £0

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Resale Price</div>
            <div class="result-price">£{prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence note
        st.caption(
            "Estimate based on a Linear Regression model trained on the Kaggle "
            "Ford Car Price dataset. Actual prices may vary."
        )
