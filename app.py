import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model  = joblib.load(os.path.join(BASE_DIR, "netflix_churn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "netflix_scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()

# ── Feature engineering (mirrors your notebook exactly) ──────────────────────
COLS_TO_SCALE = ['age', 'watch_hours', 'last_login_days', 'monthly_fee',
                 'number_of_profiles', 'avg_watch_time_per_day']

SUBSCRIPTION_MAP = {'Basic': 0, 'Standard': 1, 'Premium': 2}

# Exact one-hot columns from x_train.columns (confirmed from Colab output).
# Baselines dropped by drop_first=True (alphabetically first per group):
#   gender=Female, region=Africa, device=Desktop, payment_method=Bank Transfer,
#   favorite_genre=Action
ONE_HOT_COLUMNS = [
    'gender_Male', 'gender_Other',
    'region_Asia', 'region_Europe', 'region_North America',
    'region_Oceania', 'region_South America',
    'device_Laptop', 'device_Mobile', 'device_TV', 'device_Tablet',
    'payment_method_Crypto', 'payment_method_Debit Card',
    'payment_method_Gift Card', 'payment_method_PayPal',
    'favorite_genre_Comedy', 'favorite_genre_Documentary',
    'favorite_genre_Drama', 'favorite_genre_Horror',
    'favorite_genre_Romance', 'favorite_genre_Sci-Fi',
]

# EXACT column order the model was trained on (from x_train.columns)
FEATURE_ORDER = [
    'age', 'subscription_type', 'watch_hours', 'last_login_days',
    'monthly_fee', 'number_of_profiles', 'avg_watch_time_per_day',
    'gender_Male', 'gender_Other',
    'region_Asia', 'region_Europe', 'region_North America',
    'region_Oceania', 'region_South America',
    'device_Laptop', 'device_Mobile', 'device_TV', 'device_Tablet',
    'payment_method_Crypto', 'payment_method_Debit Card',
    'payment_method_Gift Card', 'payment_method_PayPal',
    'favorite_genre_Comedy', 'favorite_genre_Documentary',
    'favorite_genre_Drama', 'favorite_genre_Horror',
    'favorite_genre_Romance', 'favorite_genre_Sci-Fi',
]

def build_input_df(age, watch_hours, last_login_days, monthly_fee,
                   number_of_profiles, avg_watch_time_per_day,
                   subscription_type, gender, region, device,
                   payment_method, favorite_genre):
    """Build a single-row DataFrame that exactly mirrors the training features."""

    # Start with all columns set to 0
    row = {col: 0 for col in FEATURE_ORDER}

    # Numerical features
    row['age']                    = age
    row['watch_hours']            = watch_hours
    row['last_login_days']        = last_login_days
    row['monthly_fee']            = monthly_fee
    row['number_of_profiles']     = number_of_profiles
    row['avg_watch_time_per_day'] = avg_watch_time_per_day

    # Ordinal encoding for subscription_type
    row['subscription_type'] = SUBSCRIPTION_MAP[subscription_type]

    # One-hot encoding — baseline stays 0, only non-baseline gets a 1
    # Baselines: gender=Female, region=Africa, device=Desktop,
    #            payment_method=Bank Transfer, favorite_genre=Action
    if gender != 'Female':
        row[f'gender_{gender}'] = 1

    if region != 'Africa':
        row[f'region_{region}'] = 1

    if device != 'Desktop':
        row[f'device_{device}'] = 1

    if payment_method != 'Bank Transfer':
        row[f'payment_method_{payment_method}'] = 1

    if favorite_genre != 'Action':
        row[f'favorite_genre_{favorite_genre}'] = 1

    df = pd.DataFrame([row])[FEATURE_ORDER]
    return df

def preprocess_and_predict(df):
    """Scale numerical columns then predict."""
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    return prediction, probability

# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" Netflix Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ─ Numerical inputs ──────────────────────────────────────────────────────────
st.subheader(" Customer Activity")

col1, col2 = st.columns(2)
with col1:
    age                    = st.number_input("Age",                         min_value=10,  max_value=100, value=30)
    watch_hours            = st.number_input("Total Watch Hours",            min_value=0.0, max_value=5000.0, value=200.0, step=10.0)
    last_login_days        = st.number_input("Days Since Last Login",        min_value=0,   max_value=365,  value=10)
with col2:
    monthly_fee            = st.number_input("Monthly Fee ($)",             min_value=0.0, max_value=50.0,  value=15.99, step=0.01)
    number_of_profiles     = st.number_input("Number of Profiles",          min_value=1,   max_value=5,    value=2)
    avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day (hrs)", min_value=0.0, max_value=24.0, value=2.0,   step=0.1)

st.divider()

# ─ Categorical inputs ─────────────────────────────────────────────────────────
st.subheader(" Customer Profile")

col3, col4 = st.columns(2)
with col3:
    subscription_type = st.selectbox("Subscription Type",
                                     ["Basic", "Standard", "Premium"])
    gender            = st.selectbox("Gender", ["Female", "Male", "Other"])
    region            = st.selectbox("Region",
                                     ["Africa", "Asia", "Europe",
                                      "North America", "Oceania", "South America"])
with col4:
    device            = st.selectbox("Primary Device",
                                     ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
    payment_method    = st.selectbox("Payment Method",
                                     ["Bank Transfer", "Crypto", "Debit Card",
                                      "Gift Card", "PayPal"])
    favorite_genre    = st.selectbox("Favourite Genre",
                                     ["Action", "Comedy", "Documentary",
                                      "Drama", "Horror", "Romance", "Sci-Fi"])

st.divider()

# ─ Prediction ─────────────────────────────────────────────────────────────────
if st.button(" Predict Churn", use_container_width=True, type="primary"):
    input_df = build_input_df(
        age, watch_hours, last_login_days, monthly_fee,
        number_of_profiles, avg_watch_time_per_day,
        subscription_type, gender, region, device,
        payment_method, favorite_genre
    )

    prediction, probability = preprocess_and_predict(input_df)

    churn_prob  = probability[1] * 100
    retain_prob = probability[0] * 100

    st.subheader("📋 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.markdown(f"Churn probability: **{churn_prob:.1f}%** | Retention probability: **{retain_prob:.1f}%**")
        st.markdown("""
        **💡 Recommended Actions:**
        - Send a re-engagement email or push notification
        - Offer a discounted plan upgrade
        - Highlight new content matching their favourite genre
        """)
    else:
        st.success(f"✅ **This customer is likely to STAY**")
        st.markdown(f"Retention probability: **{retain_prob:.1f}%** | Churn probability: **{churn_prob:.1f}%**")
        st.markdown("""
        **💡 Keep it up:**
        - Continue personalised content recommendations
        - Reward loyalty with exclusive previews
        """)

    # Probability bar
    st.progress(int(churn_prob), text=f"Churn Risk: {churn_prob:.1f}%")

st.caption("Built by Group 1 · TechCrush AI/ML Bootcamp Capstone Project")
