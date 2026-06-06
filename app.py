import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered"
)

# ── Netflix dark theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #141414 !important; color: #FFFFFF !important; }

    /* All text */
    p, span, div, label { color: #FFFFFF !important; }

    /* Input fields */
    input[type="number"] {
        background-color: #2b2b2b !important;
        color: #FFFFFF !important;
        border: 1px solid #444 !important;
        border-radius: 4px !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background-color: #2b2b2b !important;
        border: 1px solid #444 !important;
        color: #FFFFFF !important;
    }

    /* Dropdown menu options */
    [data-baseweb="menu"] { background-color: #2b2b2b !important; }
    [role="option"] { background-color: #2b2b2b !important; color: #FFFFFF !important; }
    [role="option"]:hover { background-color: #E50914 !important; }

    /* Labels */
    label { color: #FFFFFF !important; font-weight: 500 !important; }

    /* Headers */
    h1 { color: #FFFFFF !important; }
    h2, h3, h4 { color: #FFFFFF !important; }

    /* Primary button — Netflix red */
    [data-testid="baseButton-primary"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        letter-spacing: 1px !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background-color: #b20710 !important;
    }

    /* Divider */
    hr { border-color: #333 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1f1f1f !important;
        border-radius: 8px !important;
        padding: 14px !important;
        border: 1px solid #333 !important;
    }
    [data-testid="stMetricValue"] { color: #E50914 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #aaa !important; }

    /* Number input buttons */
    [data-testid="stNumberInputField"] {
        background-color: #2b2b2b !important;
        color: #FFFFFF !important;
    }

    /* Column gaps */
    [data-testid="column"] { padding: 0 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model  = joblib.load(os.path.join(BASE_DIR, "netflix_churn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "netflix_scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()

# ── Feature constants ─────────────────────────────────────────────────────────
COLS_TO_SCALE = ['age', 'watch_hours', 'last_login_days', 'monthly_fee',
                 'number_of_profiles', 'avg_watch_time_per_day']

SUBSCRIPTION_MAP = {'Basic': 0, 'Standard': 1, 'Premium': 2}

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

FEATURE_LABELS = {
    'watch_hours': 'Watch Hours',
    'last_login_days': 'Days Since Login',
    'age': 'Age',
    'monthly_fee': 'Monthly Fee',
    'avg_watch_time_per_day': 'Avg Daily Watch Time',
    'subscription_type': 'Subscription Type',
    'number_of_profiles': 'No. of Profiles',
    'gender_Male': 'Gender: Male',
    'gender_Other': 'Gender: Other',
    'region_Asia': 'Region: Asia',
    'region_Europe': 'Region: Europe',
    'region_North America': 'Region: N. America',
    'region_Oceania': 'Region: Oceania',
    'region_South America': 'Region: S. America',
    'device_Laptop': 'Device: Laptop',
    'device_Mobile': 'Device: Mobile',
    'device_TV': 'Device: TV',
    'device_Tablet': 'Device: Tablet',
    'payment_method_Crypto': 'Payment: Crypto',
    'payment_method_Debit Card': 'Payment: Debit Card',
    'payment_method_Gift Card': 'Payment: Gift Card',
    'payment_method_PayPal': 'Payment: PayPal',
    'favorite_genre_Comedy': 'Genre: Comedy',
    'favorite_genre_Documentary': 'Genre: Documentary',
    'favorite_genre_Drama': 'Genre: Drama',
    'favorite_genre_Horror': 'Genre: Horror',
    'favorite_genre_Romance': 'Genre: Romance',
    'favorite_genre_Sci-Fi': 'Genre: Sci-Fi',
}

# ── Helper functions ──────────────────────────────────────────────────────────
def build_input_df(age, watch_hours, last_login_days, monthly_fee,
                   number_of_profiles, avg_watch_time_per_day,
                   subscription_type, gender, region, device,
                   payment_method, favorite_genre):
    row = {col: 0 for col in FEATURE_ORDER}
    row['age']                    = age
    row['watch_hours']            = watch_hours
    row['last_login_days']        = last_login_days
    row['monthly_fee']            = monthly_fee
    row['number_of_profiles']     = number_of_profiles
    row['avg_watch_time_per_day'] = avg_watch_time_per_day
    row['subscription_type']      = SUBSCRIPTION_MAP[subscription_type]
    if gender != 'Female':                row[f'gender_{gender}'] = 1
    if region != 'Africa':                row[f'region_{region}'] = 1
    if device != 'Desktop':               row[f'device_{device}'] = 1
    if payment_method != 'Bank Transfer': row[f'payment_method_{payment_method}'] = 1
    if favorite_genre != 'Action':        row[f'favorite_genre_{favorite_genre}'] = 1
    return pd.DataFrame([row])[FEATURE_ORDER]

def preprocess_and_predict(df):
    df = df.copy()
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    return prediction, probability

def plot_feature_importance(top_n=10):
    importances  = model.feature_importances_
    indices      = np.argsort(importances)[::-1][:top_n]
    top_features = [FEATURE_LABELS.get(FEATURE_ORDER[i], FEATURE_ORDER[i]) for i in indices]
    top_values   = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    colors = ['#E50914' if i == 0 else '#8B0000' for i in range(len(top_values))]
    bars = ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1], edgecolor='none', height=0.6)

    ax.set_xlabel('Importance Score', color='#aaaaaa', fontsize=9)
    ax.set_title(f'Top {top_n} Features Driving Churn Predictions', color='#FFFFFF',
                 fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors='#cccccc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.xaxis.label.set_color('#aaaaaa')

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='#ffffff', fontsize=7.5, fontweight='600')

    plt.tight_layout()
    return fig

# ── Netflix Logo Header ───────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:4px; padding-top:8px;">
    <div style="background:#E50914; color:white; font-size:2.2rem; font-weight:900;
                padding:4px 20px; border-radius:4px; letter-spacing:3px;
                font-family:'Arial Black', sans-serif; box-shadow:0 4px 15px rgba(229,9,20,0.4);">
        NETFLIX
    </div>
</div>
<h2 style="color:#FFFFFF !important; margin:10px 0 2px 0; font-size:1.5rem; font-weight:700;">
    Customer Churn Predictor
</h2>
<p style="color:#888; font-size:0.88rem; margin-bottom:0;">
    Predict whether a subscriber is likely to cancel · Powered by Random Forest · <span style="color:#E50914; font-weight:600;">98% Accuracy</span>
</p>
<hr style="border-color:#333; margin-top:14px;"/>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("####  Customer Activity")
col1, col2 = st.columns(2)
with col1:
    age                    = st.number_input("Age", min_value=10, max_value=100, value=30)
    watch_hours            = st.number_input("Total Watch Hours", min_value=0.0, max_value=5000.0, value=200.0, step=10.0)
    last_login_days        = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=10)
with col2:
    monthly_fee            = st.number_input("Monthly Fee ($)", min_value=0.0, max_value=50.0, value=15.99, step=0.01)
    number_of_profiles     = st.number_input("Number of Profiles", min_value=1, max_value=5, value=2)
    avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day (hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)

st.markdown("<hr style='border-color:#333'/>", unsafe_allow_html=True)
st.markdown("####  Customer Profile")

col3, col4 = st.columns(2)
with col3:
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    gender            = st.selectbox("Gender", ["Female", "Male", "Other"])
    region            = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
with col4:
    device            = st.selectbox("Primary Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
    payment_method    = st.selectbox("Payment Method", ["Bank Transfer", "Crypto", "Debit Card", "Gift Card", "PayPal"])
    favorite_genre    = st.selectbox("Favourite Genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])

st.markdown("<hr style='border-color:#333'/>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("  PREDICT CHURN", use_container_width=True, type="primary"):
    input_df = build_input_df(
        age, watch_hours, last_login_days, monthly_fee,
        number_of_profiles, avg_watch_time_per_day,
        subscription_type, gender, region, device,
        payment_method, favorite_genre
    )
    prediction, probability = preprocess_and_predict(input_df)
    churn_prob  = probability[1] * 100
    retain_prob = probability[0] * 100

    st.markdown("####  Prediction Result")

    # Metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Risk",       f"{churn_prob:.1f}%")
    m2.metric("Retention Chance", f"{retain_prob:.1f}%")
    m3.metric("Model Confidence", f"{max(churn_prob, retain_prob):.1f}%")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Result banner
    if prediction == 1:
        st.markdown(f"""
        <div style="background:#2a0000; border-left:5px solid #E50914;
                    padding:18px 20px; border-radius:6px; margin:12px 0;">
            <div style="font-size:1.25rem; font-weight:800; color:#E50914; margin-bottom:6px;">
                ⚠️ High Churn Risk Detected
            </div>
            <div style="color:#ffbbbb; font-size:0.95rem;">
                This customer has a <strong style="color:#E50914;">{churn_prob:.1f}%</strong>
                probability of cancelling their subscription.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **💡 Recommended Retention Actions:**
        - Send a personalised re-engagement email or push notification
        - Offer a discounted plan upgrade or a loyalty reward
        - Curate content recommendations based on their favourite genre
        - Flag for the customer success team to follow up
        """)
    else:
        st.markdown(f"""
        <div style="background:#001a0d; border-left:5px solid #00c853;
                    padding:18px 20px; border-radius:6px; margin:12px 0;">
            <div style="font-size:1.25rem; font-weight:800; color:#00c853; margin-bottom:6px;">
                ✅ Low Churn Risk
            </div>
            <div style="color:#aaffcc; font-size:0.95rem;">
                This customer has a <strong style="color:#00c853;">{retain_prob:.1f}%</strong>
                probability of staying subscribed.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **💡 Evaluation: This is good news!**
        - Continue personalised content recommendations
        - Reward loyalty with exclusive previews or early access
        """)

    # Churn risk meter
    bar_color = "#E50914" if prediction == 1 else "#00c853"
    st.markdown(f"""
    <div style="margin:16px 0 4px 0; color:#aaa; font-size:0.85rem; font-weight:600;">
        CHURN RISK METER
    </div>
    <div style="background:#2b2b2b; border-radius:20px; height:16px; overflow:hidden;">
        <div style="width:{churn_prob:.1f}%; background:{bar_color};
                    height:100%; border-radius:20px;">
        </div>
    </div>
    <div style="display:flex; justify-content:space-between;
                color:#555; font-size:0.75rem; margin-top:5px;">
        <span>0% — Safe</span><span>50%</span><span>100% — Critical</span>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("<hr style='border-color:#333; margin-top:24px'/>", unsafe_allow_html=True)
    st.markdown("#### 📈 What's Driving Churn Predictions?")
    st.markdown("<span style='color:#888; font-size:0.85rem'>Top 10 most influential features learned by the Random Forest model</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    fig = plot_feature_importance(top_n=10)
    st.pyplot(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#222; margin-top:36px'/>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#444; font-size:0.78rem; padding-bottom:16px;'>
 <strong style="color:#666;">Group 1</strong> ·
    TechCrush AI/ML Bootcamp Capstone Project
</p>
""", unsafe_allow_html=True)
