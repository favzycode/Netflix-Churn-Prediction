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
    layout="wide"
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif !important; }

    .stApp { background-color: #0a0a0a !important; }
    #MainMenu, footer, header { visibility: hidden; }

    p, span, div, label, li { color: #d0d0d0 !important; }
    h1, h2, h3, h4 { color: #ffffff !important; }

    /* Inputs */
    input[type="number"] {
        background-color: #161616 !important;
        color: #ffffff !important;
        border: 1px solid #252525 !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    input[type="number"]:focus {
        border-color: #E50914 !important;
        box-shadow: 0 0 0 3px rgba(229,9,20,0.15) !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background-color: #161616 !important;
        border: 1px solid #252525 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div > div { color: #ffffff !important; }
    [data-baseweb="menu"] { background-color: #1a1a1a !important; border: 1px solid #2a2a2a !important; }
    [role="option"] { background-color: #1a1a1a !important; color: #ccc !important; }
    [role="option"]:hover { background-color: #E50914 !important; color: #fff !important; }

    /* Labels */
    label { color: #666 !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.3px !important; }

    /* Predict button */
    [data-testid="baseButton-primary"] {
        background: #E50914 !important;
        color: #fff !important;
        border: none !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        height: 48px !important;
        box-shadow: 0 0 24px rgba(229,9,20,0.3) !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background: #c5060f !important;
        box-shadow: 0 0 36px rgba(229,9,20,0.5) !important;
    }

    hr { border-color: #1a1a1a !important; }

    /* Metric override */
    [data-testid="stMetric"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    [data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #444 !important;
        font-size: 0.7rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0f0f0f !important;
        border: 1px solid #1a1a1a !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary { color: #444 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model  = joblib.load(os.path.join(BASE_DIR, "netflix_churn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "netflix_scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()

# ── Constants ─────────────────────────────────────────────────────────────────
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
    'watch_hours': 'Watch Hours', 'last_login_days': 'Days Since Login',
    'age': 'Age', 'monthly_fee': 'Monthly Fee',
    'avg_watch_time_per_day': 'Avg Daily Watch', 'subscription_type': 'Subscription Type',
    'number_of_profiles': 'No. of Profiles', 'gender_Male': 'Gender: Male',
    'gender_Other': 'Gender: Other', 'region_Asia': 'Region: Asia',
    'region_Europe': 'Region: Europe', 'region_North America': 'Region: N. America',
    'region_Oceania': 'Region: Oceania', 'region_South America': 'Region: S. America',
    'device_Laptop': 'Device: Laptop', 'device_Mobile': 'Device: Mobile',
    'device_TV': 'Device: TV', 'device_Tablet': 'Device: Tablet',
    'payment_method_Crypto': 'Payment: Crypto', 'payment_method_Debit Card': 'Payment: Debit',
    'payment_method_Gift Card': 'Payment: Gift Card', 'payment_method_PayPal': 'Payment: PayPal',
    'favorite_genre_Comedy': 'Genre: Comedy', 'favorite_genre_Documentary': 'Genre: Documentary',
    'favorite_genre_Drama': 'Genre: Drama', 'favorite_genre_Horror': 'Genre: Horror',
    'favorite_genre_Romance': 'Genre: Romance', 'favorite_genre_Sci-Fi': 'Genre: Sci-Fi',
}

# ── Logic ─────────────────────────────────────────────────────────────────────
def build_input_df(age, watch_hours, last_login_days, monthly_fee,
                   number_of_profiles, avg_watch_time_per_day,
                   subscription_type, gender, region, device,
                   payment_method, favorite_genre):
    row = {col: 0 for col in FEATURE_ORDER}
    row.update({'age': age, 'watch_hours': watch_hours, 'last_login_days': last_login_days,
                'monthly_fee': monthly_fee, 'number_of_profiles': number_of_profiles,
                'avg_watch_time_per_day': avg_watch_time_per_day,
                'subscription_type': SUBSCRIPTION_MAP[subscription_type]})
    if gender != 'Female':                row[f'gender_{gender}'] = 1
    if region != 'Africa':                row[f'region_{region}'] = 1
    if device != 'Desktop':               row[f'device_{device}'] = 1
    if payment_method != 'Bank Transfer': row[f'payment_method_{payment_method}'] = 1
    if favorite_genre != 'Action':        row[f'favorite_genre_{favorite_genre}'] = 1
    return pd.DataFrame([row])[FEATURE_ORDER]

def preprocess_and_predict(df):
    df = df.copy()
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob

def plot_feature_importance():
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    features    = [FEATURE_LABELS.get(FEATURE_ORDER[i], FEATURE_ORDER[i]) for i in indices]
    values      = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#0f0f0f')

    colors = ['#E50914'] + ['#3a0a0a'] * 9
    ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor='none', height=0.5)

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(colors='#444', labelsize=7.5, length=0)
    ax.set_xlabel('Importance', color='#333', fontsize=7.5)
    ax.set_title('Top 10 Predictive Features', color='#555', fontsize=8.5,
                 loc='left', pad=10, fontweight='600')
    plt.tight_layout(pad=1.2)
    return fig

# ═══════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════

# Navbar
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:16px 4px 24px 4px; border-bottom:1px solid #151515; margin-bottom:32px;">
    <div style="display:flex; align-items:center; gap:16px;">
        <span style="background:#E50914; color:#fff; font-size:1rem; font-weight:900;
                     padding:6px 16px; border-radius:4px; letter-spacing:3px;
                     font-family:'Arial Black',sans-serif;">NETFLIX</span>
        <div>
            <div style="color:#fff; font-weight:700; font-size:1.05rem;">Churn Predictor</div>
            <div style="color:#333; font-size:0.7rem; letter-spacing:0.5px; margin-top:1px;">
                Random Forest Classifier &nbsp;·&nbsp; 98% Accuracy
            </div>
        </div>
    </div>
    <div style="color:#2a2a2a; font-size:0.72rem; letter-spacing:0.5px;">
        TechCrush AI/ML Bootcamp &nbsp;·&nbsp; Group 1
    </div>
</div>
""", unsafe_allow_html=True)

# Two columns
left, right = st.columns([1.05, 0.95], gap="large")

# ══════════════════════
# LEFT — Input form
# ══════════════════════
with left:

    # Activity section
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:700; letter-spacing:2.5px;
                color:#E50914; text-transform:uppercase; margin-bottom:14px;">
        Customer Activity
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        age             = st.number_input("Age", min_value=10, max_value=100, value=30)
        watch_hours     = st.number_input("Total Watch Hours", min_value=0.0, max_value=5000.0, value=200.0, step=10.0)
        last_login_days = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=10)
    with a2:
        monthly_fee            = st.number_input("Monthly Fee ($)", min_value=0.0, max_value=50.0, value=15.99, step=0.01)
        number_of_profiles     = st.number_input("Number of Profiles", min_value=1, max_value=5, value=2)
        avg_watch_time_per_day = st.number_input("Avg Watch Time / Day (hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Profile section
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:700; letter-spacing:2.5px;
                color:#E50914; text-transform:uppercase; margin-bottom:14px;">
        Customer Profile
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        gender            = st.selectbox("Gender", ["Female", "Male", "Other"])
        region            = st.selectbox("Region", ["Africa", "Asia", "Europe",
                                                     "North America", "Oceania", "South America"])
    with b2:
        device         = st.selectbox("Primary Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
        payment_method = st.selectbox("Payment Method", ["Bank Transfer", "Crypto",
                                                          "Debit Card", "Gift Card", "PayPal"])
        favorite_genre = st.selectbox("Favourite Genre", ["Action", "Comedy", "Documentary",
                                                           "Drama", "Horror", "Romance", "Sci-Fi"])

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    clicked = st.button("🔮  PREDICT CHURN", use_container_width=True, type="primary")

    # Feature importance — collapsed side note
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("📊  Model Feature Importance", expanded=False):
        st.markdown("""
        <div style="color:#3a3a3a; font-size:0.76rem; line-height:1.7; margin-bottom:12px;">
        These are the features the model weighs most heavily when
        predicting churn — independent of any individual input.
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(plot_feature_importance(), use_container_width=True)

# ══════════════════════
# RIGHT — Results
# ══════════════════════
with right:

    if not clicked:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; text-align:center;
                    min-height:460px; background:#0d0d0d;
                    border:1px dashed #1a1a1a; border-radius:14px; padding:48px 32px;">
            <div style="font-size:2.8rem; margin-bottom:18px; opacity:0.3;">🎬</div>
            <div style="color:#222; font-size:0.95rem; font-weight:600; margin-bottom:8px;">
                Awaiting prediction
            </div>
            <div style="color:#1a1a1a; font-size:0.8rem; line-height:1.8; max-width:220px;">
                Complete the form and click<br/>
                <strong style="color:#2a2a2a;">Predict Churn</strong> to see results here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        df = build_input_df(age, watch_hours, last_login_days, monthly_fee,
                            number_of_profiles, avg_watch_time_per_day,
                            subscription_type, gender, region, device,
                            payment_method, favorite_genre)
        pred, prob = preprocess_and_predict(df)
        churn_pct  = prob[1] * 100
        retain_pct = prob[0] * 100

        is_churn   = pred == 1
        accent     = "#E50914" if is_churn else "#00c853"
        bg         = "#110000" if is_churn else "#001510"
        status_txt = "High Churn Risk" if is_churn else "Low Churn Risk"
        status_ico = "⚠️" if is_churn else "✅"
        sub_txt    = ("This customer is likely to cancel their subscription."
                      if is_churn else
                      "This customer is likely to remain subscribed.")

        # Top accent line + card
        st.markdown(f"""
        <div style="border-radius:14px; background:{bg};
                    border:1px solid {accent}22;
                    box-shadow: 0 0 40px {accent}11;
                    padding:28px 28px 24px 28px;">

            <div style="font-size:0.65rem; font-weight:700; letter-spacing:2.5px;
                        color:{accent}; text-transform:uppercase; margin-bottom:16px;">
                Prediction Result
            </div>

            <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:8px;">
                <div style="font-size:1.8rem; line-height:1;">{status_ico}</div>
                <div>
                    <div style="font-size:1.45rem; font-weight:800; color:#ffffff;
                                line-height:1.2; letter-spacing:-0.3px;">
                        {status_txt}
                    </div>
                    <div style="color:#444; font-size:0.82rem; margin-top:5px;">
                        {sub_txt}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Churn Probability", f"{churn_pct:.1f}%")
        m2.metric("Retention Probability", f"{retain_pct:.1f}%")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Risk meter
        st.markdown(f"""
        <div style="background:#0f0f0f; border:1px solid #1a1a1a;
                    border-radius:12px; padding:20px 22px;">
            <div style="font-size:0.65rem; font-weight:700; letter-spacing:2px;
                        color:#333; text-transform:uppercase; margin-bottom:14px;">
                Churn Risk Meter
            </div>
            <div style="background:#161616; border-radius:99px; height:8px; overflow:hidden;">
                <div style="width:{churn_pct:.1f}%; background:{accent};
                            height:100%; border-radius:99px;
                            box-shadow:0 0 10px {accent}88; transition:width 0.6s;">
                </div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        margin-top:8px; font-size:0.68rem; color:#2a2a2a;">
                <span>0% — Safe</span>
                <span style="color:{accent}; font-weight:700;">{churn_pct:.1f}%</span>
                <span>100% — Critical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # Recommendations
        if is_churn:
            actions = [
                "Send a personalised re-engagement notification",
                "Offer a discounted upgrade or loyalty reward",
                "Curate recommendations based on favourite genre",
                "Flag for customer success team follow-up",
            ]
        else:
            actions = [
                "Continue personalised content recommendations",
                "Reward loyalty with exclusive previews or early access",
                "Offer referral incentives to increase plan stickiness",
            ]

        actions_html = "".join([
            f'<div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:10px;">'
            f'<div style="width:5px; height:5px; background:{accent}; border-radius:50%; '
            f'margin-top:6px; flex-shrink:0;"></div>'
            f'<div style="color:#555; font-size:0.82rem; line-height:1.5;">{a}</div></div>'
            for a in actions
        ])

        st.markdown(f"""
        <div style="background:#0f0f0f; border:1px solid #1a1a1a;
                    border-radius:12px; padding:20px 22px;">
            <div style="font-size:0.65rem; font-weight:700; letter-spacing:2px;
                        color:#333; text-transform:uppercase; margin-bottom:14px;">
                Recommended Actions
            </div>
            {actions_html}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:40px 0 16px 0; border-top:1px solid #111; margin-top:40px;">
    <span style="color:#1e1e1e; font-size:0.72rem; letter-spacing:0.5px;">
        Built by Group 1 &nbsp;·&nbsp; TechCrush AI/ML Bootcamp Capstone &nbsp;·&nbsp; 2025
    </span>
</div>
""", unsafe_allow_html=True)
