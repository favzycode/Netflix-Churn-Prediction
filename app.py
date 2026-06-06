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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .stApp { background-color: #0f0f0f !important; }

    /* Hide default streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* All text white by default */
    p, span, div, label, li { color: #e0e0e0 !important; }
    h1, h2, h3, h4 { color: #ffffff !important; }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #E50914 !important;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Input fields */
    input[type="number"] {
        background-color: #1c1c1c !important;
        color: #ffffff !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 6px !important;
    }
    input[type="number"]:focus {
        border-color: #E50914 !important;
        box-shadow: 0 0 0 2px rgba(229,9,20,0.2) !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background-color: #1c1c1c !important;
        border: 1px solid #2e2e2e !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    [data-baseweb="menu"] { background-color: #1c1c1c !important; }
    [role="option"] { background-color: #1c1c1c !important; color: #ffffff !important; }
    [role="option"]:hover { background-color: #E50914 !important; color: #fff !important; }

    /* Labels */
    label { color: #aaaaaa !important; font-size: 0.82rem !important; font-weight: 500 !important; }

    /* Predict button */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #E50914, #b20710) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        letter-spacing: 1.5px !important;
        padding: 0.65rem 1rem !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 20px rgba(229,9,20,0.35) !important;
        transition: all 0.2s !important;
    }
    [data-testid="baseButton-primary"]:hover {
        box-shadow: 0 6px 28px rgba(229,9,20,0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Divider */
    hr { border-color: #1f1f1f !important; margin: 20px 0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161616 !important;
        border-radius: 10px !important;
        padding: 18px 16px !important;
        border: 1px solid #2a2a2a !important;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.78rem !important; letter-spacing: 1px !important; text-transform: uppercase !important; }

    /* Number input step buttons */
    [data-testid="stNumberInputField"] {
        background-color: #1c1c1c !important;
        color: #ffffff !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0f0f0f; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
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

# ── Helpers ───────────────────────────────────────────────────────────────────
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

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    colors = ['#E50914' if i == 0 else '#4a0a0a' for i in range(len(top_values))]
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1], edgecolor='none', height=0.55)

    ax.set_xlabel('Importance Score', color='#555', fontsize=8)
    ax.tick_params(colors='#666', labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_title('Feature Importance', color='#888', fontsize=9, loc='left', pad=8)

    plt.tight_layout(pad=1.5)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT — two columns: left = form, right = results
# ═════════════════════════════════════════════════════════════════════════════

# ── Top navbar ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding: 12px 0 20px 0; border-bottom: 1px solid #1f1f1f; margin-bottom:28px;">
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="background:#E50914; color:white; font-size:1.1rem; font-weight:900;
                    padding:5px 14px; border-radius:3px; letter-spacing:2px;
                    font-family:'Arial Black',sans-serif;">
            NETFLIX
        </div>
        <div>
            <div style="color:#ffffff; font-size:1rem; font-weight:700; line-height:1.2;">
                Churn Predictor
            </div>
            <div style="color:#555; font-size:0.72rem; letter-spacing:0.5px;">
                Random Forest · 98% Accuracy
            </div>
        </div>
    </div>
    <div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:20px;
                padding:5px 14px; font-size:0.75rem; color:#666;">
        TechCrush AI/ML Bootcamp · Group 1
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# ════════════════════════════════
# LEFT — Input Form
# ════════════════════════════════
with left_col:
    st.markdown('<div class="section-label">Customer Activity</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age             = st.number_input("Age", min_value=10, max_value=100, value=30)
        watch_hours     = st.number_input("Total Watch Hours", min_value=0.0, max_value=5000.0, value=200.0, step=10.0)
        last_login_days = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=10)
    with c2:
        monthly_fee            = st.number_input("Monthly Fee ($)", min_value=0.0, max_value=50.0, value=15.99, step=0.01)
        number_of_profiles     = st.number_input("Number of Profiles", min_value=1, max_value=5, value=2)
        avg_watch_time_per_day = st.number_input("Avg Watch Time/Day (hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        gender            = st.selectbox("Gender", ["Female", "Male", "Other"])
        region            = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    with c4:
        device         = st.selectbox("Primary Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"])
        payment_method = st.selectbox("Payment Method", ["Bank Transfer", "Crypto", "Debit Card", "Gift Card", "PayPal"])
        favorite_genre = st.selectbox("Favourite Genre", ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"])

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮  PREDICT CHURN", use_container_width=True, type="primary")

    # Feature importance — tucked below the form as a side note
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    with st.expander("ℹ️  What features influence this model?", expanded=False):
        st.markdown("""
        <div style="color:#666; font-size:0.78rem; margin-bottom:10px;">
        The chart below shows the top 10 features the Random Forest model
        learned to weigh most heavily when predicting churn.
        This is a property of the model itself, not a result of your input.
        </div>
        """, unsafe_allow_html=True)
        fig = plot_feature_importance(top_n=10)
        st.pyplot(fig, use_container_width=True)

# ════════════════════════════════
# RIGHT — Results Panel
# ════════════════════════════════
with right_col:
    if not predict_clicked:
        # Idle state
        st.markdown("""
        <div style="height:100%; min-height:420px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; text-align:center;
                    background:#111; border-radius:12px; border:1px dashed #222; padding:40px 30px;">
            <div style="font-size:3rem; margin-bottom:16px;">🎬</div>
            <div style="color:#333; font-size:1rem; font-weight:600; margin-bottom:8px;">
                No prediction yet
            </div>
            <div style="color:#2a2a2a; font-size:0.82rem; line-height:1.6;">
                Fill in the customer details on the left<br/>and click <strong style="color:#444;">Predict Churn</strong> to see results.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_df = build_input_df(
            age, watch_hours, last_login_days, monthly_fee,
            number_of_profiles, avg_watch_time_per_day,
            subscription_type, gender, region, device,
            payment_method, favorite_genre
        )
        prediction, probability = preprocess_and_predict(input_df)
        churn_prob  = probability[1] * 100
        retain_prob = probability[0] * 100

        # Result card wrapper
        card_border = "#E50914" if prediction == 1 else "#00c853"
        card_bg     = "#160000" if prediction == 1 else "#001209"

        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {card_border}33;
                    border-top:3px solid {card_border};
                    border-radius:12px; padding:28px 24px;">
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div style="margin:12px 0 20px 0;">
                <div style="font-size:1.6rem; font-weight:800; color:#E50914; line-height:1.2;">
                    ⚠️ High Churn Risk
                </div>
                <div style="color:#aa4444; font-size:0.88rem; margin-top:6px;">
                    This customer is likely to cancel their subscription.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin:12px 0 20px 0;">
                <div style="font-size:1.6rem; font-weight:800; color:#00c853; line-height:1.2;">
                    ✅ Low Churn Risk
                </div>
                <div style="color:#2e7d4f; font-size:0.88rem; margin-top:6px;">
                    This customer is likely to stay subscribed.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Two metrics only
        m1, m2 = st.columns(2)
        m1.metric("Churn Probability",  f"{churn_prob:.1f}%")
        m2.metric("Retention Probability", f"{retain_prob:.1f}%")

        # Risk meter
        bar_color = "#E50914" if prediction == 1 else "#00c853"
        st.markdown(f"""
        <div style="margin: 22px 0 6px 0;">
            <div style="color:#444; font-size:0.7rem; font-weight:700;
                        letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;">
                Churn Risk Meter
            </div>
            <div style="background:#1a1a1a; border-radius:20px; height:10px; overflow:hidden;">
                <div style="width:{churn_prob:.1f}%; background:{bar_color};
                            height:100%; border-radius:20px;
                            box-shadow: 0 0 8px {bar_color}88;">
                </div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        color:#333; font-size:0.7rem; margin-top:5px;">
                <span>0% Safe</span><span>{churn_prob:.1f}%</span><span>100% Critical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("""
        <div style="color:#444; font-size:0.7rem; font-weight:700;
                    letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;">
            Recommended Actions
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <ul style="color:#888 !important; font-size:0.83rem; line-height:2; padding-left:18px; margin:0;">
                <li>Send a personalised re-engagement notification</li>
                <li>Offer a discounted plan upgrade or loyalty reward</li>
                <li>Curate content based on their favourite genre</li>
                <li>Flag for customer success team follow-up</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <ul style="color:#888 !important; font-size:0.83rem; line-height:2; padding-left:18px; margin:0;">
                <li>Continue personalised content recommendations</li>
                <li>Reward loyalty with exclusive previews</li>
            </ul>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#222; font-size:0.75rem;
            padding: 32px 0 12px 0; border-top: 1px solid #161616; margin-top:32px;">
    Built by <strong style="color:#333;">Group 1</strong> ·
    TechCrush AI/ML Bootcamp Capstone · 2024
</div>
""", unsafe_allow_html=True)
