import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor — India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f1117; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #8b8fa8;
    margin-top: 6px;
    font-weight: 300;
}

.metric-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b8fa8;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-unit {
    font-size: 12px;
    color: #8b8fa8;
    margin-top: 2px;
}

.aqi-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 30px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 8px;
}

.info-box {
    background: #1a1d2e;
    border-left: 3px solid #4f7cff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: #c5c8d6;
    margin: 12px 0;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4f7cff;
    margin-bottom: 16px;
    margin-top: 8px;
}

.stSlider > div > div > div { background: #4f7cff !important; }
div[data-testid="stSidebar"] { background: #0a0c14 !important; }
div[data-testid="stSidebar"] .stMarkdown { color: #c5c8d6; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('data/featured_aqi.csv', parse_dates=['Date'])
    return df

try:
    model, feature_cols = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model files. Make sure models/ and data/ folders are in the same directory as app.py\n\nError: {e}")


# ─────────────────────────────────────────────
#  AQI HELPERS
# ─────────────────────────────────────────────
def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "#00c46a", "#0a2e1a", "Air quality is satisfactory. Outdoor activities are safe."
    elif aqi <= 100:
        return "Satisfactory", "#a8e063", "#1a2a0a", "Air quality is acceptable. Sensitive groups should be cautious."
    elif aqi <= 200:
        return "Moderate", "#ffcc00", "#2a2200", "Sensitive individuals may experience discomfort. Limit prolonged outdoor exposure."
    elif aqi <= 300:
        return "Poor", "#ff8c00", "#2a1500", "Everyone may experience health effects. Avoid outdoor activities if possible."
    elif aqi <= 400:
        return "Very Poor", "#ff4444", "#2a0a0a", "Health alert — everyone is affected. Stay indoors with windows closed."
    else:
        return "Severe", "#9b30ff", "#1a0a2a", "Emergency conditions. Avoid all outdoor exposure. Wear N95 masks if going out."

def get_city_stats(city):
    city_df = df[df['City'] == city]
    return {
        'mean': city_df['AQI'].mean(),
        'max': city_df['AQI'].max(),
        'min': city_df['AQI'].min(),
        'worst_month': city_df.groupby('Month')['AQI'].mean().idxmax(),
        'best_month':  city_df.groupby('Month')['AQI'].mean().idxmin(),
    }

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">📍 Location & Time</div>', unsafe_allow_html=True)

    cities = sorted(df['City'].unique()) if model_loaded else ["Delhi"]
    selected_city = st.selectbox("Select City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

    selected_date = st.date_input("Prediction Date", value=pd.Timestamp("2020-03-15"))

    st.markdown('<div class="section-header" style="margin-top:24px">🧪 Pollutant Levels</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Adjust sliders to simulate different pollution scenarios.</div>', unsafe_allow_html=True)

    pm25 = st.slider("PM2.5 (μg/m³)", 0.0, 250.0, 85.0, 1.0)
    pm10 = st.slider("PM10 (μg/m³)",  0.0, 470.0, 140.0, 1.0)
    no2  = st.slider("NO₂ (μg/m³)",   0.0, 120.0, 45.0, 0.5)
    no   = st.slider("NO (μg/m³)",    0.0, 65.0,  18.0, 0.5)
    nox  = st.slider("NOx (μg/m³)",   0.0, 125.0, 52.0, 0.5)
    nh3  = st.slider("NH₃ (μg/m³)",   0.0, 135.0, 28.0, 0.5)
    co   = st.slider("CO (mg/m³)",    0.0, 4.5,   1.2, 0.1)
    so2  = st.slider("SO₂ (μg/m³)",   0.0, 45.0,  15.0, 0.5)
    o3   = st.slider("O₃ (μg/m³)",    0.0, 130.0, 38.0, 0.5)

    st.markdown('<div class="section-header" style="margin-top:24px">📈 Historical Context</div>', unsafe_allow_html=True)
    aqi_lag1 = st.slider("Yesterday's AQI",      10, 500, 160, 5)
    aqi_lag3 = st.slider("3 Days Ago AQI",        10, 500, 155, 5)
    aqi_lag7 = st.slider("7 Days Ago AQI",        10, 500, 148, 5)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown('''
<div class="hero-title">🌫️ AQI Prediction System</div>
<div class="hero-sub">Spatio-Temporal ML Model · 26 Indian Cities · Random Forest · R² = 0.9426</div>
<hr style="border-color:#2a2d3e; margin: 20px 0 24px 0">
''', unsafe_allow_html=True)

if model_loaded:

    # ── Build feature vector ──────────────────
    date = pd.Timestamp(selected_date)
    month = date.month
    season_map = {12:3, 1:3, 2:3, 3:1, 4:1, 5:1, 6:0, 7:0, 8:0, 9:0, 10:2, 11:2}
    season_encoded = season_map[month]

    city_mapping = joblib.load('models/city_mapping.pkl')
    city_encoded = city_mapping.get(selected_city, 0)

    # rolling averages approximated from lag values
    rolling_7d    = (aqi_lag1 + aqi_lag3 + aqi_lag7) / 3
    rolling_14d   = rolling_7d * 0.97
    rolling_30d   = rolling_7d * 0.94
    rolling_7d_std = abs(aqi_lag1 - aqi_lag7) / 2

    feature_vector = {
        'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2,
        'NOx': nox, 'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
        'Year': date.year, 'Month': month, 'Day': date.day,
        'DayOfWeek': date.dayofweek, 'Quarter': date.quarter,
        'Season_encoded': season_encoded,
        'AQI_lag_1': aqi_lag1, 'AQI_lag_3': aqi_lag3, 'AQI_lag_7': aqi_lag7,
        'PM2.5_lag_1': pm25 * 0.95, 'PM10_lag_1': pm10 * 0.95, 'NO2_lag_1': no2 * 0.95,
        'AQI_rolling_7d': rolling_7d, 'AQI_rolling_14d': rolling_14d,
        'AQI_rolling_30d': rolling_30d, 'AQI_rolling_7d_std': rolling_7d_std,
        'City_encoded': city_encoded
    }

    input_df = pd.DataFrame([feature_vector])[feature_cols]
    predicted_aqi = float(model.predict(input_df)[0])
    predicted_aqi = np.clip(predicted_aqi, 10, 500)

    category, color, bg_color, health_msg = get_aqi_info(predicted_aqi)
    city_stats = get_city_stats(selected_city)

    # ── Row 1: Prediction + City Stats ───────
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{color}40; background:{bg_color}; text-align:left; padding: 24px 28px;">
            <div class="metric-label">Predicted AQI — {selected_city} · {selected_date}</div>
            <div style="font-family:'Space Mono',monospace; font-size:3.5rem; font-weight:700; color:{color}; line-height:1;">
                {predicted_aqi:.0f}
            </div>
            <span class="aqi-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                {category}
            </span>
            <div style="margin-top:14px; font-size:13px; color:#c5c8d6; line-height:1.6;">
                {health_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">City Avg AQI</div>
            <div class="metric-value">{city_stats['mean']:.0f}</div>
            <div class="metric-unit">Historical mean</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Worst Month</div>
            <div class="metric-value">{MONTHS[city_stats['worst_month']-1]}</div>
            <div class="metric-unit">Highest avg AQI</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Month</div>
            <div class="metric-value">{MONTHS[city_stats['best_month']-1]}</div>
            <div class="metric-unit">Lowest avg AQI</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Charts ─────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">📅 Monthly AQI Trend — ' + selected_city + '</div>', unsafe_allow_html=True)
        city_df = df[df['City'] == selected_city]
        monthly = city_df.groupby('Month')['AQI'].mean()

        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor('#1a1d2e')
        ax.set_facecolor('#1a1d2e')

        bar_colors = []
        for v in monthly.values:
            _, c, _, _ = get_aqi_info(v)
            bar_colors.append(c)

        bars = ax.bar(monthly.index, monthly.values, color=bar_colors,
                      edgecolor='#0f1117', linewidth=1.5, width=0.7)
        ax.plot(monthly.index, monthly.values, color='white',
                linewidth=1.5, alpha=0.5, marker='o', markersize=4)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS, color='#8b8fa8', fontsize=9)
        ax.tick_params(axis='y', colors='#8b8fa8', labelsize=9)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.yaxis.grid(True, color='#2a2d3e', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel('Avg AQI', color='#8b8fa8', fontsize=9)

        # Mark predicted month
        ax.axvline(month, color='white', linewidth=1.5,
                   linestyle='--', alpha=0.6, label=f'Selected month ({MONTHS[month-1]})')
        ax.legend(facecolor='#1a1d2e', edgecolor='#2a2d3e',
                  labelcolor='#c5c8d6', fontsize=9)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">🏙️ Top 10 Cities by AQI</div>', unsafe_allow_html=True)
        top_cities = df.groupby('City')['AQI'].mean().sort_values(ascending=True).tail(10)

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor('#1a1d2e')
        ax2.set_facecolor('#1a1d2e')

        bar_cols2 = ['#4f7cff' if c == selected_city else '#2a3560' for c in top_cities.index]
        bars2 = ax2.barh(top_cities.index, top_cities.values,
                         color=bar_cols2, edgecolor='#0f1117', linewidth=1)

        ax2.tick_params(axis='x', colors='#8b8fa8', labelsize=8)
        ax2.tick_params(axis='y', colors='#c5c8d6', labelsize=8)
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.xaxis.grid(True, color='#2a2d3e', linewidth=0.8)
        ax2.set_axisbelow(True)
        ax2.set_xlabel('Avg AQI', color='#8b8fa8', fontsize=8)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig2)
        plt.close()

    # ── Row 3: AQI Scale + Model Info ────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_x, col_y = st.columns([3, 2])

    with col_x:
        st.markdown('<div class="section-header">🎚️ AQI Scale Reference</div>', unsafe_allow_html=True)
        scale_data = [
            ("0–50",   "Good",         "#00c46a"),
            ("51–100", "Satisfactory", "#a8e063"),
            ("101–200","Moderate",     "#ffcc00"),
            ("201–300","Poor",         "#ff8c00"),
            ("301–400","Very Poor",    "#ff4444"),
            ("401–500","Severe",       "#9b30ff"),
        ]
        fig3, ax3 = plt.subplots(figsize=(9, 1.1))
        fig3.patch.set_facecolor('#1a1d2e')
        ax3.set_facecolor('#1a1d2e')

        for i, (rng, label, clr) in enumerate(scale_data):
            ax3.barh(0, 1, left=i, color=clr, edgecolor='#0f1117', linewidth=2, height=0.7)
            ax3.text(i + 0.5, 0.55, label, ha='center', va='bottom',
                     fontsize=8, color='white', fontweight='bold')
            ax3.text(i + 0.5, -0.55, rng, ha='center', va='top',
                     fontsize=7.5, color='#8b8fa8')

        # Pointer for predicted AQI
        pointer_x = min(predicted_aqi / 500 * 6, 5.95)
        ax3.annotate('', xy=(pointer_x, -0.35), xytext=(pointer_x, -0.05),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2))

        ax3.set_xlim(0, 6)
        ax3.set_ylim(-0.9, 1.1)
        ax3.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig3)
        plt.close()

    with col_y:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value" style="font-size:1.4rem; color:#00c46a;">0.9426</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value" style="font-size:1.4rem; color:#4f7cff;">17.80</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value" style="font-size:1.4rem; color:#ffcc00;">10.26</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value" style="font-size:1.4rem; color:#ff8c00;">27</div>
            </div>
        </div>
        <div class="info-box" style="margin-top:12px">
            Trained on 20,268 samples across 26 cities · 2015–2019 · Tested on 2020
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Please make sure `models/` and `data/` folders are in the same directory as `app.py`")
