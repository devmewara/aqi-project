import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AQI Estimation System — India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.hero-title { font-family:'IBM Plex Mono',monospace; font-size:2.1rem; font-weight:600; color:#f0f2f5; letter-spacing:-1px; line-height:1.1; }
.hero-sub   { font-size:13px; color:#6b7280; margin-top:5px; font-weight:300; }
.hero-tag   { display:inline-block; font-size:11px; padding:2px 10px; border:0.5px solid #374151; border-radius:20px; color:#9ca3af; margin-right:6px; margin-top:6px; }
.sec-label  { font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#4f7cff; margin-bottom:10px; }
.metric-card { background:#111827; border:0.5px solid #1f2937; border-radius:10px; padding:16px 18px; text-align:center; }
.metric-label { font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:#6b7280; margin-bottom:4px; }
.metric-value { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600; color:#f0f2f5; }
.metric-unit  { font-size:11px; color:#6b7280; margin-top:2px; }
.predict-card { border-radius:12px; padding:22px 26px; border:1px solid; }
.aqi-number   { font-family:'IBM Plex Mono',monospace; font-size:4rem; font-weight:600; line-height:1; }
.aqi-cat      { display:inline-block; padding:5px 16px; border-radius:20px; font-size:12px; font-weight:500; letter-spacing:1px; text-transform:uppercase; margin-top:6px; border:1px solid; }
.info-box     { background:#0a1628; border-left:3px solid #3b82f6; padding:10px 14px; border-radius:0 8px 8px 0; font-size:12px; color:#60a5fa; margin:8px 0; line-height:1.6; }
.warn-box     { background:#1c1400; border-left:3px solid #d97706; padding:10px 14px; border-radius:0 8px 8px 0; font-size:12px; color:#d97706; margin:8px 0; line-height:1.6; }
.fest-box     { background:#1a0a2a; border-left:3px solid #a855f7; padding:10px 14px; border-radius:0 8px 8px 0; font-size:12px; color:#c084fc; margin:8px 0; line-height:1.6; }
.health-card  { background:#111827; border:0.5px solid #1f2937; border-radius:10px; padding:14px 16px; margin-top:10px; }
.health-title { font-size:12px; font-weight:500; color:#f0f2f5; margin-bottom:8px; }
.health-row   { font-size:12px; color:#9ca3af; padding:4px 0; border-bottom:0.5px solid #1f2937; line-height:1.6; }
.health-row:last-child { border-bottom:none; }
.driver-label { font-size:12px; color:#9ca3af; margin-bottom:3px; display:flex; justify-content:space-between; }
.driver-bar-bg   { background:#1f2937; border-radius:4px; height:7px; }
.driver-bar-fill { height:7px; border-radius:4px; }
.tab-content  { padding:16px 0; }
.timestamp    { font-size:11px; color:#4b5563; margin-top:6px; font-family:'IBM Plex Mono',monospace; }
div[data-testid="stSidebar"] { background:#0d1117 !important; }
hr { border-color:#1f2937 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    rf_v1        = joblib.load('models/random_forest.pkl')
    feat_v1      = joblib.load('models/feature_cols.pkl')
    city_mapping = joblib.load('models/city_mapping.pkl')
    try:
        rf_v2   = joblib.load('models/random_forest_v2.pkl')
        feat_v2 = joblib.load('models/feature_cols_v2.pkl')
    except:
        rf_v2, feat_v2 = rf_v1, feat_v1
    try:
        shap_explainer = joblib.load('models/shap_explainer.pkl')
        shap_vals      = joblib.load('models/shap_values_sample.pkl')
        X_shap         = joblib.load('models/X_shap_sample.pkl')
    except:
        shap_explainer = shap_vals = X_shap = None
    return rf_v1, feat_v1, rf_v2, feat_v2, city_mapping, shap_explainer, shap_vals, X_shap

@st.cache_data
def load_data():
    df = pd.read_csv('data/featured_aqi.csv', parse_dates=['Date'])
    return df.sort_values(['City','Date']).reset_index(drop=True)

@st.cache_data
def load_multicity():
    try:
        return pd.read_csv('data/multicity_results.csv')
    except:
        return None

@st.cache_data
def compute_city_monthly_avg(df):
    cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']
    return df.groupby(['City','Month'])[cols].mean().round(2)

try:
    rf_v1, feat_v1, rf_v2, feat_v2, city_mapping, shap_explainer, shap_vals, X_shap = load_models()
    df             = load_data()
    multicity_df   = load_multicity()
    city_monthly   = compute_city_monthly_avg(df)
    APP_OK         = True
except Exception as e:
    st.error(f"Could not load files. Make sure models/ and data/ are next to app.py\n\n{e}")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
AQI_LEVELS = [
    (50,  "Good",         "#00c46a", "#022b16", ["Children, elderly"],                             ["All activities safe"],           "No mask"),
    (100, "Satisfactory", "#84cc16", "#1a2e02", ["Severe respiratory patients"],                   ["Prolonged exertion (sensitive)"],"Optional (sensitive)"),
    (200, "Moderate",     "#facc15", "#2a1f00", ["Asthma, children, elderly"],                     ["Prolonged outdoor activity"],    "Recommended (sensitive)"),
    (300, "Poor",         "#f97316", "#2a1000", ["Everyone, especially children"],                 ["Outdoor exertion for everyone"], "Recommended"),
    (400, "Very Poor",    "#ef4444", "#2a0000", ["Entire population"],                             ["Any outdoor activity"],          "N95 mandatory"),
    (500, "Severe",       "#a855f7", "#1a0028", ["Entire population — emergency"],                 ["All outdoor exposure"],          "N95 + minimize movement"),
]

def get_aqi_info(aqi):
    for cap, cat, color, bg, risk, avoid, mask in AQI_LEVELS:
        if aqi <= cap:
            return cat, color, bg, risk, avoid, mask
    return AQI_LEVELS[-1][1:]

MONTHS      = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MAP  = {12:3,1:3,2:3,3:1,4:1,5:1,6:0,7:0,8:0,9:0,10:2,11:2}

FESTIVAL_DATES = {
    pd.Timestamp('2015-11-11'):'Diwali', pd.Timestamp('2016-10-30'):'Diwali',
    pd.Timestamp('2017-10-19'):'Diwali', pd.Timestamp('2018-11-07'):'Diwali',
    pd.Timestamp('2019-10-27'):'Diwali', pd.Timestamp('2020-11-14'):'Diwali',
    pd.Timestamp('2015-03-06'):'Holi',   pd.Timestamp('2016-03-24'):'Holi',
    pd.Timestamp('2017-03-13'):'Holi',   pd.Timestamp('2018-03-02'):'Holi',
    pd.Timestamp('2019-03-21'):'Holi',   pd.Timestamp('2020-03-10'):'Holi',
    pd.Timestamp('2016-01-01'):'New Year',pd.Timestamp('2017-01-01'):'New Year',
    pd.Timestamp('2018-01-01'):'New Year',pd.Timestamp('2019-01-01'):'New Year',
    pd.Timestamp('2020-01-01'):'New Year',
}

def check_festival(date):
    ts = pd.Timestamp(date)
    for fd, name in FESTIVAL_DATES.items():
        if abs((ts - fd).days) <= 2:
            return name, abs((ts - fd).days)
    return None, None

def get_historical_features(city, target_date):
    target_date = pd.Timestamp(target_date)
    city_df     = df[df['City'] == city]
    lag_cols    = ['AQI_lag_1','AQI_lag_3','AQI_lag_7','PM2.5_lag_1','PM10_lag_1','NO2_lag_1']
    roll_cols   = ['AQI_rolling_7d','AQI_rolling_14d','AQI_rolling_30d','AQI_rolling_7d_std']
    row = city_df[city_df['Date'] == target_date]
    if not row.empty:
        vals = row.iloc[0]
        return {c: vals[c] for c in lag_cols + roll_cols}, "exact"
    month   = target_date.month
    mdf     = city_df[city_df['Month'] == month]
    if mdf.empty: mdf = city_df
    result  = {}
    for c in lag_cols + roll_cols:
        result[c] = mdf[c].median() if c in mdf.columns else 150.0
    return result, "fallback"

def get_secondary_defaults(city, month):
    try:
        return city_monthly.loc[(city, month)].to_dict()
    except:
        return {'NO':15,'NOx':50,'NH3':25,'CO':1.0,'SO2':15,'O3':40,'PM2.5':80,'PM10':140,'NO2':40}

def get_yoy(city, target_date):
    target_date = pd.Timestamp(target_date)
    prev_date   = target_date.replace(year=target_date.year - 1)
    city_df     = df[df['City'] == city]
    row = city_df[city_df['Date'] == prev_date]
    if not row.empty:
        return row.iloc[0]['AQI'], prev_date.strftime("%d %b %Y")
    mdf = city_df[(city_df['Year']==target_date.year-1)&(city_df['Month']==target_date.month)]
    if not mdf.empty:
        return mdf['AQI'].mean(), f"{MONTHS[target_date.month-1]} {target_date.year-1} avg"
    return None, None

def get_warnings(pm25, pm10, no2, date):
    w = []
    if pm25 > 200: w.append("PM2.5 is critically high — severe pollution conditions.")
    if pm25 < 5 and pm10 < 10: w.append("Pollutants near zero — idealized scenario.")
    if pd.Timestamp(date).year < 2015 or pd.Timestamp(date).year > 2023:
        w.append(f"Date outside training range (2015–2020). Prediction is an extrapolation.")
    if pm10 < pm25: w.append("PM10 < PM2.5 — physically unusual. Please verify inputs.")
    return w

def get_festival_features(date):
    ts  = pd.Timestamp(date)
    is_fest = 0
    is_win  = 0
    for fd in FESTIVAL_DATES:
        delta = (ts - fd).days
        if delta == 0: is_fest = 1
        if 0 <= delta <= 2: is_win = 1
    return is_fest, is_win

def get_shap_drivers(input_dict, model, feature_list, top_n=5):
    importances = dict(zip(feature_list, model.feature_importances_))
    contributions = {}
    for feat, val in input_dict.items():
        if feat in importances:
            contributions[feat] = abs(float(val)) * importances[feat]
    total = sum(contributions.values()) or 1
    sorted_c = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return [(k, round(v/total*100, 1)) for k,v in sorted_c[:top_n]]

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sec-label">Location & Date</div>', unsafe_allow_html=True)
    cities         = sorted(df['City'].unique())
    selected_city  = st.selectbox("City", cities,
                                  index=cities.index("Delhi") if "Delhi" in cities else 0)
    selected_date  = st.date_input("Date", value=pd.Timestamp("2019-11-15"),
                                   min_value=pd.Timestamp("2015-01-01"),
                                   max_value=pd.Timestamp("2023-12-31"))

    st.markdown("---")
    st.markdown('<div class="sec-label">Model Version</div>', unsafe_allow_html=True)
    model_choice = st.radio("",
        ["V2 — Festival Enhanced (33 features)", "V1 — Original (26 features)"],
        index=0)
    use_v2 = "V2" in model_choice
    active_model    = rf_v2   if use_v2 else rf_v1
    active_features = feat_v2 if use_v2 else feat_v1

    st.markdown("---")
    st.markdown('<div class="sec-label">Primary Pollutants</div>', unsafe_allow_html=True)
    month  = pd.Timestamp(selected_date).month
    sec    = get_secondary_defaults(selected_city, month)
    pm25   = st.slider("PM2.5 (μg/m³)", 0.0, 250.0, float(sec.get('PM2.5',85.0)), 1.0)
    pm10   = st.slider("PM10  (μg/m³)", 0.0, 470.0, float(sec.get('PM10',140.0)), 1.0)
    no2    = st.slider("NO₂   (μg/m³)", 0.0, 120.0, float(sec.get('NO2',45.0)),   0.5)

    with st.expander("Advanced: Other pollutants (auto-filled)"):
        no  = st.slider("NO  (μg/m³)", 0.0, 65.0,  float(sec.get('NO',15.0)),  0.5)
        nox = st.slider("NOx (μg/m³)", 0.0, 125.0, float(sec.get('NOx',50.0)), 0.5)
        nh3 = st.slider("NH₃ (μg/m³)", 0.0, 135.0, float(sec.get('NH3',25.0)), 0.5)
        co  = st.slider("CO  (mg/m³)", 0.0, 4.5,   float(sec.get('CO',1.0)),   0.1)
        so2 = st.slider("SO₂ (μg/m³)", 0.0, 45.0,  float(sec.get('SO2',15.0)), 0.5)
        o3  = st.slider("O₃  (μg/m³)", 0.0, 130.0, float(sec.get('O3',40.0)),  0.5)

    st.markdown("---")
    predict_btn = st.button("Estimate AQI", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-title">AQI Estimation System</div>
<div class="hero-sub">Condition-based air quality estimation · 26 Indian cities · 2015–2020</div>
<div style="margin-top:8px">
  <span class="hero-tag">Random Forest V2</span>
  <span class="hero-tag">R² = 0.9426</span>
  <span class="hero-tag">RMSE = 12.35</span>
  <span class="hero-tag">33 Features</span>
  <span class="hero-tag">Festival-aware</span>
  <span class="hero-tag">SHAP Explainable</span>
</div>
<hr style="margin:18px 0 20px">
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🎯  AQI Estimator",
    "🏙️  City Comparison",
    "📊  SHAP Explainability"
])

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if 'prediction'  not in st.session_state: st.session_state.prediction  = None
if 'pred_meta'   not in st.session_state: st.session_state.pred_meta   = {}
if 'pred_time'   not in st.session_state: st.session_state.pred_time   = None

# ══════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════
if predict_btn:
    with st.spinner("Fetching historical context and estimating AQI..."):
        hist, hist_status = get_historical_features(selected_city, selected_date)
        date_ts           = pd.Timestamp(selected_date)
        is_fest, is_win   = get_festival_features(selected_date)
        fest_name, _      = check_festival(selected_date)

        # Momentum approximation
        momentum = hist.get('AQI_lag_1',150) - hist.get('AQI_lag_7',150)
        city_enc = city_mapping.get(selected_city, 0)
        season   = SEASON_MAP[date_ts.month]

        base_vec = {
            'PM2.5':pm25,'PM10':pm10,'NO':no,'NO2':no2,
            'NOx':nox,'NH3':nh3,'CO':co,'SO2':so2,'O3':o3,
            'Year':date_ts.year,'Month':date_ts.month,'Day':date_ts.day,
            'DayOfWeek':date_ts.dayofweek,'Quarter':date_ts.quarter,
            'Season_encoded':season,'City_encoded':city_enc,
            **hist
        }

        # Extra features for V2
        extra_v2 = {
            'Is_Festival':is_fest,
            'Is_Festival_Window':is_win,
            'Is_Weekend':int(date_ts.dayofweek >= 5),
            'AQI_momentum':momentum,
            'Season_City_interaction':season * city_enc,
            'Month_sin':np.sin(2*np.pi*date_ts.month/12),
            'Month_cos':np.cos(2*np.pi*date_ts.month/12),
        }
        full_vec = {**base_vec, **extra_v2}

        # Pick correct features
        fvec    = full_vec if use_v2 else base_vec
        inp_df  = pd.DataFrame([fvec])[active_features]
        pred    = float(active_model.predict(inp_df)[0])
        pred    = np.clip(pred, 10, 500)

        drivers  = get_shap_drivers(fvec, active_model, active_features)
        prev_aqi, prev_label = get_yoy(selected_city, selected_date)
        warns    = get_warnings(pm25, pm10, no2, selected_date)

        st.session_state.prediction = pred
        st.session_state.pred_meta  = {
            'hist':hist,'hist_status':hist_status,
            'drivers':drivers,'prev_aqi':prev_aqi,'prev_label':prev_label,
            'warns':warns,'city':selected_city,'date':selected_date,
            'fest_name':fest_name,'is_fest':is_fest,'is_win':is_win,
            'model_ver':'V2' if use_v2 else 'V1',
        }
        st.session_state.pred_time = datetime.now().strftime("%H:%M:%S")

# ══════════════════════════════════════════════════════════════
#  TAB 1 — ESTIMATOR
# ══════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.prediction is not None:
        aqi  = st.session_state.prediction
        meta = st.session_state.pred_meta
        cat, color, bg, risk, avoid, mask_rec = get_aqi_info(aqi)

        # Warnings
        for w in meta['warns']:
            st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)

        # Festival detection
        if meta['is_win']:
            st.markdown(
                f'<div class="fest-box">🎆 <b>{meta["fest_name"]} period detected.</b> '
                f'Festival fireworks and bonfires typically increase AQI by 20–40%. '
                f'This has been accounted for in the V2 model.</div>',
                unsafe_allow_html=True)

        if meta['hist_status'] == 'exact':
            st.markdown(
                f'<div class="info-box" style="background:#021a0a;border-color:#16a34a;color:#4ade80">'
                f'✓ Real historical data found for {meta["city"]} on {meta["date"]}.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="info-box">ℹ Date not in dataset. Lag features auto-filled '
                f'from {meta["city"]} monthly averages.</div>',
                unsafe_allow_html=True)

        # ── Row 1: Prediction card + stats ──
        c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
        with c1:
            yoy_html = ""
            if meta['prev_aqi']:
                diff = aqi - meta['prev_aqi']
                pct  = abs(diff)/meta['prev_aqi']*100
                arr  = "↑" if diff > 0 else "↓"
                clr2 = "#ef4444" if diff > 0 else "#22c55e"
                yoy_html = f'<div style="margin-top:8px;font-size:13px;color:{clr2}">{arr} {pct:.1f}% vs {meta["prev_label"]} ({meta["prev_aqi"]:.0f})</div>'

            st.markdown(f"""
            <div class="predict-card" style="background:{bg};border-color:{color}44">
              <div class="metric-label">{meta['city']} · {meta['date']} · Model {meta['model_ver']}</div>
              <div class="aqi-number" style="color:{color}">{aqi:.0f}</div>
              <span class="aqi-cat" style="color:{color};border-color:{color}55;background:{color}18">{cat.upper()}</span>
              <div style="margin-top:10px;font-size:13px;color:#9ca3af;line-height:1.6">
                {'⚠️ Festival period — elevated AQI expected. ' if meta['is_win'] else ''}
                Risk: {', '.join(risk)}
              </div>
              {yoy_html}
              <div class="timestamp">Estimated {st.session_state.pred_time}</div>
            </div>""", unsafe_allow_html=True)

        city_aqi = df[df['City']==meta['city']]['AQI']
        w_month  = df[df['City']==meta['city']].groupby('Month')['AQI'].mean().idxmax()
        b_month  = df[df['City']==meta['city']].groupby('Month')['AQI'].mean().idxmin()

        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">City Avg</div><div class="metric-value">{city_aqi.mean():.0f}</div><div class="metric-unit">historical</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Worst Month</div><div class="metric-value">{MONTHS[w_month-1]}</div><div class="metric-unit">highest AQI</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Best Month</div><div class="metric-value">{MONTHS[b_month-1]}</div><div class="metric-unit">lowest AQI</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Drivers + Historical + Health ──
        ca, cb, cc = st.columns([2, 1.5, 1.5])

        with ca:
            st.markdown('<div class="sec-label">Prediction Breakdown</div>', unsafe_allow_html=True)
            drivers    = meta['drivers']
            top_feat   = drivers[0][0].replace('_',' ')
            sec_feat   = drivers[1][0].replace('_',' ') if len(drivers)>1 else ""
            if aqi > 300:
                expl = f"<b>{cat}</b> AQI primarily driven by <b>{top_feat}</b> ({drivers[0][1]}%) and <b>{sec_feat}</b>. Heavy pollution accumulation."
            elif aqi > 150:
                expl = f"Moderate-to-poor air driven by <b>{top_feat}</b> ({drivers[0][1]}%). Recent pollution trend is secondary."
            else:
                expl = f"Relatively clean. <b>{top_feat}</b> dominates at {drivers[0][1]}%, suggesting stable recent air quality."
            st.markdown(f'<div style="font-size:12px;color:#9ca3af;margin-bottom:10px;line-height:1.7">{expl}</div>', unsafe_allow_html=True)

            d_colors = [color,'#3b82f6','#8b5cf6','#06b6d4','#84cc16']
            for i,(feat,pct) in enumerate(drivers):
                label = feat.replace('_',' ')
                dc    = d_colors[i % len(d_colors)]
                st.markdown(f"""
                <div style="margin:6px 0">
                  <div class="driver-label"><span>{label}</span><span>{pct}%</span></div>
                  <div class="driver-bar-bg"><div class="driver-bar-fill" style="width:{pct}%;background:{dc}"></div></div>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:#4b5563;margin-top:6px">Weighted by RF feature importances</div>', unsafe_allow_html=True)

        with cb:
            st.markdown('<div class="sec-label">Historical Context</div>', unsafe_allow_html=True)
            h = meta['hist']
            st.markdown(f"""
            <div class="health-card">
              <div class="health-title">Real Lag Features Used</div>
              <div class="health-row">Yesterday AQI &nbsp;&nbsp;&nbsp;&nbsp; <b>{h.get('AQI_lag_1',0):.0f}</b></div>
              <div class="health-row">3 Days Ago &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{h.get('AQI_lag_3',0):.0f}</b></div>
              <div class="health-row">7 Days Ago &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{h.get('AQI_lag_7',0):.0f}</b></div>
              <div class="health-row">7-Day Rolling Avg &nbsp; <b>{h.get('AQI_rolling_7d',0):.0f}</b></div>
              <div class="health-row">30-Day Rolling Avg &nbsp;<b>{h.get('AQI_rolling_30d',0):.0f}</b></div>
            </div>""", unsafe_allow_html=True)
            if meta['prev_aqi']:
                diff = aqi - meta['prev_aqi']
                tc   = "#ef4444" if diff>0 else "#22c55e"
                tw   = "worse" if diff>0 else "better"
                st.markdown(f"""
                <div class="health-card" style="margin-top:10px">
                  <div class="metric-label">Year-over-Year</div>
                  <div style="font-size:12px;color:#9ca3af;margin-top:4px;line-height:1.7">
                    {meta['prev_label']}: <b style="color:#f0f2f5">{meta['prev_aqi']:.0f}</b><br>
                    Today estimate: <b style="color:{color}">{aqi:.0f}</b><br>
                    <span style="color:{tc}">{abs(diff):.0f} AQI points {tw}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

        with cc:
            st.markdown('<div class="sec-label">Health Advisory</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="health-card">
              <div class="health-title">At-risk groups</div>
              {''.join([f'<div class="health-row">{r}</div>' for r in risk])}
            </div>
            <div class="health-card">
              <div class="health-title">Avoid</div>
              {''.join([f'<div class="health-row">{a}</div>' for a in avoid])}
            </div>
            <div class="health-card">
              <div class="health-title">Mask</div>
              <div class="health-row" style="color:{color}">{mask_rec}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 3: Charts ──
        cp, cq = st.columns([3,2])
        with cp:
            st.markdown(f'<div class="sec-label">Monthly AQI — {meta["city"]}</div>', unsafe_allow_html=True)
            city_df = df[df['City']==meta['city']]
            monthly = city_df.groupby('Month')['AQI'].mean()
            sel_m   = pd.Timestamp(selected_date).month
            fig,ax  = plt.subplots(figsize=(9,3.2))
            fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
            bcs = []
            for v in monthly.values:
                c2,_,_,_,_,_ = get_aqi_info(v)
                lookup = {l[1]:l[2] for l in AQI_LEVELS}
                bcs.append(lookup.get(c2,'#4f7cff'))
            ax.bar(monthly.index, monthly.values, color=bcs, edgecolor='#111827', width=0.7, alpha=0.85)
            ax.plot(monthly.index, monthly.values, color='white', linewidth=1.2, alpha=0.4, marker='o', markersize=3)
            ax.axvline(sel_m, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.axhline(aqi,   color=color,   linewidth=1,   linestyle=':', alpha=0.8, label=f'Prediction {aqi:.0f}')
            ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS, color='#6b7280', fontsize=8)
            ax.tick_params(axis='y', colors='#6b7280', labelsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            ax.yaxis.grid(True, color='#1f2937', linewidth=0.8); ax.set_axisbelow(True)
            ax.legend(facecolor='#111827', edgecolor='#1f2937', labelcolor='#9ca3af', fontsize=8)
            plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

        with cq:
            st.markdown('<div class="sec-label">Top 10 Cities by AQI</div>', unsafe_allow_html=True)
            top10 = df.groupby('City')['AQI'].mean().sort_values(ascending=True).tail(10)
            fig2,ax2 = plt.subplots(figsize=(5,3.2))
            fig2.patch.set_facecolor('#111827'); ax2.set_facecolor('#111827')
            bc2 = [color if c==meta['city'] else '#1f2937' for c in top10.index]
            ax2.barh(top10.index, top10.values, color=bc2, edgecolor='#111827', height=0.65)
            ax2.tick_params(axis='x', colors='#6b7280', labelsize=8)
            ax2.tick_params(axis='y', colors='#d1d5db', labelsize=8)
            for s in ax2.spines.values(): s.set_visible(False)
            ax2.xaxis.grid(True, color='#1f2937', linewidth=0.8); ax2.set_axisbelow(True)
            plt.tight_layout(pad=0.4); st.pyplot(fig2); plt.close()

        # AQI Scale
        st.markdown('<div class="sec-label">AQI Scale Reference</div>', unsafe_allow_html=True)
        fig3,ax3 = plt.subplots(figsize=(12,0.9))
        fig3.patch.set_facecolor('#111827'); ax3.set_facecolor('#111827')
        scale = [(l[0],l[1],l[2]) for l in AQI_LEVELS]
        for i,(cap,cat_s,clr) in enumerate(scale):
            ax3.barh(0,1,left=i,color=clr,edgecolor='#111827',linewidth=2,height=0.65)
            ax3.text(i+0.5, 0.48, cat_s, ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
            rng = f"0–{cap}" if i==0 else f"{scale[i-1][0]+1}–{cap}"
            ax3.text(i+0.5,-0.48, rng, ha='center', va='top', fontsize=7, color='#6b7280')
        ptr_x = min(aqi/500*6, 5.92)
        ax3.annotate('', xy=(ptr_x,-0.3), xytext=(ptr_x,-0.05),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax3.set_xlim(0,6); ax3.set_ylim(-0.8,0.9); ax3.axis('off')
        plt.tight_layout(pad=0); st.pyplot(fig3); plt.close()

        # Model info
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        for col,label,val in zip([m1,m2,m3,m4,m5],
            ["R² Score","RMSE (V2)","MAE","Features","Cities"],
            ["0.9426","12.35","~10","33","26"]):
            col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.3rem">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("""<div class="info-box" style="margin-top:12px">
        <b>About:</b> Condition-based AQI estimation using Random Forest V2 trained on 24,000+ daily observations
        across 26 Indian cities (2015–2020). Festival indicators and AQI momentum features added in Phase 2
        improved RMSE by 19.92%. Not a future forecast — a condition-based estimation tool.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#4b5563">
          <div style="font-size:48px;margin-bottom:16px">🌫️</div>
          <div style="font-size:16px;font-weight:500;color:#9ca3af;margin-bottom:8px">
            Configure inputs in the sidebar and click <b style="color:#f0f2f5">Estimate AQI</b>
          </div>
          <div style="font-size:13px;max-width:480px;margin:0 auto;line-height:1.8">
            Historical lag features are loaded automatically from the dataset.
            Only PM2.5, PM10 and NO₂ require manual input.
            Festival periods are detected and flagged automatically.
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — CITY COMPARISON
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">Multi-City Model Performance — 22 Cities Evaluated</div>', unsafe_allow_html=True)

    if multicity_df is not None:
        c1, c2, c3 = st.columns(3)
        best  = multicity_df.loc[multicity_df['RMSE'].idxmin()]
        worst = multicity_df.loc[multicity_df['RMSE'].idxmax()]
        above90 = (multicity_df['R2'] >= 0.90).sum()

        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Best Predicted City</div><div class="metric-value" style="font-size:1.3rem;color:#00c46a">{best["City"]}</div><div class="metric-unit">RMSE={best["RMSE"]:.1f}  R²={best["R2"]:.4f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Hardest to Predict</div><div class="metric-value" style="font-size:1.3rem;color:#ef4444">{worst["City"]}</div><div class="metric-unit">RMSE={worst["RMSE"]:.1f}  R²={worst["R2"]:.4f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Cities with R² ≥ 0.90</div><div class="metric-value" style="color:#facc15">{above90}</div><div class="metric-unit">out of {len(multicity_df)} evaluated</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="sec-label">RMSE Ranking — All Cities</div>', unsafe_allow_html=True)
            sorted_df = multicity_df.sort_values('RMSE', ascending=True)
            fig,ax = plt.subplots(figsize=(8,7))
            fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
            clrs = ['#1a9850' if r<20 else '#fee08b' if r<35 else '#f97316' if r<55 else '#d73027'
                    for r in sorted_df['RMSE']]
            bars = ax.barh(sorted_df['City'], sorted_df['RMSE'], color=clrs, edgecolor='#111827', height=0.7)
            for bar,val in zip(bars, sorted_df['RMSE']):
                ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8, color='#d1d5db')
            ax.axvline(multicity_df['RMSE'].mean(), color='white', linewidth=1.2, linestyle='--', alpha=0.5)
            ax.set_xlabel('RMSE', color='#6b7280', fontsize=9)
            ax.tick_params(axis='x', colors='#6b7280', labelsize=8)
            ax.tick_params(axis='y', colors='#d1d5db', labelsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.grid(True, color='#1f2937', linewidth=0.8); ax.set_axisbelow(True)
            plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

        with col_b:
            st.markdown('<div class="sec-label">R² Score Ranking</div>', unsafe_allow_html=True)
            sorted_r2 = multicity_df.sort_values('R2', ascending=True)
            fig2,ax2 = plt.subplots(figsize=(8,7))
            fig2.patch.set_facecolor('#111827'); ax2.set_facecolor('#111827')
            r2_clrs = ['#1a9850' if r>0.90 else '#fee08b' if r>0.80 else '#f97316' if r>0.60 else '#d73027'
                       for r in sorted_r2['R2']]
            bars2 = ax2.barh(sorted_r2['City'], sorted_r2['R2'], color=r2_clrs, edgecolor='#111827', height=0.7)
            for bar,val in zip(bars2, sorted_r2['R2']):
                ax2.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8, color='#d1d5db')
            ax2.axvline(0.90, color='white', linewidth=1.2, linestyle='--', alpha=0.5, label='R²=0.90')
            ax2.set_xlim(0,1.08)
            ax2.set_xlabel('R² Score', color='#6b7280', fontsize=9)
            ax2.tick_params(axis='x', colors='#6b7280', labelsize=8)
            ax2.tick_params(axis='y', colors='#d1d5db', labelsize=8)
            for s in ax2.spines.values(): s.set_visible(False)
            ax2.xaxis.grid(True, color='#1f2937', linewidth=0.8); ax2.set_axisbelow(True)
            ax2.legend(facecolor='#111827', edgecolor='#1f2937', labelcolor='#9ca3af', fontsize=8)
            plt.tight_layout(pad=0.4); st.pyplot(fig2); plt.close()

        st.markdown("""<div class="info-box">
        <b>Why do some cities predict better?</b> Cities with stable, consistent pollution patterns
        (Hyderabad, Mumbai, Delhi) are highly predictable. Cities with irregular industrial activity
        or agricultural burning (Amritsar — Parali season) show higher RMSE because the model
        hasn't seen those specific pollution spikes during training.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:16px">Full Results Table</div>', unsafe_allow_html=True)
        display_df = multicity_df[['City','RMSE','MAE','R2','Mean_AQI']].sort_values('R2', ascending=False)
        st.dataframe(display_df.style.background_gradient(subset=['R2'], cmap='RdYlGn')
                                     .background_gradient(subset=['RMSE'], cmap='RdYlGn_r')
                                     .format({'RMSE':'{:.2f}','MAE':'{:.2f}','R2':'{:.4f}','Mean_AQI':'{:.1f}'}),
                     use_container_width=True, height=400)
    else:
        st.info("Run `07_multicity_analysis.ipynb` first to generate multi-city results.")

# ══════════════════════════════════════════════════════════════
#  TAB 3 — SHAP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">SHAP Explainability — Global Feature Impact</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    SHAP (SHapley Additive exPlanations) shows how much each feature pushes AQI predictions
    up or down across 500 test samples. Unlike simple feature importance, SHAP reveals
    both direction and magnitude of each feature's contribution.
    </div>""", unsafe_allow_html=True)

    if shap_vals is not None and X_shap is not None:
        mean_shap = np.abs(shap_vals).mean(axis=0)
        shap_imp  = pd.Series(mean_shap, index=feat_v1).sort_values(ascending=True).tail(15)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-label">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(8,6))
            fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
            clrs_s = plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(shap_imp)))
            bars = ax.barh(shap_imp.index, shap_imp.values, color=clrs_s, edgecolor='#111827', height=0.7)
            for bar,val in zip(bars,shap_imp.values):
                ax.text(val+0.2, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8, color='#d1d5db')
            ax.set_xlabel('Mean |SHAP Value|', color='#6b7280', fontsize=9)
            ax.tick_params(axis='x', colors='#6b7280', labelsize=8)
            ax.tick_params(axis='y', colors='#d1d5db', labelsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.grid(True, color='#1f2937', linewidth=0.8); ax.set_axisbelow(True)
            plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

        with col2:
            st.markdown('<div class="sec-label">Top 5 Findings</div>', unsafe_allow_html=True)
            top5 = shap_imp.tail(5).iloc[::-1]
            for i,(feat,val) in enumerate(top5.items(), 1):
                pct = val / shap_imp.sum() * 100
                st.markdown(f"""
                <div style="background:#111827;border:0.5px solid #1f2937;border-radius:8px;padding:10px 14px;margin-bottom:8px">
                  <div style="font-size:12px;font-weight:500;color:#f0f2f5">#{i} {feat.replace('_',' ')}</div>
                  <div style="font-size:11px;color:#6b7280;margin-top:2px">
                    Avg impact: <b style="color:#facc15">{val:.1f} AQI units</b> · {pct:.1f}% of total
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Key Insight</div>', unsafe_allow_html=True)
        top_feat_name = shap_imp.index[-1].replace('_',' ')
        top_feat_val  = shap_imp.values[-1]
        second_name   = shap_imp.index[-2].replace('_',' ')
        st.markdown(f"""<div class="info-box">
        <b>{top_feat_name}</b> has the highest global SHAP importance ({top_feat_val:.1f} AQI units average impact),
        confirming that recent pollution history is the strongest predictor of today's AQI.
        <b>{second_name}</b> is the strongest physical driver.
        This validates our spatio-temporal feature engineering approach — temporal memory
        features outperform raw pollutant readings in predictive power.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:16px">Saved SHAP Visualizations</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        try:
            from PIL import Image
            with sc1:
                img = Image.open('visualizations/14_shap_beeswarm.png')
                st.image(img, caption='SHAP Beeswarm — Feature Impact Distribution', use_column_width=True)
            with sc2:
                img2 = Image.open('visualizations/15_shap_waterfall.png')
                st.image(img2, caption='SHAP Waterfall — Single Prediction Explained', use_column_width=True)
        except:
            st.info("Visualization images will appear here after running the SHAP notebook.")
    else:
        st.info("Run `06_shap_explainability.ipynb` first to generate SHAP values.")
