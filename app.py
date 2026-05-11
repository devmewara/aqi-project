import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Desh Ki Hawa — AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.main, .stApp { background: #07090F; }
.block-container { padding: 1rem 2rem 3rem !important; max-width: 1300px; }

/* Hide Streamlit's own top header to avoid double header */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0C0F19 !important;
    border-right: 1px solid #151929;
}
section[data-testid="stSidebar"] * { color: #E2E8FF !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #636B8A !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px;
}
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 14px !important; padding: 14px !important;
    width: 100% !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px #4F46E540 !important;
    transition: all 0.2s !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 8px 32px #4F46E560 !important;
    transform: translateY(-2px) !important;
}

/* ── Top nav bar ── */
.topbar {
    background: linear-gradient(90deg, #0C0F19 0%, #0F1221 100%);

    padding: 16px 28px;
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    border-radius: 14px;
    border: 1px solid #151929;

}
.topbar-brand {
    font-size: 20px; font-weight: 800;
    background: linear-gradient(135deg, #818CF8, #C084FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.topbar-tags { display: flex; gap: 8px; }
.topbar-tag {
    font-size: 10px; padding: 4px 10px;
    border: 1px solid #1E2440;
    border-radius: 20px; color: #4B5280;
    background: #0C0F19; letter-spacing: 0.5px;
}

/* ── Dashboard grid ── */
.dash-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}
.dash-grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}
.dash-grid-wide {
    display: grid;
    grid-template-columns: 1.6fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}

/* ── Cards ── */
.card {
    background: #0C0F19;
    border: 1px solid #151929;
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.card:hover { border-color: #1E2440; }
.card-glow::after {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--glow) 0%, transparent 70%);
    opacity: 0.06; pointer-events: none;
}
.card-label {
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3D4468; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.card-label-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}

/* ── AQI display ── */
.aqi-display {
    border-radius: 20px;
    padding: 32px 36px;
    border: 1px solid;
    position: relative; overflow: hidden;
    margin-bottom: 0;
}
.aqi-city-date {
    font-size: 11px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; opacity: 0.5; margin-bottom: 12px;
}
.aqi-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 96px; font-weight: 500; line-height: 1;
    margin-bottom: 10px;
}
.aqi-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 18px; border-radius: 10px;
    font-size: 13px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    border: 1px solid; margin-bottom: 18px;
}
.aqi-message { font-size: 14px; opacity: 0.6; line-height: 1.6; max-width: 360px; }
.aqi-time { font-size: 11px; opacity: 0.25; margin-top: 14px; font-family: 'JetBrains Mono', monospace; }

/* ── Stat mini ── */
.stat-mini {
    background: #0C0F19; border: 1px solid #151929;
    border-radius: 14px; padding: 18px 20px;
    text-align: center; height: 100%;
}
.stat-mini-lbl {
    font-size: 10px; color: #2D3355; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 8px;
}
.stat-mini-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem; font-weight: 500; color: #E2E8FF;
    margin-bottom: 2px;
}
.stat-mini-sub { font-size: 11px; color: #2D3355; }

/* ── Driver bars ── */
.drv { margin: 10px 0; }
.drv-top {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #4B5280; margin-bottom: 5px;
}
.drv-top-val { color: #9BA3C8; font-family: 'JetBrains Mono', monospace; }
.drv-bg { background: #12162A; border-radius: 6px; height: 6px; }
.drv-fill { height: 6px; border-radius: 6px; }

/* ── Health rows ── */
.health-blk {
    background: #0A0D16; border: 1px solid #151929;
    border-radius: 12px; padding: 14px 16px; margin-top: 10px;
}
.health-lbl {
    font-size: 10px; color: #2D3355; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 10px;
}
.health-row {
    font-size: 12px; color: #636B8A; padding: 6px 0;
    border-bottom: 1px solid #0F1221; line-height: 1.6;
    display: flex; gap: 8px; align-items: flex-start;
}
.health-row:last-child { border-bottom: none; }
.h-dot { width: 5px; height: 5px; border-radius: 50%; margin-top: 7px; flex-shrink: 0; }

/* ── Lag rows ── */
.lag-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 8px 0;
    border-bottom: 1px solid #0F1221; font-size: 12px;
}
.lag-row:last-child { border-bottom: none; }
.lag-k { color: #3D4468; }
.lag-v { font-family: 'JetBrains Mono', monospace; color: #E2E8FF; font-size: 13px; }

/* ── Notification boxes ── */
.nb {
    padding: 12px 16px; border-radius: 10px;
    font-size: 12px; line-height: 1.65;
    margin: 10px 0; border-left: 3px solid;
    border-top: 1px solid; border-right: 1px solid; border-bottom: 1px solid;
    border-top-left-radius: 0; border-bottom-left-radius: 0;
}
.nb-info    { background:#070E1C; border-color:#1D4ED8; color:#93C5FD; border-top-color:#1D4ED820; border-right-color:#1D4ED820; border-bottom-color:#1D4ED820; }
.nb-warn    { background:#0F0900; border-color:#B45309; color:#FCD34D; border-top-color:#B4530920; border-right-color:#B4530920; border-bottom-color:#B4530920; }
.nb-success { background:#030F07; border-color:#059669; color:#34D399; border-top-color:#05966920; border-right-color:#05966920; border-bottom-color:#05966920; }
.nb-fest    { background:#0A0514; border-color:#7C3AED; color:#C4B5FD; border-top-color:#7C3AED20; border-right-color:#7C3AED20; border-bottom-color:#7C3AED20; }

/* ── Section title ── */
.sec-title {
    font-size: 10px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #252A42; margin-bottom: 16px;
    padding-bottom: 12px; border-bottom: 1px solid #0F1221;
}

/* ── YoY ── */
.yoy-row { font-size: 13px; margin-top: 14px; }

/* ── Model badge row ── */
.model-info-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;
}
.model-chip {
    font-size: 11px; padding: 4px 12px;
    border: 1px solid #1E2440;
    border-radius: 20px; color: #4B5280; background: #0C0F19;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 100px 20px;
}
.empty-icon { font-size: 72px; opacity: 0.07; margin-bottom: 20px; }
.empty-title {
    font-size: 22px; font-weight: 700;
    color: #1A1F36; margin-bottom: 8px;
}
.empty-sub { font-size: 14px; color: #252A42; max-width: 400px; margin: 0 auto; line-height: 1.7; }

/* ── Tabs ── */
div[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #151929 !important;
    gap: 0 !important;
}
div[data-testid="stTabs"] button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #3D4468 !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom: 2px solid #818CF8 !important;
}

/* ── Streamlit overrides ── */
div[data-testid="stSelectbox"] > div > div {
    background: #0C0F19 !important; border: 1px solid #1E2440 !important;
    border-radius: 10px !important; color: #E2E8FF !important;
}
div[data-testid="stDateInput"] input {
    background: #0C0F19 !important; border: 1px solid #1E2440 !important;
    border-radius: 10px !important; color: #E2E8FF !important;
}
hr { border-color: #151929 !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    m    = joblib.load('models/random_forest_v2.pkl')
    feat = joblib.load('models/feature_cols_v2.pkl')
    cmap = joblib.load('models/city_mapping.pkl')
    try:
        sv  = joblib.load('models/shap_values_sample.pkl')
        fv1 = joblib.load('models/feature_cols.pkl')
        xs  = joblib.load('models/X_shap_sample.pkl')
    except:
        sv = fv1 = xs = None
    return m, feat, cmap, sv, fv1, xs

@st.cache_data
def load_data():
    df = pd.read_csv('data/featured_aqi_v2.csv', parse_dates=['Date'])
    return df.sort_values(['City','Date']).reset_index(drop=True)

@st.cache_data
def load_mc():
    try:    return pd.read_csv('data/multicity_results.csv')
    except: return None

@st.cache_data
def city_monthly_avg(df):
    cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']
    return df.groupby(['City','Month'])[cols].mean().round(2)

try:
    model, feat, cmap, shap_vals, feat_v1, X_shap = load_models()
    df    = load_data()
    mc_df = load_mc()
    cma   = city_monthly_avg(df)
except Exception as e:
    st.error(f"Could not load files: {e}"); st.stop()


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
AQI_CFG = [
    (50,  "Good",        "#10B981","#071811","#34D399","💚",
     "Air quality is excellent. Safe for all outdoor activities."),
    (100, "Satisfactory","#84CC16","#0D1A03","#BEF264","🟡",
     "Air quality is acceptable. Sensitive groups should take precautions."),
    (200, "Moderate",    "#F59E0B","#1A1200","#FDE68A","🟠",
     "Sensitive individuals may experience health effects. Limit prolonged outdoor exertion."),
    (300, "Poor",        "#F97316","#1A0C00","#FDBA74","🔴",
     "Everyone may experience health effects. Avoid outdoor physical activity."),
    (400, "Very Poor",   "#EF4444","#1A0505","#FCA5A5","🔴",
     "Health alert. Entire population affected. Stay indoors, keep windows closed."),
    (500, "Severe",      "#A855F7","#12082A","#D8B4FE","🟣",
     "Emergency conditions. Do not go outdoors. Use N95 mask if movement is necessary."),
]
HEALTH = {
    "Good":        (["Safe for everyone"],             ["Nothing — enjoy the clean air!"],  "No mask needed 🌿"),
    "Satisfactory":(["Severe respiratory patients"],  ["Prolonged outdoor exertion"],       "Optional for sensitive groups"),
    "Moderate":    (["Asthma, children, elderly"],    ["Prolonged outdoor activity"],       "Recommended for sensitive groups"),
    "Poor":        (["Everyone, esp. children"],      ["All outdoor physical exertion"],    "Recommended for everyone"),
    "Very Poor":   (["Entire population"],            ["Any outdoor activity"],             "N95 mandatory outdoors"),
    "Severe":      (["Entire population — emergency"],["All outdoor exposure"],             "N95 mandatory, stay indoors"),
}
MONTHS     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MAP = {12:3,1:3,2:3,3:1,4:1,5:1,6:0,7:0,8:0,9:0,10:2,11:2}
FESTIVALS  = {
    pd.Timestamp('2015-11-11'):'Diwali',  pd.Timestamp('2016-10-30'):'Diwali',
    pd.Timestamp('2017-10-19'):'Diwali',  pd.Timestamp('2018-11-07'):'Diwali',
    pd.Timestamp('2019-10-27'):'Diwali',  pd.Timestamp('2020-11-14'):'Diwali',
    pd.Timestamp('2015-03-06'):'Holi',    pd.Timestamp('2016-03-24'):'Holi',
    pd.Timestamp('2017-03-13'):'Holi',    pd.Timestamp('2018-03-02'):'Holi',
    pd.Timestamp('2019-03-21'):'Holi',    pd.Timestamp('2020-03-10'):'Holi',
    pd.Timestamp('2016-01-01'):'New Year',pd.Timestamp('2017-01-01'):'New Year',
    pd.Timestamp('2018-01-01'):'New Year',pd.Timestamp('2019-01-01'):'New Year',
    pd.Timestamp('2020-01-01'):'New Year',
}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def aqi_cfg(aqi):
    for cap,cat,col,bg,lt,ic,msg in AQI_CFG:
        if aqi<=cap: return cat,col,bg,lt,ic,msg
    c=AQI_CFG[-1]; return c[1],c[2],c[3],c[4],c[5],c[6]

def festival_features(d):
    ts=pd.Timestamp(d); f,w,name=0,0,None
    for fd,nm in FESTIVALS.items():
        delta=(ts-fd).days
        if delta==0: f=1; name=nm
        if 0<=delta<=2: w=1; name=name or nm
    return f,w,name

def get_hist(city,d):
    ts=pd.Timestamp(d); cdf=df[df['City']==city]
    lc=['AQI_lag_1','AQI_lag_3','AQI_lag_7','PM2.5_lag_1','PM10_lag_1','NO2_lag_1']
    rc=['AQI_rolling_7d','AQI_rolling_14d','AQI_rolling_30d','AQI_rolling_7d_std']
    row=cdf[cdf['Date']==ts]
    if not row.empty:
        v=row.iloc[0]; return {c:v[c] for c in lc+rc},"exact"
    past=cdf[cdf['Date']<ts].sort_values('Date')
    if not past.empty:
        last=past.iloc[-1]
        a7=past.tail(7)['AQI'].mean() if 'AQI' in past.columns else 150.0
        a14=past.tail(14)['AQI'].mean() if 'AQI' in past.columns else 150.0
        a30=past.tail(30)['AQI'].mean() if 'AQI' in past.columns else 150.0
        s7=past.tail(7)['AQI'].std() if 'AQI' in past.columns else 20.0
        return {
            'AQI_lag_1':last.get('AQI',150),
            'AQI_lag_3':past.iloc[-3].get('AQI',150) if len(past)>=3 else 150,
            'AQI_lag_7':past.iloc[-7].get('AQI',150) if len(past)>=7 else 150,
            'PM2.5_lag_1':last.get('PM2.5',80),'PM10_lag_1':last.get('PM10',140),
            'NO2_lag_1':last.get('NO2',45),'AQI_rolling_7d':a7,
            'AQI_rolling_14d':a14,'AQI_rolling_30d':a30,'AQI_rolling_7d_std':s7,
        },"estimated"
    m=ts.month; mdf=cdf[cdf['Month']==m] if not cdf[cdf['Month']==m].empty else cdf
    lc2=lc+rc
    return {c:mdf[c].median() if c in mdf.columns else 150.0 for c in lc2},"fallback"

def sec_defaults(city,month):
    try:    return cma.loc[(city,month)].to_dict()
    except: return {'PM2.5':80,'PM10':140,'NO':15,'NO2':45,'NOx':50,'NH3':25,'CO':1.0,'SO2':15,'O3':40}

def predict(city,d):
    ts=pd.Timestamp(d); hist,hs=get_hist(city,d)
    f,w,fn=festival_features(d)
    season=SEASON_MAP[ts.month]; cenc=cmap.get(city,0)
    sec=sec_defaults(city,ts.month)
    momentum=hist.get('AQI_lag_1',150)-hist.get('AQI_lag_7',150)
    vec={
        'PM2.5':sec['PM2.5'],'PM10':sec['PM10'],'NO':sec['NO'],'NO2':sec['NO2'],
        'NOx':sec['NOx'],'NH3':sec['NH3'],'CO':sec['CO'],'SO2':sec['SO2'],'O3':sec['O3'],
        'Year':ts.year,'Month':ts.month,'Day':ts.day,
        'DayOfWeek':ts.dayofweek,'Quarter':ts.quarter,
        'Season_encoded':season,'City_encoded':cenc,
        'Is_Festival':f,'Is_Festival_Window':w,
        'Is_Weekend':int(ts.dayofweek>=5),'AQI_momentum':momentum,
        'Season_City_interaction':season*cenc,
        'Month_sin':np.sin(2*np.pi*ts.month/12),
        'Month_cos':np.cos(2*np.pi*ts.month/12),
        **hist
    }
    inp=pd.DataFrame([vec])[feat]
    p=float(np.clip(model.predict(inp)[0],10,500))
    return p,hist,hs,vec,fn

def get_drivers(vec,n=5):
    imp=dict(zip(feat,model.feature_importances_))
    cont={f2:abs(float(vec.get(f2,0)))*imp.get(f2,0) for f2 in feat}
    tot=sum(cont.values()) or 1
    return [(k,round(v/tot*100,1)) for k,v in sorted(cont.items(),key=lambda x:x[1],reverse=True)[:n]]

def get_yoy(city,d):
    ts=pd.Timestamp(d)
    try: prev=ts.replace(year=ts.year-1)
    except: return None,None
    cdf=df[df['City']==city]
    row=cdf[cdf['Date']==prev]
    if not row.empty: return row.iloc[0]['AQI'],prev.strftime("%d %b %Y")
    mdf=cdf[(cdf['Year']==ts.year-1)&(cdf['Month']==ts.month)]
    if not mdf.empty: return mdf['AQI'].mean(),f"{MONTHS[ts.month-1]} {ts.year-1} avg"
    return None,None

# Chart style helper
def style_ax(ax, fig, bgcolor='#0C0F19'):
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors='#2D3355', labelsize=8)
    ax.yaxis.grid(True, color='#0F1221', linewidth=0.8, linestyle='--')
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════
#  TOP NAV BAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">🌿 Desh Ki Hawa</div>
  <div class="topbar-tags">
    <span class="topbar-tag">Random Forest</span>
    <span class="topbar-tag">R² = 0.9426</span>
    <span class="topbar-tag">RMSE = 12.35</span>
    <span class="topbar-tag">26 Cities</span>
    <span class="topbar-tag">SHAP Explainable</span>
    <span class="topbar-tag">Festival-Aware</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Predict AQI")
    st.markdown("Select a city and date. The model predicts AQI using learned temporal and seasonal patterns.")
    st.markdown("---")

    cities = sorted(df['City'].unique())
    sel_city = st.selectbox("🏙️ City", cities,
                            index=cities.index("Delhi") if "Delhi" in cities else 0)
    sel_date = st.date_input("📅 Date", value=date.today(),
                             min_value=date(2015,1,1),
                             max_value=date(2030,12,31))
    st.markdown("---")
    go = st.button("⚡  Predict AQI", use_container_width=True)

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
The model uses **33 ML features**:
- Lag AQI (1, 3, 7 days)
- Rolling averages (7, 14, 30 days)
- Season & city patterns
- Festival indicators
- AQI momentum

Pollutant values are **auto-filled** from city historical averages — no manual input needed.
""")
    st.markdown("---")
    st.caption("Trained on CPCB data · 2015–2020 · 24,850 observations")


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for k in ['aqi','meta','ptime']:
    if k not in st.session_state: st.session_state[k]=None


# ══════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════
if go:
    with st.spinner("Predicting..."):
        aqi_val,hist,hs,vec,fest = predict(sel_city,sel_date)
        drvs   = get_drivers(vec)
        py,pl  = get_yoy(sel_city,sel_date)
        warns  = []
        if sel_date.year > 2020:
            warns.append(f"Date beyond training range (2015–2020). Model extrapolates using learned seasonal patterns for {sel_city}.")
        st.session_state.aqi   = aqi_val
        st.session_state.ptime = datetime.now().strftime("%H:%M:%S")
        st.session_state.meta  = {
            'hist':hist,'hs':hs,'drvs':drvs,'py':py,'pl':pl,
            'warns':warns,'city':sel_city,'date':sel_date,'fest':fest,
        }


# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════
if st.session_state.aqi is not None:
    aqi  = st.session_state.aqi
    meta = st.session_state.meta
    pt   = st.session_state.ptime
    cat,col,bg,lt,ic,msg = aqi_cfg(aqi)
    risk,avoid,mask = HEALTH[cat]

    tab1,tab2,tab3 = st.tabs([
        "  🎯  Prediction Dashboard  ",
        "  🏙️  City Analysis  ",
        "  📊  Model Insights  "
    ])

    # ══════════════════════════════════════
    with tab1:
        # Notifications
        for w in meta['warns']:
            st.markdown(f'<div class="nb nb-warn">⚠ {w}</div>', unsafe_allow_html=True)
        if meta['fest']:
            st.markdown(f'<div class="nb nb-fest">🎆 <b>{meta["fest"]} period detected</b> — Festival indicators are active in this prediction. Fireworks and bonfires typically raise AQI by 20–40%.</div>', unsafe_allow_html=True)
        if meta['hs']=='exact':
            st.markdown('<div class="nb nb-success">✓ Real historical data matched — lag and rolling features loaded from actual recorded dataset values.</div>', unsafe_allow_html=True)
        elif meta['hs']=='estimated':
            st.markdown('<div class="nb nb-info">ℹ Future date — lag features seeded from the most recent recorded data available for this city.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="nb nb-info">ℹ Lag features estimated from {meta["city"]} seasonal monthly averages.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 1: AQI card + 3 stats ──
        r1,r2,r3,r4 = st.columns([2.8,1,1,1], gap="medium")

        with r1:
            yoy_html=""
            if meta['py']:
                diff=aqi-meta['py']; pct=abs(diff)/meta['py']*100
                arrow="↑" if diff>0 else "↓"
                c2="#EF4444" if diff>0 else "#10B981"
                word="worse" if diff>0 else "better"
                yoy_html=f'<div class="yoy-row" style="color:{c2}">{arrow} {pct:.1f}% {word} than {meta["pl"]} ({meta["py"]:.0f})</div>'

            date_str=pd.Timestamp(meta['date']).strftime("%d %B %Y")
            st.markdown(f"""
            <div class="aqi-display" style="background:{bg};border-color:{col}30"
                 data-glow="{col}">
              <div class="aqi-city-date">{meta['city']} &nbsp;·&nbsp; {date_str}</div>
              <div class="aqi-number" style="color:{col}">{aqi:.0f}</div>
              <div>
                <span class="aqi-badge" style="color:{col};border-color:{col}40;background:{col}15">
                  {ic} &nbsp; {cat}
                </span>
              </div>
              <div class="aqi-message">{msg}</div>
              {yoy_html}
              <div class="aqi-time">Predicted at {pt} · Random Forest V2 · R²=0.9426</div>
            </div>""", unsafe_allow_html=True)

        ca=df[df['City']==meta['city']]['AQI']
        wm=df[df['City']==meta['city']].groupby('Month')['AQI'].mean().idxmax()
        bm=df[df['City']==meta['city']].groupby('Month')['AQI'].mean().idxmin()

        for col2,lbl,val,sub in [
            (r2,"City Avg AQI",f"{ca.mean():.0f}","historical mean"),
            (r3,"Worst Month",MONTHS[wm-1],"highest pollution"),
            (r4,"Best Month",MONTHS[bm-1],"cleanest month"),
        ]:
            with col2:
                st.markdown(f"""
                <div class="stat-mini">
                  <div class="stat-mini-lbl">{lbl}</div>
                  <div class="stat-mini-val">{val}</div>
                  <div class="stat-mini-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 2: Drivers | Historical | Health ──
        a1,a2,a3 = st.columns([2,1.6,1.6], gap="medium")

        with a1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">⚡ Why this prediction?</div>', unsafe_allow_html=True)
            drvs=meta['drvs']
            top=drvs[0][0].replace('_',' ')
            s2=drvs[1][0].replace('_',' ') if len(drvs)>1 else ""
            if aqi>200:
                expl=f"<b style='color:{col}'>{cat}</b> conditions. <b>{top}</b> ({drvs[0][1]}%) and <b>{s2}</b> are the primary drivers."
            elif aqi>100:
                expl=f"Moderate conditions. <b>{top}</b> is dominant at {drvs[0][1]}% — recent pollution trend is significant."
            else:
                expl=f"Clean air. <b>{top}</b> leads at {drvs[0][1]}%, reflecting stable and improving recent conditions."
            st.markdown(f'<div style="font-size:12px;color:#4B5280;margin-bottom:16px;line-height:1.7">{expl}</div>', unsafe_allow_html=True)
            d_cols=[col,'#818CF8','#C084FC','#22D3EE','#34D399']
            for i,(f2,pct) in enumerate(drvs):
                dc=d_cols[i%len(d_cols)]
                st.markdown(f"""
                <div class="drv">
                  <div class="drv-top">
                    <span>{f2.replace('_',' ')}</span>
                    <span class="drv-top-val">{pct}%</span>
                  </div>
                  <div class="drv-bg">
                    <div class="drv-fill" style="width:{pct}%;background:linear-gradient(90deg,{dc}99,{dc})"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:#1E2440;margin-top:12px">Weighted by RF feature importances × input magnitudes</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with a2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📅 Historical Context</div>', unsafe_allow_html=True)
            h=meta['hist']
            for k,v in [
                ("Yesterday AQI",h.get('AQI_lag_1',0)),
                ("3 Days Ago",   h.get('AQI_lag_3',0)),
                ("7 Days Ago",   h.get('AQI_lag_7',0)),
                ("7-Day Avg",    h.get('AQI_rolling_7d',0)),
                ("30-Day Avg",   h.get('AQI_rolling_30d',0)),
            ]:
                st.markdown(f'<div class="lag-row"><span class="lag-k">{k}</span><span class="lag-v">{v:.0f}</span></div>', unsafe_allow_html=True)
            if meta['py']:
                diff=aqi-meta['py']
                c3="#EF4444" if diff>0 else "#10B981"
                w3="higher" if diff>0 else "lower"
                st.markdown(f'<div style="font-size:12px;color:{c3};margin-top:14px;padding-top:12px;border-top:1px solid #0F1221">{abs(diff):.0f} AQI points {w3} than {meta["pl"]} ({meta["py"]:.0f})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with a3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🏥 Health Advisory</div>', unsafe_allow_html=True)
            for lbl,items in [("At-risk groups",risk),("Avoid",avoid)]:
                st.markdown(f'<div class="health-blk"><div class="health-lbl">{lbl}</div>', unsafe_allow_html=True)
                for item in items:
                    st.markdown(f'<div class="health-row"><div class="h-dot" style="background:{col}"></div>{item}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="health-blk">
              <div class="health-lbl">Mask recommendation</div>
              <div class="health-row" style="color:{col};font-weight:600">
                <div class="h-dot" style="background:{col}"></div>{mask}
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 3: Monthly chart | City ranking ──
        ch1,ch2 = st.columns([3,2], gap="medium")

        with ch1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="sec-title">📈 Monthly AQI Pattern — {meta["city"]}</div>', unsafe_allow_html=True)
            cdf2=df[df['City']==meta['city']]
            monthly=cdf2.groupby('Month')['AQI'].mean()
            sel_m=pd.Timestamp(meta['date']).month
            fig,ax=plt.subplots(figsize=(9,3.4))
            style_ax(ax,fig)
            bcs=[aqi_cfg(v)[1] for v in monthly.values]
            bars=ax.bar(monthly.index,monthly.values,color=bcs,edgecolor='#07090F',width=0.72,alpha=0.9,zorder=2)
            # Glow effect on selected month
            if sel_m in monthly.index:
                ax.bar([sel_m],[monthly[sel_m]],color=col,edgecolor='#07090F',
                       width=0.72,alpha=0.15,zorder=1)
            ax.plot(monthly.index,monthly.values,color='#E2E8FF',
                    lw=1,alpha=0.2,marker='o',ms=2.5,zorder=3)
            ax.axvline(sel_m,color='white',lw=1.5,ls='--',alpha=0.3,
                       label=f'Selected: {MONTHS[sel_m-1]}')
            ax.axhline(aqi,color=col,lw=1.2,ls=':',alpha=0.7,
                       label=f'Prediction: {aqi:.0f}')
            ax.set_xticks(range(1,13))
            ax.set_xticklabels(MONTHS,color='#2D3355',fontsize=8)
            ax.tick_params(axis='y',colors='#2D3355',labelsize=8)
            ax.set_ylabel('Avg AQI',color='#2D3355',fontsize=8)
            ax.legend(facecolor='#0C0F19',edgecolor='#151929',labelcolor='#636B8A',fontsize=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🏆 Top Polluted Cities</div>', unsafe_allow_html=True)
            top10=df.groupby('City')['AQI'].mean().sort_values(ascending=True).tail(10)
            fig2,ax2=plt.subplots(figsize=(5,3.4))
            style_ax(ax2,fig2)
            bc2=[col if c==meta['city'] else '#1E2440' for c in top10.index]
            ax2.barh(top10.index,top10.values,color=bc2,edgecolor='#07090F',height=0.65,zorder=2)
            # Highlight selected city label
            for label in ax2.get_yticklabels():
                if label.get_text()==meta['city']:
                    label.set_color(col); label.set_fontweight('bold')
                else:
                    label.set_color('#2D3355')
            ax2.tick_params(axis='x',colors='#2D3355',labelsize=8)
            ax2.set_xlabel('Avg AQI',color='#2D3355',fontsize=8)
            ax2.xaxis.grid(True,color='#0F1221',lw=0.8,linestyle='--')
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── AQI Scale ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📏 AQI Scale Reference</div>', unsafe_allow_html=True)
        sc=[(c[0],c[1],c[2]) for c in AQI_CFG]
        fig3,ax3=plt.subplots(figsize=(12,1.1))
        fig3.patch.set_facecolor('#0C0F19'); ax3.set_facecolor('#0C0F19')
        for i,(cap,cats,clr) in enumerate(sc):
            # Gradient-like effect
            ax3.barh(0,1,left=i,color=clr,edgecolor='#07090F',lw=2,height=0.65,alpha=0.9)
            ax3.text(i+0.5,0.45,cats,ha='center',va='bottom',
                     fontsize=8,color='white',fontweight='bold')
            rng=f"0–{cap}" if i==0 else f"{sc[i-1][0]+1}–{cap}"
            ax3.text(i+0.5,-0.45,rng,ha='center',va='top',fontsize=7,color='#3D4468')
        ptr=min(aqi/500*6,5.92)
        # Triangle pointer
        ax3.annotate('',xy=(ptr,-0.28),xytext=(ptr,-0.05),
                     arrowprops=dict(arrowstyle='->',color='white',lw=2.5))
        ax3.text(ptr,-0.55,f'{aqi:.0f}',ha='center',va='top',
                 fontsize=8,color=col,fontweight='bold',fontfamily='JetBrains Mono')
        ax3.set_xlim(0,6); ax3.set_ylim(-0.8,0.9); ax3.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig3); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bottom model info ──
        b1,b2,b3,b4,b5 = st.columns(5, gap="medium")
        for col2,lbl,val in zip([b1,b2,b3,b4,b5],
            ["R² Score","RMSE","Features","Cities Covered","Training Observations"],
            ["0.9426","12.35","33","26","24,850"]):
            with col2:
                st.markdown(f"""
                <div class="stat-mini">
                  <div class="stat-mini-lbl">{lbl}</div>
                  <div class="stat-mini-val" style="font-size:1.1rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="nb nb-info" style="margin-top:14px">
        <b>How this prediction works:</b>
        The model predicts AQI using <b>temporal patterns</b> — not the AQI formula.
        Key inputs: yesterday's AQI (AQI_lag_1), weekly rolling average, seasonal encoding,
        city patterns, and festival indicators. Pollutant values are auto-filled from city
        historical averages. AQI_lag_1 alone contributes <b>47.7 AQI units</b> of average
        predictive impact — nearly 2× more than any single pollutant reading.
        </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        if mc_df is not None:
            best=mc_df.loc[mc_df['RMSE'].idxmin()]
            worst=mc_df.loc[mc_df['RMSE'].idxmax()]
            ab90=(mc_df['R2']>=0.90).sum()

            # Summary cards
            t1,t2,t3,t4 = st.columns(4, gap="medium")
            for cc,lbl,val,sub,clr in [
                (t1,"Best Predicted",best['City'],f"RMSE={best['RMSE']:.1f} · R²={best['R2']:.3f}","#10B981"),
                (t2,"Hardest City",worst['City'],f"RMSE={worst['RMSE']:.1f} · R²={worst['R2']:.3f}","#EF4444"),
                (t3,"Cities R²≥0.90",str(ab90),f"out of {len(mc_df)} evaluated","#F59E0B"),
                (t4,"Avg R²",f"{mc_df['R2'].mean():.4f}","across all cities","#818CF8"),
            ]:
                with cc:
                    st.markdown(f"""
                    <div class="stat-mini">
                      <div class="stat-mini-lbl">{lbl}</div>
                      <div class="stat-mini-val" style="color:{clr};font-size:1.15rem">{val}</div>
                      <div class="stat-mini-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ch_a,ch_b = st.columns(2, gap="medium")

            for ax_col,skey,xlabel,title in [
                (ch_a,'RMSE','RMSE (AQI units)','RMSE by City — lower is better'),
                (ch_b,'R2','R² Score','R² Score by City — higher is better'),
            ]:
                with ax_col:
                    st.markdown(f'<div class="card"><div class="sec-title">{title}</div>', unsafe_allow_html=True)
                    sdf=mc_df.sort_values(skey,ascending=(skey=='RMSE'))
                    fig,ax=plt.subplots(figsize=(8,7))
                    style_ax(ax,fig)
                    norm=plt.Normalize(sdf[skey].min(),sdf[skey].max())
                    cmap_n='RdYlGn_r' if skey=='RMSE' else 'RdYlGn'
                    clrs=plt.cm.get_cmap(cmap_n)(norm(sdf[skey].values))
                    bars=ax.barh(sdf['City'],sdf[skey],color=clrs,edgecolor='#07090F',height=0.68,zorder=2)
                    for bar,val in zip(bars,sdf[skey]):
                        ax.text(val+0.004 if skey=='R2' else val+0.3,
                                bar.get_y()+bar.get_height()/2,
                                f'{val:.3f}' if skey=='R2' else f'{val:.1f}',
                                va='center',fontsize=7.5,color='#636B8A')
                    if skey=='R2': ax.axvline(0.90,color='#636B8A',lw=1,ls='--',alpha=0.5); ax.set_xlim(0,1.08)
                    ax.set_xlabel(xlabel,color='#2D3355',fontsize=8)
                    ax.tick_params(axis='x',colors='#2D3355',labelsize=7.5)
                    ax.tick_params(axis='y',colors='#636B8A',labelsize=7.5)
                    ax.xaxis.grid(True,color='#0F1221',lw=0.8,linestyle='--')
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig); plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">📋 Full Results Table</div>', unsafe_allow_html=True)
            disp=mc_df[['City','RMSE','MAE','R2','Mean_AQI']].sort_values('R2',ascending=False).reset_index(drop=True)
            st.dataframe(disp.style
                .background_gradient(subset=['R2'],   cmap='Greens')
                .background_gradient(subset=['RMSE'], cmap='Reds_r')
                .format({'RMSE':'{:.2f}','MAE':'{:.2f}','R2':'{:.4f}','Mean_AQI':'{:.1f}'}),
                use_container_width=True, height=420)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""<div class="nb nb-info" style="margin-top:12px">
            <b>Insight:</b> Cities with stable, consistent pollution patterns (Hyderabad, Delhi)
            are most predictable. Amritsar scores lowest (R²=0.48) due to irregular Parali
            (agricultural stubble burning) spikes — event-driven pollution not well-represented
            in training data. This is an honest and expected limitation of pattern-based ML.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Run the multi-city analysis notebook to generate data.")


    # ══════════════════════════════════════
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🧠 SHAP — How the Model Decides</div>', unsafe_allow_html=True)
        st.markdown("""<div class="nb nb-info" style="margin-bottom:16px">
        SHAP (SHapley Additive exPlanations) is the gold standard for ML explainability.
        It shows exactly how much each feature pushes each prediction up or down.
        <b>AQI_lag_1 contributes 47.7 AQI units</b> — nearly 2× more than PM2.5 (24.7).
        This confirms the model relies on <b>temporal memory</b>, not pollutant calculation.
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if shap_vals is not None and feat_v1 is not None:
            mean_shap=np.abs(shap_vals).mean(axis=0)
            simp=pd.Series(mean_shap,index=feat_v1).sort_values(ascending=True).tail(15)

            s1,s2=st.columns(2, gap="medium")
            with s1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-title">📊 Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(8,6.5))
                style_ax(ax,fig)
                norm=plt.Normalize(simp.values.min(),simp.values.max())
                clrs=plt.cm.RdYlGn_r(norm(simp.values))
                bars=ax.barh(simp.index,simp.values,color=clrs,edgecolor='#07090F',height=0.7,zorder=2)
                for bar,val in zip(bars,simp.values):
                    ax.text(val+0.15,bar.get_y()+bar.get_height()/2,
                            f'{val:.1f}',va='center',fontsize=8,color='#636B8A')
                ax.set_xlabel('Mean |SHAP Value| — avg AQI impact',color='#2D3355',fontsize=8)
                ax.tick_params(axis='x',colors='#2D3355',labelsize=8)
                ax.tick_params(axis='y',colors='#8896B0',labelsize=8)
                ax.xaxis.grid(True,color='#0F1221',lw=0.8,linestyle='--')
                plt.tight_layout(pad=0.5)
                st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            with s2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-title">🏅 Top 5 Key Drivers</div>', unsafe_allow_html=True)
                top5=simp.tail(5).iloc[::-1]
                icons=['🥇','🥈','🥉','4️⃣','5️⃣']
                for i,(f2,val) in enumerate(top5.items()):
                    pct=val/simp.sum()*100
                    bar_w=val/top5.values.max()*100
                    st.markdown(f"""
                    <div style="background:#0A0D16;border:1px solid #151929;border-radius:12px;
                                padding:14px 18px;margin-bottom:10px;position:relative;overflow:hidden">
                      <div style="position:absolute;bottom:0;left:0;height:3px;width:{bar_w}%;
                                  background:linear-gradient(90deg,#818CF8,#C084FC);opacity:0.4"></div>
                      <div style="display:flex;align-items:center;gap:10px">
                        <span style="font-size:20px">{icons[i]}</span>
                        <div>
                          <div style="font-size:13px;font-weight:600;color:#E2E8FF">
                            {f2.replace('_',' ')}
                          </div>
                          <div style="font-size:11px;color:#3D4468;margin-top:3px">
                            Avg impact: <b style="color:#10B981">{val:.1f} AQI</b>
                            &nbsp;·&nbsp; {pct:.1f}% of total
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🖼️ SHAP Visualizations</div>', unsafe_allow_html=True)
            try:
                from PIL import Image
                p1,p2=st.columns(2, gap="medium")
                with p1:
                    st.image(Image.open('visualizations/14_shap_beeswarm.png'),
                             caption='Beeswarm — direction & magnitude across 500 predictions',
                             use_container_width=True)
                with p2:
                    st.image(Image.open('visualizations/15_shap_waterfall.png'),
                             caption='Waterfall — single prediction fully explained',
                             use_container_width=True)
            except:
                st.info("Run the SHAP notebook to generate these visualizations.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run the SHAP explainability notebook first.")

        st.markdown("<br>", unsafe_allow_html=True)
        # Model summary card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🤖 About the Model</div>', unsafe_allow_html=True)
        ab1,ab2=st.columns(2, gap="medium")
        with ab1:
            st.markdown("""
            **Algorithm:** Random Forest Regressor

            **Training data:** 20,268 samples · 2015–2019

            **Test data:** 4,582 samples · 2020

            **Features:** 33 engineered features including lag, rolling averages, season, city, festival indicators

            **Why Random Forest?** Handles non-linear pollutant relationships, robust to outliers, provides built-in feature importance — ideal for AQI data with heavy-tailed distributions.
            """)
        with ab2:
            st.markdown("""
            **Performance:**
            | Metric | Score |
            |---|---|
            | R² Score | **0.9426** |
            | RMSE | **12.35** |
            | MAE | ~10.1 |
            | Improvement vs baseline | **+19.92%** |

            **Key insight:** Festival indicators (Diwali, Holi etc.) reduced RMSE by 19.92% — proving domain knowledge outperforms model complexity.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">🔭 Future Scope</div>', unsafe_allow_html=True)
        fs1,fs2,fs3 = st.columns(3, gap="medium")
        for col2,icon,title,desc in [
            (fs1,"📡","Real-time Integration","Connect to OpenAQ or CPCB live API for real-time sensor data. Enable true now-casting with current pollution readings."),
            (fs2,"🧠","Deep Learning Models","Implement LSTM or Transformer-based models for true sequential multi-day forecasting with uncertainty quantification."),
            (fs3,"🌍","Expanded Coverage","Scale to 100+ cities, add district-level granularity, integrate satellite AOD (Aerosol Optical Depth) data from NASA."),
        ]:
            with col2:
                st.markdown(f"""
                <div style="background:#0A0D16;border:1px solid #151929;border-radius:12px;padding:16px 18px">
                  <div style="font-size:24px;margin-bottom:8px">{icon}</div>
                  <div style="font-size:13px;font-weight:600;color:#E2E8FF;margin-bottom:6px">{title}</div>
                  <div style="font-size:12px;color:#3D4468;line-height:1.6">{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🌿</div>
      <div class="empty-title">Select a city and date to predict AQI</div>
      <div class="empty-sub">
        Use the sidebar on the left to choose a city and date, then click
        <b style="color:#818CF8">⚡ Predict AQI</b> to get an instant AI prediction.
        Works for any date — past, present, or future.
      </div>
    </div>
    """, unsafe_allow_html=True)
