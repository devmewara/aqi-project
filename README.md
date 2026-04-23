# 🌫️ AQI Estimation System — India

> Condition-based Air Quality Index estimation using spatio-temporal machine learning across 26 Indian cities.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![R2 Score](https://img.shields.io/badge/R²-0.9426-brightgreen?style=flat-square)
![RMSE](https://img.shields.io/badge/RMSE-17.80-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 What This Project Does

This system estimates the Air Quality Index (AQI) for a given Indian city under specified atmospheric and pollutant conditions. It uses patterns learned from **24,000+ daily observations** across **26 Indian cities** between **2015 and 2020**.

> ⚠️ **Important distinction:** This is a **condition-based AQI estimation** system, not a time-series forecast. It answers the question: *"Given these pollutant levels and recent air quality history, what AQI is expected?"* — not *"What will tomorrow's AQI be?"*

---

## 🚀 Live Demo

```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score | Meaning |
|--------|-------|---------|
| R² Score | **0.9426** | Model explains 94.26% of AQI variance |
| RMSE | **17.80** | Avg error of ~17.8 AQI units (scale: 0–500) |
| MAE | **10.26** | Median absolute error of 10.3 AQI units |
| Training samples | **20,268** | 2015–2019, 26 cities |
| Test samples | **~4,500** | 2020 (includes COVID lockdown period) |

> The model performed well even on 2020 test data — an anomalous year due to COVID-19 lockdowns causing unusually low AQI — demonstrating robustness to distribution shift.

---

## 🗂️ Project Structure

```
aqi_project/
│
├── app.py                        ← Streamlit web application
│
├── data/
│   ├── city_day.csv              ← Raw dataset (Kaggle)
│   ├── clean_aqi.csv             ← After cleaning & outlier removal
│   ├── featured_aqi.csv          ← After feature engineering
│   ├── X_train.csv / X_test.csv  ← Model-ready train/test sets
│   └── y_train.csv / y_test.csv  ← Target values
│
├── models/
│   ├── random_forest.pkl         ← Trained Random Forest model
│   ├── xgboost.pkl               ← Trained XGBoost model
│   ├── feature_cols.pkl          ← Feature column order
│   └── city_mapping.pkl          ← City → integer encoding
│
├── notebooks/
│   ├── 01_eda.ipynb              ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   ← Data cleaning & validation
│   ├── 03_feature_engineering.ipynb ← Lag, rolling, time features
│   ├── 04_modeling.ipynb         ← Model training & evaluation
│   └── 05_visualizations.ipynb  ← All 12 project plots
│
└── visualizations/
    ├── 01_aqi_overview.png
    ├── 04_feature_engineering.png
    ├── 05_actual_vs_predicted.png
    ├── 07_feature_importance.png
    ├── 12_covid_lockdown_effect.png
    └── ... (12 total)
```

---

## 🔧 Features Used (27 Total)

### Pollutant Features (current conditions)
`PM2.5` · `PM10` · `NO` · `NO2` · `NOx` · `NH3` · `CO` · `SO2` · `O3`

### Temporal Features
`Year` · `Month` · `Day` · `DayOfWeek` · `Quarter` · `Season_encoded`

### Lag Features (historical memory)
`AQI_lag_1` · `AQI_lag_3` · `AQI_lag_7` · `PM2.5_lag_1` · `PM10_lag_1` · `NO2_lag_1`

### Rolling Average Features (trend)
`AQI_rolling_7d` · `AQI_rolling_14d` · `AQI_rolling_30d` · `AQI_rolling_7d_std`

### Spatial Feature
`City_encoded`

> **Key insight:** `AQI_lag_1` (yesterday's AQI) has ~0.92 correlation with today's AQI — making it the strongest single predictor. This validates the spatio-temporal feature engineering approach.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aqi-prediction-ml.git
cd aqi-prediction-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib
```

### 3. Download the dataset
- Go to: [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- Download and place `city_day.csv` in the `data/` folder

### 4. Run the notebooks in order
```
01_eda.ipynb → 02_preprocessing.ipynb → 03_feature_engineering.ipynb → 04_modeling.ipynb
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 📱 App Features

| Feature | Description |
|---------|-------------|
| **City & Date Selection** | 26 Indian cities, any date in range |
| **Smart Input Reduction** | Only 3 primary sliders (PM2.5, PM10, NO₂) |
| **Auto Historical Lookup** | Lag & rolling features fetched from real dataset |
| **Predict Button** | On-demand prediction with loading spinner |
| **Prediction Breakdown** | Top 5 feature drivers with percentage contributions |
| **Health Advisory** | At-risk groups, activities to avoid, mask recommendation |
| **Year-over-Year** | Compares prediction to same period previous year |
| **Input Validation** | Warns on extreme or physically impossible inputs |
| **AQI Scale Reference** | Visual scale with pointer showing prediction position |

---

## 📈 Key Findings

1. **AQI is strongly auto-correlated** — yesterday's AQI explains 92% of today's, confirming temporal persistence of pollution
2. **PM2.5 is the dominant physical driver** — highest correlation (~0.78) among individual pollutants
3. **Seasonal patterns are strong** — AQI peaks in winter (Oct–Feb) and drops ~60% during monsoon (Jun–Sep)
4. **COVID-19 lockdown effect visible** — Delhi's AQI dropped ~40% during March–May 2020, captured cleanly in the data
5. **Ahmedabad shows higher historical average than Delhi** — due to longer data coverage (from 2015) and industrial base; Delhi has higher peak values but more variability

---

## 🧪 Model Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **Random Forest** ✅ | **17.80** | **10.26** | **0.9426** |
| XGBoost | 19.77 | 12.10 | 0.9292 |

Random Forest outperformed XGBoost on this dataset. Both models used identical 27-feature inputs and time-ordered train/test split.

---

## 🗓️ Phase 2 Roadmap (In Progress)

- [ ] LSTM time-series model for sequential AQI forecasting
- [ ] Multi-city simultaneous prediction pipeline  
- [ ] Weather feature integration (temperature, humidity, wind)
- [ ] SHAP-based explainability
- [ ] Real-time data via OpenAQ API
- [ ] Automated retraining pipeline

---

## 📄 Dataset

**Source:** [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
**Original source:** Central Pollution Control Board (CPCB), Government of India  
**Coverage:** 26 cities · 2015–2020 · Daily readings  
**Raw rows:** 29,531 · **After cleaning:** 24,850  

---

## 👤 Author

**Devmewara**  
Engineering Student | ML Enthusiast  
📎 [GitHub](https://github.com/devmewara) · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

---

## 📝 License

MIT License — free to use, modify, and distribute with attribution.
