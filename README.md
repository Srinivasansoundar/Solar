# ☀️ SmartSolar — Intelligent Energy Decision Engine

**Multi-model energy intelligence platform** combining solar generation forecasting, household consumption prediction, and grid tariff forecasting into a unified decision dashboard.

| Model | Algorithm | Horizon |
|-------|-----------|---------|
| ☀️ Solar Forecast | XGBoost Meta-Model (VMD + hybrid feature selection) | t+1h, t+2h, t+3h |
| ⚡ Energy Consumption | Recursive Hybrid XGBoost + ResidualLSTM | 24-hour |
| 💰 Tariff Forecast | 2-layer BiLSTM | 12-hour |

---

## 📁 Project Structure

```
smartsolar/
├── saved_models/                      # Solar model files (not in Git)
│   ├── scaler_features.pkl
│   ├── scaler_target.pkl
│   ├── feature_names.pkl
│   ├── xgb_enhanced_12_3_h0.pkl
│   ├── xgb_enhanced_12_3_h1.pkl
│   ├── xgb_enhanced_12_3_h2.pkl
│   ├── base_models_12_3.pkl
│   └── selected_indices_12_3.pkl
├── energy_consumption_saved_models/   # Energy model files (not in Git)
│   ├── xgb_consumption.pkl
│   ├── lstm_consumption.pt
│   ├── res_scaler_consumption.pkl
│   ├── feature_names_consumption.pkl
│   └── seq_len.pkl
├── tariff_forcasting_model/           # Tariff model (client-side, included)
│   ├── lstm_model.py                  # BiLSTM architecture + predict_tariff()
│   └── tariff_lstm_weights.pth        # Trained weights
├── backend/
│   ├── main.py                        # FastAPI (Solar + Energy endpoints)
│   └── requirements.txt
├── frontend/
│   ├── app.py                         # Streamlit app (4 tabs)
│   ├── requirements.txt
│   ├── sample_solar.csv               # Sample CSV for ☀️ Solar tab
│   ├── sample_consumption_24h.csv     # Sample CSV for ⚡ Energy tab
│   └── sample_tariff.csv              # Sample CSV for 💰 Tariff tab
└── .gitignore
```

---

## ⚙️ Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Srinivasansoundar/Solar.git
cd smartsolar
```

---

### 2️⃣ Download Saved Models

Model weight files are excluded from Git due to size. Download and place them in the correct folders.

**Solar models** → place in `smartsolar/saved_models/`

👉 [Google Drive — Solar Models](https://drive.google.com/drive/folders/1XqG_3399_OTNGNdEJ8jX81tDFSMMao1X?usp=sharing)

**Energy consumption models** → place in `smartsolar/energy_consumption_saved_models/`

👉 [Google Drive — Energy Models](https://drive.google.com/drive/folders/1k_uSKYQq0d2LNQeHJMHbvwZ9Li5PYDzb?usp=sharing)

> **Tariff model** -> place in 'smartsolar/tariff_forcasting_model/'

👉 [Google Drive — Tariff Model](https://drive.google.com/drive/folders/1YwLhgDQGljp5xPgyacOXe6hmVzdg6eXQ?usp=sharing)

---

### 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

## 🚀 Running the Application

Both servers must be running simultaneously.

### ▶ Start Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- API: [http://localhost:8000](http://localhost:8000)
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### ▶ Start Frontend (Streamlit)

Open a **new terminal**:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

- App: [http://localhost:8501](http://localhost:8501)

---

## 📊 Using the App

The app has **4 tabs**. Run each model tab first, then visit the Combined Dashboard.

### ☀️ Tab 1 — Solar Forecast

Predicts GHI and solar energy output for the next **3 hours**.

- **Manual Entry:** Enter current sensor readings → app auto-generates 35 context rows
- **Upload CSV:** Use `sample_solar.csv` as a template (min. 30 rows)

### ⚡ Tab 2 — Energy Consumption

Predicts household power consumption for the next **24 hours**.

- **Manual Entry:** Enter current `Global_active_power` reading
- **Upload CSV:** Use `sample_consumption_24h.csv` as a template (min. 24 rows)

### 💰 Tab 3 — Tariff Forecast

Predicts grid electricity tariff (₹/kWh) for the next **12 hours** — runs entirely client-side (no backend needed).

- **Manual Entry:** Set hour, day, month, temperature, historical avg tariff, demand index
- **Upload CSV:** Use `sample_tariff.csv` as a template (last row is used)

### 📊 Tab 4 — Combined Dashboard

Synthesises all three model outputs:

- **⚡ Next 1-Hour Decision** — compares solar generated vs energy needed for the next hour; if solar > demand, recommends selling the surplus at the tariff peak hour identified by the BiLSTM model
- **⚖️ 3-Hour Energy Balance** — solar vs consumption comparison over 3-hour window
- **📈 Unified Timeline** — overlaid line chart of all three forecasts
- **💡 Combined Recommendations** — actionable sell/buy/defer decisions

> Run all three model tabs first before visiting the Combined Dashboard.

---

## 📄 CSV Formats

### ☀️ Solar (`sample_solar.csv`) — min. 30 rows

| Column | Description | Unit |
|--------|-------------|------|
| datetime | Timestamp | YYYY-MM-DD HH:MM |
| GHI | Global Horizontal Irradiance | W/m² |
| DNI | Direct Normal Irradiance | W/m² |
| DHI | Diffuse Horizontal Irradiance | W/m² |
| T_amb | Ambient Temperature | °C |
| RH | Relative Humidity | % |
| WS | Wind Speed | m/s |
| WS_gust | Wind Gust Speed | m/s |
| WD | Wind Direction | ° |
| WD_std | Wind Direction Std Dev | ° |
| BP | Barometric Pressure | hPa |

### ⚡ Energy (`sample_consumption_24h.csv`) — min. 24 rows

| Column | Description | Unit |
|--------|-------------|------|
| datetime **or** dt | Timestamp | YYYY-MM-DD HH:MM |
| Global_active_power | Household active power | kW |

> Extra columns (Voltage, Sub_metering, etc.) are automatically ignored.

### 💰 Tariff (`sample_tariff.csv`) — 1+ rows (last row used)

| Column | Description | Range |
|--------|-------------|-------|
| hour | Hour of day | 0–23 |
| day_of_week | Day (0=Mon … 6=Sun) | 0–6 |
| month | Month | 1–12 |
| temperature | Ambient temperature | °C |
| historical_avg_tariff | Region avg tariff | ₹/kWh |
| demand_index | Grid demand level | 0.0–1.0 |

---

## 🔧 Panel Settings (Sidebar — Solar Tab)

| Setting | Default |
|---------|---------|
| Panel Area | 10 m² |
| Panel Efficiency | 20% |

---

## ⚡ Solar Energy Formula

```
Energy (kWh) = GHI × Area × Efficiency / 1000
```

---

## 🧠 Model Pipelines

### ☀️ Solar (XGBoost Meta-Model)

```
Raw sensor data (11 features)
        ↓
Feature Engineering (~80 features)
  · Cyclic encoding (hour, month, day-of-year, day-of-week)
  · Solar elevation + clear-sky index
  · Lag features (1,2,3,6,12,24h) for GHI, DNI, DHI, T_amb
  · Rolling stats (mean, std, min, max) over 3,6,12,24h windows
  · EMA (3,6,12 spans), VMD proxy modes (×7)
        ↓
RobustScaler → Lookback sequence [12 × n_features]
        ↓
Stacking meta-features (RF + ExtraTrees + Ridge + Lasso)
        ↓
Hybrid feature selection (Mutual Info + F-stat + GBM importance)
        ↓
XGBoost → h0 (t+1h), h1 (t+2h), h2 (t+3h) → GHI (W/m²) → kWh
```

### ⚡ Energy Consumption (Hybrid XGB + ResidualLSTM)

```
Hourly Global_active_power history
        ↓
Calendar features (hour, day, month, weekday)
Lag features (1,2,3,6,12,24,48h)
Rolling stats (mean×6h, mean×24h, std×24h)
        ↓
XGBoost prediction (base)
        ↓
ResidualLSTM (2-layer, 64 hidden) on last 24 scaled residuals
        ↓
Hybrid prediction = XGB + LSTM residual correction
        ↓
Recursive 24-step forecast
```

### 💰 Tariff (BiLSTM)

```
6 input parameters (hour, day, month, temp, hist_avg, demand)
        ↓
24-step feature sequence (14 features: cyclic + demand + lag proxies)
        ↓
2-layer Bidirectional LSTM (128 hidden units)
        ↓
12-hour tariff forecast (₹/kWh)
```

---

## 🛠️ Requirements

- Python **3.10+**
- `backend/requirements.txt` — FastAPI, XGBoost, PyTorch, scikit-learn, joblib
- `frontend/requirements.txt` — Streamlit, pandas, numpy, requests, torch

---

## 📌 Notes

- Ensure `saved_models/` and `energy_consumption_saved_models/` exist before starting the backend
- Backend must be running **before** launching the Streamlit frontend
- The **Tariff tab** runs fully client-side — no backend required
- The **Combined Dashboard** requires at least one model tab to have been run first
- Recommended: run inside a virtual environment
