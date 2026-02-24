# ☀️ SmartSolar — 3-Hour Ahead Solar Forecasting

**3-hour ahead GHI prediction and solar energy estimation** using an **XGBoost Meta-Model** with **VMD decomposition** and **hybrid feature selection**.

* **Model:** XGBoost Enhanced
* **Window:** (12,3)
* **Forecast Horizon:** t+1h, t+2h, t+3h
* **Location:** Islamabad, Pakistan

---

## 📁 Project Structure

```
smartsolar/
├── saved_models/              # trained model files (not in Git)
│   ├── scaler_features.pkl
│   ├── scaler_target.pkl
│   ├── feature_names.pkl
│   ├── xgb_enhanced_12_3_h0.pkl
│   ├── xgb_enhanced_12_3_h1.pkl
│   ├── xgb_enhanced_12_3_h2.pkl
│   ├── base_models_12_3.pkl
│   └── selected_indices_12_3.pkl
├── backend/
│   ├── main.py                # FastAPI backend
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit frontend
│   └── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/smartsolar.git
cd smartsolar
```

---

### 2️⃣ Download Saved Models

Model files are excluded from Git due to size.

Download the **saved_models/** folder from:

👉 [https://drive.google.com/drive/folders/1XqG_3399_OTNGNdEJ8jX81tDFSMMao1X?usp=sharing](https://drive.google.com/drive/folders/1XqG_3399_OTNGNdEJ8jX81tDFSMMao1X?usp=sharing)

Place it in the project root:

```
smartsolar/
└── saved_models/
```

---

### 3️⃣ Create Virtual Environment

```bash
python -m venv myenv
```

**Windows**

```bash
myenv\Scripts\activate
```

**Mac/Linux**

```bash
source myenv/bin/activate
```

---

## 🚀 Running the Application

### ▶ Start Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

* Backend: [http://localhost:8000](http://localhost:8000)
* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### ▶ Start Frontend (Streamlit)

Open **new terminal**

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

* Frontend: [http://localhost:8501](http://localhost:8501)

---

## 📊 Using the App

### ✔ Option 1 — Upload CSV

1. Prepare CSV with **40 rows** of hourly sensor data
2. Final row = forecast start timestamp
3. Upload in Streamlit UI
4. Use `sample_input_realistic.csv` as template

---

### ✔ Option 2 — Manual Entry

* Enter current sensor readings
* App auto-generates **35 context rows**

---

## 📄 Required CSV Columns

| Column   | Description                   | Unit             |
| -------- | ----------------------------- | ---------------- |
| datetime | Timestamp                     | YYYY-MM-DD HH:MM |
| GHI      | Global Horizontal Irradiance  | W/m²             |
| DNI      | Direct Normal Irradiance      | W/m²             |
| DHI      | Diffuse Horizontal Irradiance | W/m²             |
| T_amb    | Ambient Temperature           | °C               |
| RH       | Relative Humidity             | %                |
| WS       | Wind Speed                    | m/s              |
| WS_gust  | Wind Gust Speed               | m/s              |
| WD       | Wind Direction                | °                |
| WD_std   | Wind Direction Std Dev        | °                |
| BP       | Barometric Pressure           | hPa              |

---

## 🔧 Panel Settings (Sidebar)

* **Panel Area (m²)** → default: `10`
* **Panel Efficiency (%)** → default: `20`

---

## ⚡ Energy Formula

```
Energy (kWh) = GHI × Area × Efficiency / 1000
```

---

## 🧠 Model Pipeline

```
Raw sensor data (11 features)
        ↓
Feature Engineering (~80 features)
  · Cyclic encoding
  · Solar position
  · Lag features (1,2,3,6,12,24h)
  · Rolling stats, EMA, VMD modes
        ↓
RobustScaler transform
        ↓
Lookback sequence [12 × n_features] → flatten
        ↓
Stacking meta-features
(RF + ExtraTrees + Ridge + Lasso)
        ↓
Hybrid feature selection
(Mutual Info + F-stat + GBM importance)
        ↓
XGBoost models:
  h0 → t+1h
  h1 → t+2h
  h2 → t+3h
        ↓
Inverse scaling → GHI (W/m²)
        ↓
Solar Energy estimation (kWh)
```

---

## 🛠️ Requirements

* Python **3.10+**
* See:

  * `backend/requirements.txt`
  * `frontend/requirements.txt`

---

## 📌 Notes

* Ensure `saved_models/` exists before starting backend
* Backend must run **before** launching Streamlit frontend
* Recommended: run inside virtual environment



