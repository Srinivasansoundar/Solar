"""
Solar Forecasting FastAPI Backend
==================================
Endpoint: POST /predict
Window: (12,3) — predicts 3 hours ahead, converts GHI to energy (kWh)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Solar Energy Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ────────────────────────────────────────────────────────────────
# AFTER — point to your actual saved_models folder
import os
MODEL_DIR = r"C:\Users\Srinivasan\Documents\skills\Projects\pw2_new\smartsolar\saved_models"
WINDOW      = "(12,3)"
WINDOW_CFG  = {"lookback": 12, "horizon": 3, "tag": "12_3"}

# Panel defaults (can be overridden per request)
DEFAULT_PANEL_AREA_M2    = 10.0   # m²
DEFAULT_PANEL_EFFICIENCY = 0.20   # 20%

# ── Model store (loaded once at startup) ─────────────────────────────────────
store = {}


@app.on_event("startup")
def load_models():
    d   = MODEL_DIR
    tag = WINDOW_CFG["tag"]

    try:
        store["scaler_features"]  = joblib.load(os.path.join(d, "scaler_features.pkl"))
        store["scaler_target"]    = joblib.load(os.path.join(d, "scaler_target.pkl"))
        store["feature_names"]    = joblib.load(os.path.join(d, "feature_names.pkl"))

        # XGBoost — 3 horizon models
        store["xgb_models"] = []
        for h in range(3):
            store["xgb_models"].append(
                joblib.load(os.path.join(d, f"xgb_enhanced_{tag}_h{h}.pkl"))
            )

        # Stacking base models
        bp = os.path.join(d, f"base_models_{tag}.pkl")
        store["base_models"] = joblib.load(bp) if os.path.exists(bp) else None

        # Hybrid selected indices
        sp = os.path.join(d, f"selected_indices_{tag}.pkl")
        store["selected_indices"] = joblib.load(sp) if os.path.exists(sp) else None

        print("✓ All models loaded successfully")
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        raise


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"]      = df["datetime"].dt.hour
    df["month"]     = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["dayofweek"] = df["datetime"].dt.dayofweek

    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["solar_elevation"] = (90 - np.abs(df["hour"] - 12) * 7.5).clip(lower=0)
    df["clear_sky_index"] = np.where(
        df["solar_elevation"] > 0,
        df["GHI"] / (df["solar_elevation"] * 10 + 1), 0
    )

    df["DNI_DHI_ratio"] = np.where(df["DHI"] > 0, df["DNI"] / df["DHI"], 0)
    df["temp_humidity"] = df["T_amb"] * df["RH"]
    df["wind_pressure"] = df["WS"]   * df["BP"]

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"GHI_lag_{lag}"]   = df["GHI"].shift(lag)
        df[f"DNI_lag_{lag}"]   = df["DNI"].shift(lag)
        df[f"DHI_lag_{lag}"]   = df["DHI"].shift(lag)
        df[f"T_amb_lag_{lag}"] = df["T_amb"].shift(lag)

    for lag in [1, 3, 6]:
        df[f"GHI_diff_{lag}"] = df["GHI"] - df[f"GHI_lag_{lag}"]

    for w in [3, 6, 12, 24]:
        df[f"GHI_rolling_mean_{w}"] = df["GHI"].rolling(w, min_periods=1).mean()
        df[f"GHI_rolling_std_{w}"]  = df["GHI"].rolling(w, min_periods=1).std().fillna(0)
        df[f"GHI_rolling_min_{w}"]  = df["GHI"].rolling(w, min_periods=1).min()
        df[f"GHI_rolling_max_{w}"]  = df["GHI"].rolling(w, min_periods=1).max()

    for span in [3, 6, 12]:
        df[f"GHI_ema_{span}"] = df["GHI"].ewm(span=span, adjust=False).mean()

    for i in range(7):
        w = 2 ** (i + 1) + 1
        df[f"VMD_mode_{i+1}"] = df["GHI"].rolling(w, center=True, min_periods=1).mean()

    df["VMD_high_freq"] = df["VMD_mode_1"] + df["VMD_mode_2"]
    df["VMD_low_freq"]  = df["VMD_mode_6"] + df["VMD_mode_7"]

    # ── Explicit feature list matching training order ──────────────────────
    feature_list = (
        ["DNI", "DHI", "T_amb", "RH", "WS", "WS_gust", "WD", "WD_std", "BP",
         "hour_sin", "hour_cos", "month_sin", "month_cos",
         "dayofyear_sin", "dayofyear_cos", "dayofweek_sin", "dayofweek_cos",
         "solar_elevation", "clear_sky_index",
         "DNI_DHI_ratio", "temp_humidity", "wind_pressure"]
        + [f"{v}_lag_{l}" for l in [1,2,3,6,12,24]
                           for v in ["GHI","DNI","DHI","T_amb"]]
        + [f"GHI_diff_{l}" for l in [1,3,6]]
        + [f"GHI_rolling_{s}_{w}" for w in [3,6,12,24]
                                   for s in ["mean","std","min","max"]]
        + [f"GHI_ema_{sp}" for sp in [3,6,12]]
        + [f"VMD_mode_{i+1}" for i in range(7)]
        + ["VMD_high_freq", "VMD_low_freq"]
    )

    df = df.dropna(subset=feature_list).reset_index(drop=True)
    return df


def ghi_to_energy_kwh(ghi_wm2: float, area_m2: float, efficiency: float) -> float:
    """Convert GHI (W/m²) to energy generated (kWh) over 1 hour."""
    return max(0.0, round(ghi_wm2 * area_m2 * efficiency / 1000.0, 4))


def create_meta_features(X: np.ndarray, base_models: dict) -> np.ndarray:
    meta_cols = [m.predict(X).reshape(-1, 1) for m in base_models.values()]
    return np.hstack([X] + meta_cols)


def run_pipeline(df_eng: pd.DataFrame) -> list[float]:
    """Scale → sequence → stack → select → predict."""
    feat     = store["feature_names"]   # guaranteed same order as feature_list
    lookback = WINDOW_CFG["lookback"]   # 12

    X_scaled = store["scaler_features"].transform(df_eng[feat])
    X_seq    = X_scaled[-lookback:].reshape(1, -1)   # [1, 12 × n_features]

    # Stack meta-features
    if store["base_models"]:
        X_seq = create_meta_features(X_seq, store["base_models"])

    # Hybrid feature selection
    if store["selected_indices"] is not None:
        X_seq = X_seq[:, store["selected_indices"]]

    # Predict 3 horizon steps
    preds = []
    for model in store["xgb_models"]:
        y_s   = model.predict(X_seq)
        y_inv = store["scaler_target"].inverse_transform(
                    y_s.reshape(-1, 1)).flatten()
        preds.append(float(np.clip(y_inv[0], 0, None)))

    return preds

# ── Schemas ───────────────────────────────────────────────────────────────────

class HistoricalRow(BaseModel):
    datetime  : str
    GHI       : float
    DNI       : float
    DHI       : float
    T_amb     : float
    RH        : float
    WS        : float
    WS_gust   : float
    WD        : float
    WD_std    : float
    BP        : float


class PredictRequest(BaseModel):
    historical_data : list[HistoricalRow] = Field(
        ...,
        description="At least 30 hourly rows of sensor data for context"
    )
    panel_area_m2   : float = Field(DEFAULT_PANEL_AREA_M2,   gt=0)
    panel_efficiency: float = Field(DEFAULT_PANEL_EFFICIENCY, gt=0, le=1)


class HorizonResult(BaseModel):
    step          : int
    hour_label    : str
    ghi_wm2       : float
    energy_kwh    : float
    irradiance_category: str


class PredictResponse(BaseModel):
    window          : str
    forecasts       : list[HorizonResult]
    total_energy_kwh: float
    last_timestamp  : str
    panel_area_m2   : float
    panel_efficiency: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def irradiance_category(ghi: float) -> str:
    if ghi < 100:  return "Low ☁️"
    if ghi < 400:  return "Moderate 🌤️"
    if ghi < 700:  return "High 🌥️"
    return "Excellent ☀️"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Solar Forecasting API — POST /predict to get forecasts"}


@app.get("/health")
def health():
    return {
        "status" : "ok",
        "models_loaded": len(store.get("xgb_models", [])) == 3,
        "window" : WINDOW,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.historical_data) < 30:
        raise HTTPException(
            status_code=422,
            detail="Need at least 30 rows of historical data for context."
        )

    # Build DataFrame
    df_raw = pd.DataFrame([r.dict() for r in req.historical_data])

    try:
        df_eng = engineer_features(df_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering error: {e}")

    if len(df_eng) < WINDOW_CFG["lookback"]:
        raise HTTPException(
            status_code=422,
            detail=f"After feature engineering only {len(df_eng)} rows remain. "
                   f"Need at least {WINDOW_CFG['lookback']}."
        )

    try:
        ghi_preds = run_pipeline(df_eng)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    last_ts  = pd.to_datetime(df_eng["datetime"].iloc[-1])
    forecasts = []
    for h, ghi in enumerate(ghi_preds, 1):
        energy   = ghi_to_energy_kwh(ghi, req.panel_area_m2, req.panel_efficiency)
        hour_lbl = (last_ts + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
        forecasts.append(HorizonResult(
            step              = h,
            hour_label        = hour_lbl,
            ghi_wm2           = round(ghi, 2),
            energy_kwh        = energy,
            irradiance_category = irradiance_category(ghi),
        ))

    return PredictResponse(
        window           = WINDOW,
        forecasts        = forecasts,
        total_energy_kwh = round(sum(f.energy_kwh for f in forecasts), 4),
        last_timestamp   = str(last_ts),
        panel_area_m2    = req.panel_area_m2,
        panel_efficiency = req.panel_efficiency,
    )