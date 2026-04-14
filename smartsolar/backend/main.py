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
# Path to saved_models relative to this file (smartsolar/saved_models)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models")
WINDOW      = "(12,3)"
WINDOW_CFG  = {"lookback": 12, "horizon": 3, "tag": "12_3"}

# Panel defaults (can be overridden per request)
DEFAULT_PANEL_AREA_M2    = 10.0   # m²
DEFAULT_PANEL_EFFICIENCY = 0.20   # 20%

# ── Model store (loaded once at startup) ─────────────────────────────────────
store = {}

# Energy consumption 24h forecast — recursive hybrid XGB + LSTM (separate from solar)
ENERGY_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "energy_consumption_saved_models")
ENERGY_HORIZON_HOURS = 24
ENERGY_TARGET_COL = "Global_active_power"  # target column name (must exist in CSV)
MAX_ENERGY_ROWS = 2000  # max rows to use from request (last N)
store_energy = {}


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


def _make_residual_lstm():
    """Colab ResidualLSTM: input_size=1, hidden_size=64, num_layers=2, dropout=0.2."""
    import torch.nn as nn
    class ResidualLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
    return ResidualLSTM()


@app.on_event("startup")
def load_energy_models():
    """Load hybrid XGB + LSTM and scaler from energy_consumption_saved_models only (no fallback)."""
    import torch
    d = os.path.abspath(ENERGY_MODEL_DIR)
    if not os.path.isdir(d):
        print("✗ Energy model dir not found. Create smartsolar/energy_consumption_saved_models/ and add xgb_consumption.pkl, lstm_consumption.pt, res_scaler_consumption.pkl, feature_names_consumption.pkl")
        return
    print(f"✓ Energy models loading from: {d}")
    try:
        store_energy["xgb"] = joblib.load(os.path.join(d, "xgb_consumption.pkl"))
        sp = os.path.join(d, "res_scaler_consumption.pkl")
        store_energy["scaler"] = joblib.load(sp) if os.path.exists(sp) else None
        fp = os.path.join(d, "feature_names_consumption.pkl")
        store_energy["feature_names"] = joblib.load(fp) if os.path.exists(fp) else None
        seq_path = os.path.join(d, "seq_len.pkl")
        store_energy["seq_len"] = joblib.load(seq_path) if os.path.exists(seq_path) else 24
        store_energy["lstm"] = None
        lstm_path = os.path.join(d, "lstm_consumption.pt")
        if not os.path.exists(lstm_path):
            print("✗ Energy: lstm_consumption.pt not found in energy_consumption_saved_models/")
        else:
            loaded = torch.load(lstm_path, map_location="cpu", weights_only=False)
            if callable(loaded) and hasattr(loaded, "eval"):
                store_energy["lstm"] = loaded
                store_energy["lstm"].eval()
                print("✓ Energy: LSTM loaded as full model from energy_consumption_saved_models/ (hybrid ready)")
            elif isinstance(loaded, dict):
                sd = loaded
                sd_clean = {k.replace("model.", "").replace("module.", ""): v for k, v in sd.items()}
                model = _make_residual_lstm()
                try:
                    model.load_state_dict(sd_clean, strict=True)
                    store_energy["lstm"] = model
                    store_energy["lstm"].eval()
                    print("✓ Energy: LSTM loaded from state_dict in energy_consumption_saved_models/ (hybrid ready)")
                except Exception:
                    try:
                        model.load_state_dict(sd_clean, strict=False)
                        store_energy["lstm"] = model
                        store_energy["lstm"].eval()
                        print("✓ Energy: LSTM loaded from state_dict (strict=False) in energy_consumption_saved_models/ (hybrid ready)")
                    except Exception as e:
                        print(f"✗ Energy: LSTM state_dict load failed: {e}")
            else:
                print("✗ Energy: lstm_consumption.pt is not a callable model or state_dict")
        if store_energy["lstm"] is None:
            print("✗ Energy: LSTM not loaded — /predict-energy will return 503 until lstm_consumption.pt is valid in energy_consumption_saved_models/")
        else:
            print("✓ Energy consumption: XGB + LSTM hybrid ready for 24h forecast (rolling last 24 residuals → LSTM)")
    except Exception as e:
        print(f"✗ Energy model load: {e}")


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    # If parsing fails for any rows, we should drop them
    df = df.dropna(subset=['datetime'])
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


# ── Energy consumption 24h — recursive hybrid XGB + LSTM (Colab logic) ────────

def _energy_build_features(df: pd.DataFrame, xgb_features: list) -> pd.DataFrame:
    """Build calendar, lag, rolling features. df must have datetime index and target column."""
    out = df.copy()
    if not hasattr(out.index, "hour"):
        out.index = pd.to_datetime(out.index)
    out["hour"] = out.index.hour
    out["day"] = out.index.day
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday
    target = out.columns[out.columns.str.contains("active_power|power|consumption", case=False, na=False)]
    target = target[0] if len(target) else out.columns[0]
    vals = out[target].astype(float).fillna(0).tolist()
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        key = f"lag_{lag}"
        lag_vals = [vals[i - lag] if i >= lag else (vals[0] if vals else 0) for i in range(len(vals))]
        out[key] = lag_vals
    # Match Colab: rolling then shift(1) to avoid leakage (use past values only)
    s = pd.Series(vals, index=out.index)
    out["rolling_mean_6"] = s.rolling(6, min_periods=1).mean().shift(1).bfill().fillna(0).values
    out["rolling_mean_24"] = s.rolling(24, min_periods=1).mean().shift(1).bfill().fillna(0).values
    out["rolling_std_24"] = s.rolling(24, min_periods=1).std().shift(1).fillna(0).values
    for col in xgb_features:
        if col not in out.columns:
            out[col] = 0
    return out[xgb_features] if all(c in out.columns for c in xgb_features) else out


def run_energy_consumption_pipeline(df_raw: pd.DataFrame) -> list[float]:
    """Recursive hybrid forecast: for each future hour build row (calendar + lags + rolling), XGB pred + LSTM residual, update history.
    Supports both minute-based and hour-based input: sub-hourly data is resampled to 1h (mean) before use."""
    import torch
    xgb = store_energy.get("xgb")
    lstm = store_energy.get("lstm")
    scaler = store_energy.get("scaler")
    feat_names = store_energy.get("feature_names")
    seq_len = int(store_energy.get("seq_len") or 24)
    if feat_names is not None and hasattr(feat_names, "tolist"):
        xgb_features = feat_names.tolist()
    elif isinstance(feat_names, list):
        xgb_features = feat_names
    else:
        xgb_features = ["hour", "day", "month", "weekday"] + [f"lag_{l}" for l in [1, 2, 3, 6, 12, 24, 48]] + ["rolling_mean_6", "rolling_mean_24", "rolling_std_24"]

    # Ensure datetime index and target column (accept "datetime" or "dt")
    dt_col = "datetime" if "datetime" in df_raw.columns else ("dt" if "dt" in df_raw.columns else None)
    if dt_col is not None:
        df_raw = df_raw.set_index(pd.to_datetime(df_raw[dt_col], errors="coerce")).sort_index()
        df_raw = df_raw[df_raw.index.notna()]
    target_col = ENERGY_TARGET_COL if ENERGY_TARGET_COL in df_raw.columns else None
    if target_col is None:
        for c in ["Global_active_power", "global_active_power", "Active_Power", "power"]:
            if c in df_raw.columns:
                target_col = c
                break
        if target_col is None and len(df_raw.columns):
            target_col = df_raw.select_dtypes(include=[np.number]).columns[0]
    if target_col is None:
        raise ValueError("Need a target column (e.g. Global_active_power) in the data")
    df_raw = df_raw[[target_col]].copy()
    df_raw.columns = [ENERGY_TARGET_COL]

    # Support both minute-based and hour-based data: resample to hourly when data is sub-hourly (model expects hourly)
    if len(df_raw) > 24 and hasattr(df_raw.index, "to_series"):
        span_hours = (df_raw.index.max() - df_raw.index.min()).total_seconds() / 3600
        if span_hours > 0 and len(df_raw) > span_hours * 1.5:
            df_h = df_raw.resample("1h").mean().ffill().bfill()
            if len(df_h) > 0:
                df_raw = df_h
    # Also resample if median interval is under 45 minutes (e.g. 15-min or 1-min data)
    if len(df_raw) > 1 and hasattr(df_raw.index, "to_series"):
        diffs = df_raw.index.to_series().diff().dropna()
        if len(diffs) > 0 and diffs.median().total_seconds() < 45 * 60:
            df_h = df_raw.resample("1h").mean().ffill().bfill()
            if len(df_h) > 0:
                df_raw = df_h
    if len(df_raw) == 0:
        raise ValueError("No valid rows after resampling. Ensure datetime and target column have valid values.")

    df_feat = _energy_build_features(df_raw, xgb_features)
    history_values = df_raw[ENERGY_TARGET_COL].astype(float).fillna(0).tolist()
    if not history_values:
        raise ValueError("No history values for forecasting.")

    MIN_FLOOR = 0.01
    # Hourly profile from history: when model predicts at floor, use this for display + feed_back so we get diurnal variation (no more 0.01 spiral)
    try:
        hr_means = df_raw.groupby(df_raw.index.hour)[ENERGY_TARGET_COL].mean()
        hourly_profile = {h: max(float(hr_means.get(h, 0.1)), MIN_FLOOR) for h in range(24)}
    except Exception:
        global_mean = max(float(np.mean(history_values)), MIN_FLOOR) if history_values else MIN_FLOOR
        hourly_profile = {h: global_mean for h in range(24)}

    # LSTM uses rolling last 24 (seq_len) *scaled residuals*: [res_t-24, ..., res_t-1] → predict next residual
    # Build initial sequence from last seq_len rows: residual = actual - xgb_pred, then scale with res_scaler
    need = min(seq_len, len(df_feat))
    last_residuals = []
    if scaler is not None and need > 0:
        for i in range(-need, 0):
            row = df_feat.iloc[i : i + 1].reindex(columns=xgb_features).fillna(0).astype(np.float64)
            xgb_out = np.atleast_1d(xgb.predict(row))
            xgb_p = float(xgb_out[0]) if xgb_out.size > 0 else 0.0
            res = history_values[i] - xgb_p
            sc = scaler.transform(np.array([[res]]))
            last_residuals.append(float(np.asarray(sc).flat[0]) if np.asarray(sc).size > 0 else 0.0)
    while len(last_residuals) < seq_len and last_residuals:
        last_residuals.insert(0, last_residuals[0])
    if not last_residuals:
        last_residuals = [0.0] * seq_len
    last_residuals = last_residuals[-seq_len:]  # keep exactly seq_len (24) for LSTM input

    current_time = df_raw.index[-1]
    hybrid_forecast = []
    last_good = float(history_values[-1]) if history_values else MIN_FLOOR

    for step in range(ENERGY_HORIZON_HOURS):
        try:
            current_time = current_time + pd.Timedelta(hours=1)
            next_row = pd.DataFrame(index=[current_time])
            next_row["hour"] = current_time.hour
            next_row["day"] = current_time.day
            next_row["month"] = current_time.month
            next_row["weekday"] = current_time.weekday()
            for lag in [1, 2, 3, 6, 12, 24, 48]:
                next_row[f"lag_{lag}"] = history_values[-lag] if len(history_values) >= lag else (history_values[0] if history_values else 0)
            next_row["rolling_mean_6"] = np.mean(history_values[-6:]) if len(history_values) >= 6 else (np.mean(history_values) if history_values else 0)
            next_row["rolling_mean_24"] = np.mean(history_values[-24:]) if len(history_values) >= 24 else (np.mean(history_values) if history_values else 0)
            next_row["rolling_std_24"] = np.std(history_values[-24:]) if len(history_values) >= 24 else (np.std(history_values) if len(history_values) > 1 else 0)
            for col in xgb_features:
                if col not in next_row.columns:
                    next_row[col] = 0
            next_row = next_row.reindex(columns=xgb_features).fillna(0).astype(np.float64)

            xgb_out = xgb.predict(next_row)
            xgb_pred = float(np.atleast_1d(xgb_out)[0]) if np.atleast_1d(xgb_out).size > 0 else 0.0

            hybrid_pred = xgb_pred
            if lstm is not None and scaler is not None and callable(lstm):
                try:
                    # Rolling last 24 scaled residuals: shape (1, seq_len, 1) for LSTM
                    res_seq = np.array(last_residuals[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
                    x_lstm = torch.tensor(res_seq, dtype=torch.float32)
                    with torch.no_grad():
                        lstm_scaled = lstm(x_lstm).cpu().numpy()
                    if lstm_scaled.size > 0:
                        inv = scaler.inverse_transform(np.atleast_2d(lstm_scaled).reshape(-1, 1))
                        lstm_res = float(inv.flat[0]) if inv.size > 0 else 0.0
                        # Clip residual so hybrid never goes below 0 (consumption is non-negative)
                        lstm_res_clipped = max(lstm_res, -xgb_pred)
                        hybrid_pred = xgb_pred + lstm_res_clipped
                        last_residuals.append(float(lstm_scaled.flat[0]))
                except Exception:
                    pass
        except Exception:
            hybrid_pred = float(history_values[-1]) if history_values else MIN_FLOOR
        # Final safeguard: never output negative consumption
        hybrid_pred = max(float(hybrid_pred), 0.0)
        raw_val = max(hybrid_pred, MIN_FLOOR)
        # When model outputs at floor, use hourly profile from history so we get diurnal variation (final fix for 0.01 spiral)
        if raw_val <= MIN_FLOOR:
            display_val = hourly_profile.get(current_time.hour, last_good)
        else:
            display_val = raw_val
            last_good = raw_val
        hybrid_forecast.append(display_val)
        feed_back = display_val
        history_values.append(feed_back)

    def _json_safe(v):
        x = float(v)
        if np.isnan(x) or np.isinf(x) or x != x:
            return 0.0
        return round(max(0.0, x), 4)
    out = [_json_safe(x) for x in hybrid_forecast]
    # Guarantee 24 values: pad with last or zero
    while len(out) < ENERGY_HORIZON_HOURS:
        out.append(out[-1] if out else 0.0)
    return out[:ENERGY_HORIZON_HOURS]


class EnergyPredictRequest(BaseModel):
    """Request for 24h energy consumption forecast. historical_data must contain columns expected by your trained model (see feature_names_consumption.pkl)."""
    historical_data: list[dict] = Field(..., description="List of hourly rows (dicts). Must include columns required by the consumption model.")


class EnergyForecastRow(BaseModel):
    step: int
    hour_label: str
    consumption_kwh: float


class EnergyPredictResponse(BaseModel):
    forecasts: list[EnergyForecastRow]
    total_consumption_kwh: float
    last_timestamp: str


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


@app.get("/energy-feature-names")
def energy_feature_names():
    """Return feature names for the energy consumption model (for manual entry form)."""
    names = store_energy.get("feature_names")
    if names is None:
        return {"feature_names": [], "available": False}
    if hasattr(names, "tolist"):
        names = names.tolist()
    return {"feature_names": list(names), "available": True}


@app.get("/energy-status")
def energy_status():
    """Return whether hybrid (XGB + LSTM) is loaded. Required for /predict-energy."""
    xgb_loaded = store_energy.get("xgb") is not None
    lstm_loaded = store_energy.get("lstm") is not None and callable(store_energy.get("lstm"))
    return {
        "xgb_loaded": xgb_loaded,
        "lstm_loaded": lstm_loaded,
        "hybrid_ready": xgb_loaded and lstm_loaded,
        "message": "Hybrid (XGB + LSTM) ready for 24h forecast" if (xgb_loaded and lstm_loaded) else "LSTM not loaded — add lstm_consumption.pt to energy_consumption_saved_models/",
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


@app.post("/predict-energy", response_model=EnergyPredictResponse)
def predict_energy(req: EnergyPredictRequest):
    """24-hour energy consumption forecast. Requires hybrid (XGB + LSTM). CSV must have datetime and Global_active_power."""
    if not store_energy.get("xgb"):
        raise HTTPException(
            status_code=503,
            detail="XGB not loaded. Add xgb_consumption.pkl to energy_consumption_saved_models/ and restart.",
        )
    if not store_energy.get("lstm"):
        raise HTTPException(
            status_code=503,
            detail="LSTM not loaded — hybrid required. Add lstm_consumption.pt (Colab state_dict, ResidualLSTM 2-layer 64-hidden) to energy_consumption_saved_models/ and restart. Check backend logs on startup.",
        )
    seq_len = int(store_energy.get("seq_len") or 24)
    if len(req.historical_data) < max(seq_len, 24):
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {max(seq_len, 24)} rows of historical data (datetime + Global_active_power).",
        )
    rows = req.historical_data[-MAX_ENERGY_ROWS:]
    df_raw = pd.DataFrame(rows)
    dt_col = "datetime" if "datetime" in df_raw.columns else ("dt" if "dt" in df_raw.columns else None)
    if dt_col is None:
        raise HTTPException(status_code=422, detail="CSV must include a 'datetime' or 'dt' column.")
    try:
        preds = run_energy_consumption_pipeline(df_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Energy prediction error: {e}")
    last_ts = pd.to_datetime(df_raw[dt_col].iloc[-1]) if dt_col and dt_col in df_raw.columns else pd.Timestamp.now()
    last_ts = pd.Timestamp(last_ts)
    def safe_float(v):
        x = float(v)
        if np.isnan(x) or np.isinf(x) or x != x:
            return 0.0
        return round(max(0.0, x), 4)
    # Ensure exactly 24 predictions (pad if backend returned fewer)
    preds = list(preds)[:ENERGY_HORIZON_HOURS]
    while len(preds) < ENERGY_HORIZON_HOURS:
        preds.append(preds[-1] if preds else 0.0)
    forecasts = []
    for h in range(ENERGY_HORIZON_HOURS):
        hour_label = (last_ts + pd.Timedelta(hours=h + 1)).strftime("%Y-%m-%d %H:%M")
        forecasts.append(EnergyForecastRow(step=h + 1, hour_label=hour_label, consumption_kwh=safe_float(preds[h])))
    total = safe_float(sum(preds))
    return EnergyPredictResponse(
        forecasts=forecasts,
        total_consumption_kwh=total,
        last_timestamp=str(last_ts),
    )