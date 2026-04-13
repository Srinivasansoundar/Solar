"""
Solar Energy Forecasting — Streamlit Frontend
==============================================
Tabs: ☀️ Solar Forecast | ⚡ Energy Consumption | 💰 Tariff Forecast
"""

import sys
import os
import logging
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Suppress the harmless "Examining the path of torch.classes" Streamlit watcher warning
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ── Inject tariff model path ───────────────────────────────────────────────────
_TARIFF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tariff_forcasting_model")
)
if _TARIFF_DIR not in sys.path:
    sys.path.insert(0, _TARIFF_DIR)

from lstm_model import predict_tariff  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSolar Intelligent Energy Decision Engine ",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --sun: #F5A623;
    --amber: #E8820C;
    --deep: #0D1117;
    --card: #161B22;
    --card2: #1C2128;
    --border: #30363D;
    --text: #E6EDF3;
    --muted: #8B949E;
    --green: #3FB950;
    --blue: #58A6FF;
    --purple: #BC8CFF;
    --tariff-peak: #FF6B6B;
    --tariff-off:  #3FB950;
    --tariff-mid:  #F5A623;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--deep);
    color: var(--text);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #F5A623 0%, #FF6B35 50%, #F5A623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}

.main-title-blue {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58A6FF 0%, #7B61FF 50%, #58A6FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}

.main-title-tariff {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #BC8CFF 0%, #FF6B6B 50%, #BC8CFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.3rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F5A623, #FF6B35);
}

.metric-card-blue::before {
    background: linear-gradient(90deg, #58A6FF, #7B61FF);
}

.metric-card-tariff::before {
    background: linear-gradient(90deg, #BC8CFF, #FF6B6B);
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #F5A623;
    margin: 0.3rem 0;
}

.metric-value-blue { color: #58A6FF; }
.metric-value-tariff { color: #BC8CFF; }

.metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.metric-unit {
    font-size: 0.85rem;
    color: var(--muted);
}

.forecast-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    transition: transform 0.15s, border-color 0.15s;
}
.forecast-card:hover {
    transform: translateX(3px);
    border-color: rgba(245,166,35,0.4);
}

.hour-badge {
    background: linear-gradient(135deg, #F5A623, #FF6B35);
    color: #0D1117;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    white-space: nowrap;
    min-width: 50px;
    text-align: center;
}

.tariff-badge-peak {
    background: linear-gradient(135deg, #FF6B6B, #e03c3c);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    white-space: nowrap;
    text-align: center;
}

.tariff-badge-off {
    background: linear-gradient(135deg, #3FB950, #2da540);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    white-space: nowrap;
    text-align: center;
}

.tariff-badge-mid {
    background: linear-gradient(135deg, #F5A623, #E8820C);
    color: #0D1117;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    white-space: nowrap;
    text-align: center;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    border-left: 3px solid #F5A623;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

.section-header-blue { border-left-color: #58A6FF; }
.section-header-tariff { border-left-color: #BC8CFF; }

.info-box {
    background: rgba(245, 166, 35, 0.08);
    border: 1px solid rgba(245, 166, 35, 0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.info-box-tariff {
    background: rgba(188, 140, 255, 0.08);
    border: 1px solid rgba(188, 140, 255, 0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.recommendation-box {
    background: linear-gradient(135deg, rgba(63,185,80,0.12), rgba(63,185,80,0.05));
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.warning-box {
    background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.05));
    border: 1px solid rgba(255,107,107,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #F5A623 0%, #E8820C 100%);
    color: #0D1117;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.03em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(245, 166, 35, 0.4);
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}

.total-energy-box {
    background: linear-gradient(135deg, rgba(245,166,35,0.15), rgba(232,130,12,0.08));
    border: 1px solid rgba(245,166,35,0.4);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.total-tariff-box {
    background: linear-gradient(135deg, rgba(188,140,255,0.15), rgba(188,140,255,0.05));
    border: 1px solid rgba(188,140,255,0.4);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.total-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #F5A623;
}

.total-value-tariff {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #BC8CFF;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: var(--muted);
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #F5A623, #E8820C) !important;
    color: #0D1117 !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ SmartSolar Intelligent Energy Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Solar Generation · Energy Consumption · Tariff Intelligence </div>', unsafe_allow_html=True)

# ── Session State — shared inputs & results ───────────────────────────────────
_now = datetime.now()
if "shared_dt" not in st.session_state:
    st.session_state["shared_dt"] = _now.strftime("%Y-%m-%d %H:00")
if "shared_temp" not in st.session_state:
    st.session_state["shared_temp"] = 28.0
if "ss_solar" not in st.session_state:
    st.session_state["ss_solar"] = None
if "ss_energy" not in st.session_state:
    st.session_state["ss_energy"] = None
if "ss_tariff" not in st.session_state:
    st.session_state["ss_tariff"] = None

# ── Common Settings Sidebar (top, rendered once) ──────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 Common Settings")
    st.markdown('<div class="info-box" style="background:rgba(88,166,255,0.07);border-color:rgba(88,166,255,0.25);">These values auto-fill matching fields across all 3 forecast tabs.</div>', unsafe_allow_html=True)
    _shared_dt = st.text_input(
        "📅 Reference Datetime",
        value=st.session_state["shared_dt"],
        help="Format: YYYY-MM-DD HH:MM — pre-fills Solar, Energy, and Tariff tabs",
        key="_sidebar_shared_dt"
    )
    _shared_temp = st.number_input(
        "🌡️ Ambient Temperature (°C)",
        min_value=-10.0, max_value=55.0,
        value=float(st.session_state["shared_temp"]),
        step=0.5,
        help="Pre-fills T_amb (Solar) and Temperature (Tariff) fields",
        key="_sidebar_shared_temp"
    )
    # persist back to session state
    st.session_state["shared_dt"]   = _shared_dt
    st.session_state["shared_temp"] = _shared_temp
    st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "☀️  Solar Forecast",
    "⚡  Energy Consumption",
    "💰  Tariff Forecast",
    "📊  Combined Dashboard",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Solar 3h Forecast
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-title">☀️ Solar Energy Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">3-Hour Ahead GHI Prediction → Energy Generation  ·  XGBoost Meta-Model  ·  Window (12,3)</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Panel Settings")
        st.markdown('<div class="info-box">Configure your solar panel parameters to calculate actual energy output.</div>', unsafe_allow_html=True)

        panel_area = st.number_input(
            "Panel Area (m²)",
            min_value=0.1, max_value=10000.0,
            value=10.0, step=0.5,
            help="Total solar panel surface area"
        )
        panel_efficiency = st.slider(
            "Panel Efficiency (%)",
            min_value=5, max_value=35,
            value=20, step=1,
            help="Typical monocrystalline: 20–22%, polycrystalline: 15–17%"
        ) / 100.0

        st.markdown("---")
        st.markdown("### 🔗 API Status")
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.status_code == 200 and r.json().get("models_loaded"):
                st.success("Backend connected ✓")
            else:
                st.warning("Backend reachable but models not loaded")
        except Exception:
            st.error("Backend offline — start FastAPI server")

        st.markdown("---")
        st.caption("Model: XGBoost Enhanced · Window (12,3) · Predicts t+1h, t+2h, t+3h")

    # ── Input Mode ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📥 Input Historical Sensor Data</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["Manual Entry (last row + auto-fill context)", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="info-box">⚠️ The model needs at least <b>30 hourly rows</b> of context. Upload a CSV with full history, or use manual entry which auto-generates realistic context rows around your input.</div>', unsafe_allow_html=True)


    def generate_context(row: dict, n: int = 35) -> list[dict]:
        """Generate n synthetic historical rows ending at the given row."""
        base_dt = pd.to_datetime(row["datetime"]) - pd.Timedelta(hours=n)
        rows = []
        for i in range(n):
            dt   = base_dt + pd.Timedelta(hours=i)
            hour = dt.hour
            elev = max(0, 90 - abs(hour - 12) * 7.5)
            ghi  = max(0, elev * 10 + np.random.normal(0, 20))
            rows.append({
                "datetime" : dt.strftime("%Y-%m-%d %H:%M"),
                "GHI"      : round(ghi, 2),
                "DNI"      : round(max(0, ghi * 0.7  + np.random.normal(0, 15)), 2),
                "DHI"      : round(max(0, ghi * 0.25 + np.random.normal(0, 8)),  2),
                "T_amb"    : round(row["T_amb"] + np.random.normal(0, 1.5), 2),
                "RH"       : round(np.clip(row["RH"]   + np.random.normal(0, 3), 10, 100), 2),
                "WS"       : round(max(0, row["WS"]    + np.random.normal(0, 0.5)), 2),
                "WS_gust"  : round(max(0, row["WS_gust"] + np.random.normal(0, 0.8)), 2),
                "WD"       : round(row["WD"] % 360, 2),
                "WD_std"   : round(abs(row["WD_std"] + np.random.normal(0, 2)), 2),
                "BP"       : round(row["BP"] + np.random.normal(0, 1), 2),
            })
        rows.append(row)
        return rows


    historical_data = None

    if input_mode == "Manual Entry (last row + auto-fill context)":
        st.markdown('<div class="section-header">🌡️ Current Sensor Readings</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            dt_input = st.text_input("Datetime",
                                      value=st.session_state.get("shared_dt", datetime.now().strftime("%Y-%m-%d %H:00")),
                                      help="Format: YYYY-MM-DD HH:MM — synced from Common Settings")
            GHI      = st.number_input("GHI — Global Horizontal Irradiance (W/m²)",
                                        min_value=0.0, max_value=1500.0, value=450.0, step=10.0)
            DNI      = st.number_input("DNI — Direct Normal Irradiance (W/m²)",
                                        min_value=0.0, max_value=1200.0, value=320.0, step=10.0)
            DHI      = st.number_input("DHI — Diffuse Horizontal Irradiance (W/m²)",
                                        min_value=0.0, max_value=600.0,  value=130.0, step=5.0)

        with col2:
            T_amb    = st.number_input("T_amb — Ambient Temperature (°C)",
                                        min_value=-20.0, max_value=60.0,
                                        value=float(st.session_state.get("shared_temp", 28.0)), step=0.5)
            RH       = st.number_input("RH — Relative Humidity (%)",
                                        min_value=0.0, max_value=100.0, value=55.0, step=1.0)
            WS       = st.number_input("WS — Wind Speed (m/s)",
                                        min_value=0.0, max_value=50.0, value=3.5, step=0.1)

        with col3:
            WS_gust  = st.number_input("WS_gust — Wind Gust (m/s)",
                                        min_value=0.0, max_value=80.0, value=5.2, step=0.1)
            WD       = st.number_input("WD — Wind Direction (°)",
                                        min_value=0.0, max_value=360.0, value=180.0, step=5.0)
            WD_std   = st.number_input("WD_std — Wind Direction Std Dev (°)",
                                        min_value=0.0, max_value=180.0, value=15.0, step=1.0)
            BP       = st.number_input("BP — Barometric Pressure (hPa)",
                                        min_value=800.0, max_value=1100.0, value=1013.0, step=0.5)

        current_row = {
            "datetime": dt_input, "GHI": GHI, "DNI": DNI, "DHI": DHI,
            "T_amb": T_amb, "RH": RH, "WS": WS, "WS_gust": WS_gust,
            "WD": WD, "WD_std": WD_std, "BP": BP,
        }
        historical_data = generate_context(current_row, n=35)

    else:
        uploaded = st.file_uploader(
            "Upload CSV (columns: datetime, GHI, DNI, DHI, T_amb, RH, WS, WS_gust, WD, WD_std, BP)",
            type=["csv"]
        )
        if uploaded:
            df_up = pd.read_csv(uploaded)
            required = ["datetime","GHI","DNI","DHI","T_amb","RH","WS","WS_gust","WD","WD_std","BP"]
            missing  = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif len(df_up) < 30:
                st.error(f"Need at least 30 rows, file has {len(df_up)}.")
            else:
                historical_data = df_up[required].to_dict(orient="records")
                st.success(f"✓ {len(historical_data)} rows loaded")

    # ── Predict Button ────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Generate 3-Hour Solar Forecast", key="solar_btn")

    if predict_clicked:
        if historical_data is None:
            st.warning("Please provide input data first.")
        else:
            with st.spinner("Running XGBoost meta-model pipeline..."):
                payload = {
                    "historical_data"  : historical_data,
                    "panel_area_m2"    : panel_area,
                    "panel_efficiency" : panel_efficiency,
                }
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

                    if resp.status_code == 200:
                        data      = resp.json()
                        forecasts = data["forecasts"]

                        st.markdown('<div class="section-header">📊 Forecast Summary</div>', unsafe_allow_html=True)

                        c1, c2, c3, c4 = st.columns(4)
                        avg_ghi    = np.mean([f["ghi_wm2"]    for f in forecasts])
                        max_ghi    = np.max( [f["ghi_wm2"]    for f in forecasts])
                        total_kwh  = data["total_energy_kwh"]
                        avg_kwh    = np.mean([f["energy_kwh"] for f in forecasts])

                        # ── Store results for Combined Dashboard ──────────────
                        st.session_state["ss_solar"] = {
                            "total_kwh":    total_kwh,
                            "forecasts":    forecasts,
                            "panel_area":   panel_area,
                            "panel_eff":    panel_efficiency,
                            "avg_ghi":      float(avg_ghi),
                        }

                        with c1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Avg GHI</div>
                                <div class="metric-value">{avg_ghi:.0f}</div>
                                <div class="metric-unit">W/m²</div>
                            </div>""", unsafe_allow_html=True)

                        with c2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Peak GHI</div>
                                <div class="metric-value">{max_ghi:.0f}</div>
                                <div class="metric-unit">W/m²</div>
                            </div>""", unsafe_allow_html=True)

                        with c3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Energy</div>
                                <div class="metric-value">{total_kwh:.3f}</div>
                                <div class="metric-unit">kWh (3 hrs)</div>
                            </div>""", unsafe_allow_html=True)

                        with c4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Avg / Hour</div>
                                <div class="metric-value">{avg_kwh:.3f}</div>
                                <div class="metric-unit">kWh</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="total-energy-box">
                            <div class="metric-label">⚡ Total Solar Energy Generated (next 3 hours)</div>
                            <div class="total-value">{total_kwh:.4f} kWh</div>
                            <div class="metric-unit">
                                {panel_area} m² panel  ·  {panel_efficiency*100:.0f}% efficiency
                            </div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown('<div class="section-header">🕐 Hourly Breakdown</div>', unsafe_allow_html=True)

                        col_a, col_b = st.columns([3, 2])

                        with col_a:
                            for f in forecasts:
                                st.markdown(f"""
                                <div class="forecast-card">
                                    <div class="hour-badge">t+{f['step']}h</div>
                                    <div style="flex:1">
                                        <div style="font-family:'Syne',sans-serif;font-weight:600;font-size:0.95rem;">
                                            {f['hour_label']}
                                        </div>
                                        <div style="color:#8B949E;font-size:0.82rem;margin-top:2px;">
                                            {f['irradiance_category']}
                                        </div>
                                    </div>
                                    <div style="text-align:right">
                                        <div style="font-family:'Syne',sans-serif;font-weight:700;
                                                    color:#F5A623;font-size:1.1rem;">
                                            {f['ghi_wm2']:.1f} W/m²
                                        </div>
                                        <div style="color:#3FB950;font-weight:500;font-size:0.9rem;">
                                            {f['energy_kwh']:.4f} kWh
                                        </div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                        with col_b:
                            chart_df = pd.DataFrame({
                                "Hour"       : [f["hour_label"].split(" ")[1] for f in forecasts],
                                "GHI (W/m²)" : [f["ghi_wm2"]    for f in forecasts],
                                "Energy (kWh)": [f["energy_kwh"] for f in forecasts],
                            }).set_index("Hour")

                            st.markdown("**GHI Forecast (W/m²)**")
                            st.bar_chart(chart_df[["GHI (W/m²)"]], color="#F5A623", height=180)

                            st.markdown("**Energy Output (kWh)**")
                            st.bar_chart(chart_df[["Energy (kWh)"]], color="#3FB950", height=180)

                        with st.expander("📋 View Raw Forecast Data"):
                            st.json(data)

                    else:
                        err = resp.json().get("detail", resp.text)
                        st.error(f"API Error {resp.status_code}: {err}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Run: `uvicorn main:app --reload`")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#8B949E;font-size:0.8rem;">
        XGBoost Meta-Model · VMD Decomposition · Hybrid Feature Selection · Window (12,3)
        <br>Energy = GHI × Area × Efficiency / 1000
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Energy Consumption 24h
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="main-title-blue">⚡ Energy Consumption Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">24-Hour Ahead Consumption Forecast  ·  Recursive Hybrid (XGB + LSTM)  ·  Manual Entry or Upload CSV</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Energy Forecast Settings")
        st.markdown('<div class="info-box">Configure input for the 24h consumption model. Use manual entry or upload a CSV.</div>', unsafe_allow_html=True)
        energy_max_rows = st.number_input(
            "Max rows from CSV to use",
            min_value=1, max_value=10000, value=500, step=100,
            help="Only the last N rows are sent to the API (last row = forecast start).",
            key="energy_max_rows"
        )
        st.markdown("---")
        st.markdown("### 🔗 Energy API Status")
        try:
            re = requests.get(f"{API_URL}/energy-status", timeout=3)
            if re.status_code == 200:
                edata = re.json()
                if edata.get("hybrid_ready"):
                    st.success("Hybrid (XGB + LSTM) ready ✓")
                else:
                    st.warning(edata.get("message", "LSTM not loaded — add lstm_consumption.pt"))
            else:
                r2 = requests.get(f"{API_URL}/health", timeout=3)
                if r2.status_code == 200:
                    st.warning("Backend up but energy-status failed")
                else:
                    st.error("Backend offline — start FastAPI server")
        except Exception:
            st.error("Backend offline — start FastAPI server")
        st.markdown("---")
        st.caption("24h consumption · Hybrid (XGB + LSTM) required")

    # ── Input Mode ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header section-header-blue">📥 Input Historical Consumption Data</div>', unsafe_allow_html=True)

    energy_input_mode = st.radio(
        "Input method",
        ["Manual Entry (last row + auto-fill context)", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
        key="energy_input_mode"
    )

    st.markdown('<div class="info-box">⚠️ The model needs at least <b>24 hourly rows</b> of context. Upload a CSV with full history, or use manual entry which auto-generates realistic context rows around your input.</div>', unsafe_allow_html=True)

    def generate_energy_context(row: dict, n: int = 35) -> list[dict]:
        """Generate n synthetic historical rows ending at the given row for energy consumption."""
        base_dt = pd.to_datetime(row["datetime"]) - pd.Timedelta(hours=n)
        rows = []
        for i in range(n):
            dt = base_dt + pd.Timedelta(hours=i)
            # Simple context generation: vary around the input value
            power = max(0.0, row["Global_active_power"] + np.random.normal(0, 0.5))
            rows.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                "Global_active_power": round(power, 2),
            })
        rows.append(row)
        return rows

    energy_data = None

    if energy_input_mode == "Manual Entry (last row + auto-fill context)":
        st.markdown('<div class="section-header section-header-blue">🔌 Current Consumption Reading</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            energy_dt_input = st.text_input("Datetime",
                                             value=st.session_state.get("shared_dt", datetime.now().strftime("%Y-%m-%d %H:00")),
                                             help="Format: YYYY-MM-DD HH:MM — synced from Common Settings", key="energy_dt_input")
        with col2:
            energy_power = st.number_input("Global Active Power (kW)",
                                           min_value=0.0, max_value=10.0, value=1.5, step=0.1,
                                           help="Current total active power consumed by the household (kW)",
                                           key="energy_power")

        current_energy_row = {
            "datetime": energy_dt_input,
            "Global_active_power": energy_power,
        }
        energy_data = generate_energy_context(current_energy_row, n=35)

    else: # Upload CSV
        energy_uploaded = st.file_uploader("CSV for energy consumption", type=["csv"], key="energy_csv")
        if energy_uploaded:
            try:
                df_energy = pd.read_csv(energy_uploaded)
                # Normalize column names to handle variations
                df_energy.columns = [c.strip().lower().replace(" ", "_") for c in df_energy.columns]

                dt_col = None
                if "datetime" in df_energy.columns:
                    dt_col = "datetime"
                elif "dt" in df_energy.columns:
                    dt_col = "dt"

                power_col = None
                if "global_active_power" in df_energy.columns:
                    power_col = "global_active_power"

                if dt_col is None:
                    st.error("CSV must include a 'datetime' or 'dt' column.")
                elif power_col is None:
                    st.error("CSV must include 'Global_active_power' (or 'global_active_power').")
                elif len(df_energy) < 24:
                    st.error(f"Need at least 24 rows of hourly data, file has {len(df_energy)}.")
                else:
                    # Ensure only required columns are passed and renamed if necessary
                    df_energy_filtered = df_energy[[dt_col, power_col]].copy()
                    df_energy_filtered.rename(columns={dt_col: "datetime", power_col: "Global_active_power"}, inplace=True)

                    n_use = min(int(energy_max_rows), len(df_energy_filtered))
                    energy_data = df_energy_filtered.tail(n_use).to_dict(orient="records")
                    st.success(f"✓ Using last **{n_use}** rows (of {len(df_energy_filtered)}).")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    energy_clicked = st.button("⚡ Generate 24h Consumption Forecast", key="energy_btn")
    if energy_clicked:
        if energy_data is None or len(energy_data) == 0:
            st.warning("Please provide input data first (manual entry or upload CSV with at least 24 rows).")
        else:
            with st.spinner("Running energy consumption model..."):
                try:
                    payload = {"historical_data": energy_data}
                    r = requests.post(f"{API_URL}/predict-energy", json=payload, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        forecasts = data["forecasts"]
                        total = data["total_consumption_kwh"]

                        # ── Store results for Combined Dashboard ──────────────
                        st.session_state["ss_energy"] = {
                            "total_kwh": total,
                            "forecasts":  forecasts,
                        }
                        st.markdown('<div class="section-header section-header-blue">📊 24h Consumption Forecast</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="total-energy-box" style="border-color:rgba(88,166,255,0.4);background:linear-gradient(135deg,rgba(88,166,255,0.12),rgba(88,166,255,0.05));">
                            <div class="metric-label">Total Predicted Consumption (24h)</div>
                            <div class="total-value" style="color:#58A6FF;">{total:.4f} kWh</div>
                        </div>""", unsafe_allow_html=True)
                        chart_df = pd.DataFrame({
                            "Hour": [f["hour_label"] for f in forecasts],
                            "Consumption (kWh)": [f["consumption_kwh"] for f in forecasts],
                        })
                        st.bar_chart(chart_df.set_index("Hour"), color="#58A6FF", height=300)
                        with st.expander("View hourly table"):
                            st.dataframe(pd.DataFrame(forecasts), use_container_width=True)
                        with st.expander("Raw response"):
                            st.json(data)
                    else:
                        st.error(f"API error {r.status_code}: " + (r.json().get("detail", r.text) if r.text else ""))
                except requests.exceptions.ConnectionError:
                    st.error("Backend offline. Start the FastAPI server.")
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tariff Forecast (12h, BiLSTM, client-side)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="main-title-tariff">💰 Tariff Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">12-Hour Ahead Electricity Tariff Prediction  ·  BiLSTM (2-layer, 128 hidden)  ·  Indian Grid Tariff (₹/kWh)</div>', unsafe_allow_html=True)

    # ── Input Mode Toggle ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header section-header-tariff">📥 Input Tariff Conditions</div>', unsafe_allow_html=True)

    tariff_input_mode = st.radio(
        "Tariff input method",
        ["Manual Entry", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
        key="tariff_input_mode"
    )

    st.markdown('<div class="info-box-tariff">Configure conditions for the BiLSTM tariff model. Use <b>Manual Entry</b> to set values directly, or <b>Upload CSV</b> with columns: <code>hour</code>, <code>day_of_week</code>, <code>month</code>, <code>temperature</code>, <code>historical_avg_tariff</code>, <code>demand_index</code>. The <b>last row</b> of the CSV is used for forecasting.</div>', unsafe_allow_html=True)

    now = datetime.now()
    _shared_hour = int(pd.to_datetime(st.session_state.get("shared_dt", now.strftime("%Y-%m-%d %H:00")), errors="coerce").hour)
    _shared_temp_int = int(round(float(st.session_state.get("shared_temp", 30))))

    # Default values (used by both modes, overridden by CSV if uploaded)
    t_hour    = _shared_hour
    t_dow     = now.weekday()
    t_month   = now.month
    t_temp    = _shared_temp_int
    t_hist_avg = 6.5
    t_demand  = 0.5
    tariff_params_ready = False

    if tariff_input_mode == "Manual Entry":
        st.markdown('<div class="section-header section-header-tariff">🌡️ Current Tariff Conditions</div>', unsafe_allow_html=True)

        ti_c1, ti_c2, ti_c3 = st.columns(3)
        with ti_c1:
            t_hour = st.slider(
                "Current Hour of Day", min_value=0, max_value=23,
                value=_shared_hour, step=1,
                help="0 = midnight, 12 = noon, 23 = 11 PM — synced from Common Settings",
                key="t_hour"
            )
            t_dow = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
                index=now.weekday(),
                key="t_dow"
            )
        with ti_c2:
            t_month = st.slider(
                "Month", min_value=1, max_value=12,
                value=now.month, step=1,
                format="%d",
                key="t_month"
            )
            t_temp = st.slider(
                "Temperature (°C)", min_value=-10, max_value=50,
                value=_shared_temp_int, step=1,
                help="Ambient temperature — synced from Common Settings",
                key="t_temp"
            )
        with ti_c3:
            t_hist_avg = st.number_input(
                "Historical Avg Tariff (₹/kWh)",
                min_value=1.0, max_value=20.0,
                value=6.5, step=0.1,
                help="Your region's average tariff rate",
                key="t_hist_avg"
            )
            t_demand = st.slider(
                "Demand Index (0–1)", min_value=0.0, max_value=1.0,
                value=0.5, step=0.05,
                help="0 = very low grid demand, 1 = peak demand",
                key="t_demand"
            )
        tariff_params_ready = True

    else:
        # ── CSV Upload mode ───────────────────────────────────────────────────
        tariff_csv = st.file_uploader(
            "Upload CSV (columns: hour, day_of_week, month, temperature, historical_avg_tariff, demand_index)",
            type=["csv"],
            key="tariff_csv"
        )
        if tariff_csv:
            try:
                df_t = pd.read_csv(tariff_csv)
                required_cols = ["hour", "day_of_week", "month", "temperature", "historical_avg_tariff", "demand_index"]
                # case-insensitive column check
                df_t.columns = [c.strip().lower().replace(" ", "_") for c in df_t.columns]
                missing_cols = [c for c in required_cols if c not in df_t.columns]
                if missing_cols:
                    st.error(f"CSV is missing columns: {missing_cols}")
                elif len(df_t) == 0:
                    st.error("CSV has no rows.")
                else:
                    last = df_t.iloc[-1]
                    t_hour     = int(np.clip(float(last["hour"]), 0, 23))
                    t_dow      = int(np.clip(float(last["day_of_week"]), 0, 6))
                    t_month    = int(np.clip(float(last["month"]), 1, 12))
                    t_temp     = float(last["temperature"])
                    t_hist_avg = float(last["historical_avg_tariff"])
                    t_demand   = float(np.clip(float(last["demand_index"]), 0.0, 1.0))
                    tariff_params_ready = True
                    st.success(f"✓ Using last row (row {len(df_t)}) — Hour: {t_hour}, Day: {t_dow}, Month: {t_month}, Temp: {t_temp}°C, Avg tariff: ₹{t_hist_avg}, Demand: {t_demand:.2f}")
                    if len(df_t) > 1:
                        with st.expander("📋 Preview uploaded data"):
                            st.dataframe(df_t.tail(10), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    # ── AM/PM helper ──────────────────────────────────────────────────────────
    def _ampm(h: int) -> str:
        """Convert 0–23 hour int to 12h AM/PM string."""
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:00 {suffix}"

    if tariff_params_ready:
        # ── Parameter summary cards ───────────────────────────────────────────
        st.markdown('<div class="section-header section-header-tariff">📋 Forecast Parameters</div>', unsafe_allow_html=True)

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        with p1:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Hour</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.3rem;">{_ampm(t_hour)}</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Day</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.5rem;">{day_names[t_dow]}</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Month</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.5rem;">{month_names[t_month-1]}</div>
            </div>""", unsafe_allow_html=True)
        with p4:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Temp</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.5rem;">{t_temp}°C</div>
            </div>""", unsafe_allow_html=True)
        with p5:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Hist. Avg</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.5rem;">₹{t_hist_avg:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with p6:
            st.markdown(f"""<div class="metric-card metric-card-tariff" style="padding:0.9rem 0.7rem;">
                <div class="metric-label">Demand</div>
                <div class="metric-value metric-value-tariff" style="font-size:1.5rem;">{t_demand:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        tariff_clicked = st.button("💰 Generate 12h Tariff Forecast", key="tariff_btn")
    else:
        tariff_clicked = False

    if tariff_clicked:
        with st.spinner("Running BiLSTM tariff model..."):
            try:
                tariffs, stats = predict_tariff(
                    hour_of_day    = float(t_hour),
                    day_of_week    = float(t_dow),
                    month          = float(t_month),
                    temperature    = float(t_temp),
                    historical_avg = float(t_hist_avg),
                    demand_index   = float(t_demand),
                )

                # Determine thresholds for badge colouring
                peak_threshold    = stats["avg"] + 0.4 * (stats["max"] - stats["avg"])
                off_peak_threshold = stats["avg"] - 0.4 * (stats["avg"] - stats["min"])

                # ── Store results for Combined Dashboard ──────────────
                st.session_state["ss_tariff"] = {
                    "tariffs":    tariffs,
                    "stats":      stats,
                    "start_hour": t_hour,
                }

                # ── Summary Metrics ───────────────────────────────────────────
                st.markdown('<div class="section-header section-header-tariff">📊 12-Hour Tariff Summary</div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card metric-card-tariff">
                        <div class="metric-label">Min Tariff</div>
                        <div class="metric-value metric-value-tariff">₹{stats['min']:.2f}</div>
                        <div class="metric-unit">per kWh</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card metric-card-tariff">
                        <div class="metric-label">Max Tariff</div>
                        <div class="metric-value" style="color:#FF6B6B;">₹{stats['max']:.2f}</div>
                        <div class="metric-unit">per kWh</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card metric-card-tariff">
                        <div class="metric-label">Avg Tariff</div>
                        <div class="metric-value metric-value-tariff">₹{stats['avg']:.2f}</div>
                        <div class="metric-unit">per kWh</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    peak_h_abs = (t_hour + stats['peak_hour'] + 1) % 24
                    off_h_abs  = (t_hour + stats['off_peak_hour'] + 1) % 24
                    st.markdown(f"""
                    <div class="metric-card metric-card-tariff">
                        <div class="metric-label">Peak Hour</div>
                        <div class="metric-value" style="color:#FF6B6B;">{_ampm(peak_h_abs)}</div>
                        <div class="metric-unit">Off-peak: {_ampm(off_h_abs)}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Highlights box ─────────────────────────────────────────────
                st.markdown(f"""
                <div class="total-tariff-box">
                    <div class="metric-label">📈 Average Forecasted Tariff (next 12 hours)</div>
                    <div class="total-value-tariff">₹{stats['avg']:.2f} / kWh</div>
                    <div class="metric-unit" style="margin-top:0.3rem;">
                        Range: ₹{stats['min']:.2f} – ₹{stats['max']:.2f}  ·  Spread: ₹{stats['max']-stats['min']:.2f}
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Chart + Hour cards ─────────────────────────────────────────
                st.markdown('<div class="section-header section-header-tariff">📈 12-Hour Tariff Trend & Breakdown</div>', unsafe_allow_html=True)

                chart_col, cards_col = st.columns([2, 3])

                with chart_col:
                    hours_abs = [(t_hour + i + 1) % 24 for i in range(12)]
                    hour_labels = [_ampm(h) for h in hours_abs]

                    chart_df = pd.DataFrame({
                        "Time"        : hour_labels,
                        "Tariff (₹/kWh)": tariffs,
                    }).set_index("Time")

                    st.markdown("**Tariff Forecast Line**")
                    st.line_chart(chart_df, color="#BC8CFF", height=280)

                    # Mini stats bar
                    st.markdown(f"""
                    <div style="display:flex;gap:0.5rem;margin-top:0.5rem;flex-wrap:wrap;">
                        <div style="background:rgba(255,107,107,0.15);border:1px solid rgba(255,107,107,0.4);border-radius:8px;padding:0.4rem 0.8rem;font-size:0.8rem;">
                            🔴 Peak: {_ampm(peak_h_abs)} · ₹{stats['max']:.2f}
                        </div>
                        <div style="background:rgba(63,185,80,0.15);border:1px solid rgba(63,185,80,0.4);border-radius:8px;padding:0.4rem 0.8rem;font-size:0.8rem;">
                            🟢 Off-peak: {_ampm(off_h_abs)} · ₹{stats['min']:.2f}
                        </div>
                    </div>""", unsafe_allow_html=True)

                with cards_col:
                    st.markdown("**Hourly Breakdown**")
                    for i, (t_val, h_abs, h_lbl) in enumerate(zip(tariffs, hours_abs, hour_labels)):
                        if t_val >= peak_threshold:
                            badge_class = "tariff-badge-peak"
                            badge_icon  = "🔴 Peak"
                            tariff_color = "#FF6B6B"
                        elif t_val <= off_peak_threshold:
                            badge_class = "tariff-badge-off"
                            badge_icon  = "🟢 Off-peak"
                            tariff_color = "#3FB950"
                        else:
                            badge_class = "tariff-badge-mid"
                            badge_icon  = "🟡 Mid"
                            tariff_color = "#F5A623"

                        bar_pct = int(((t_val - stats['min']) / max(stats['max'] - stats['min'], 0.01)) * 100)

                        st.markdown(f"""
                        <div class="forecast-card" style="padding:0.85rem 1.1rem;">
                            <div class="{badge_class}">{badge_icon}</div>
                            <div style="flex:1">
                                <div style="font-family:'Syne',sans-serif;font-weight:600;font-size:0.9rem;">
                                    t+{i+1}h &nbsp; <span style="color:#8B949E;font-weight:400;">{h_lbl}</span>
                                </div>
                                <div style="margin-top:5px;height:5px;background:#30363D;border-radius:3px;overflow:hidden;">
                                    <div style="width:{bar_pct}%;height:100%;background:{tariff_color};border-radius:3px;transition:width 0.5s;"></div>
                                </div>
                            </div>
                            <div style="text-align:right;min-width:70px;">
                                <div style="font-family:'Syne',sans-serif;font-weight:800;
                                            color:{tariff_color};font-size:1.1rem;">
                                    ₹{t_val:.2f}
                                </div>
                                <div style="color:#8B949E;font-size:0.78rem;">/kWh</div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                # ── Recommendations ────────────────────────────────────────────
                st.markdown('<div class="section-header section-header-tariff">💡 Smart Recommendations</div>', unsafe_allow_html=True)

                best_hours = sorted(enumerate(tariffs), key=lambda x: x[1])[:3]
                worst_hours = sorted(enumerate(tariffs), key=lambda x: x[1], reverse=True)[:3]

                best_labels  = ", ".join(f"{(t_hour + i + 1)%24:02d}:00 (₹{v:.2f})" for i, v in best_hours)
                worst_labels = ", ".join(f"{(t_hour + i + 1)%24:02d}:00 (₹{v:.2f})" for i, v in worst_hours)

                savings = (stats['max'] - stats['min']) * 1.0  # for 1 kWh
                st.markdown(f"""
                <div class="recommendation-box">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;color:#3FB950;font-size:1rem;margin-bottom:0.5rem;">
                        ✅ Best Times to Use High-Power Appliances
                    </div>
                    <div style="font-size:0.9rem;color:#E6EDF3;">{best_labels}</div>
                    <div style="font-size:0.82rem;color:#8B949E;margin-top:0.4rem;">
                        Shift heavy loads (washing machines, EV charging, dishwashers) to these hours to minimize your electricity bill.
                    </div>
                </div>
                <div class="warning-box" style="margin-top:0.5rem;">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;color:#FF6B6B;font-size:1rem;margin-bottom:0.5rem;">
                        ⚠️ Avoid High Consumption at Peak Hours
                    </div>
                    <div style="font-size:0.9rem;color:#E6EDF3;">{worst_labels}</div>
                    <div style="font-size:0.82rem;color:#8B949E;margin-top:0.4rem;">
                        You could save up to <b style="color:#F5A623;">₹{savings:.2f} per kWh</b> by shifting usage from peak to off-peak hours.
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Raw data expander ──────────────────────────────────────────
                with st.expander("📋 View Raw Tariff Data"):
                    raw_df = pd.DataFrame({
                        "Step"        : [f"t+{i+1}h" for i in range(12)],
                        "Time"        : hour_labels,
                        "Tariff (₹/kWh)": [round(t, 4) for t in tariffs],
                    })
                    st.dataframe(raw_df, use_container_width=True)
                    st.json(stats)

            except Exception as e:
                st.error(f"Tariff model error: {e}")
                st.exception(e)

    else:
        # Placeholder when button not yet clicked
        st.markdown("""
        <div class="info-box-tariff" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">💰</div>
            <div style="font-family:'Syne',sans-serif;font-weight:600;font-size:1.1rem;color:#E6EDF3;margin-bottom:0.4rem;">
                Configure & Generate Tariff Forecast
            </div>
            <div style="font-size:0.88rem;color:#8B949E;">
                Adjust the parameters in the sidebar and click <b>Generate 12h Tariff Forecast</b> to see predictions.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#8B949E;font-size:0.8rem;">
        BiLSTM · 2-layer Bidirectional · 128 hidden units · 12-hour horizon · Input: 14 temporal + weather features
    </div>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Combined Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="main-title" style="background:linear-gradient(135deg,#F5A623 0%,#58A6FF 50%,#BC8CFF 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">📊 Combined Energy Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unified view · Solar Generation · Consumption · Tariff Intelligence</div>', unsafe_allow_html=True)

    # ── Pull from session state ────────────────────────────────────────────────
    _sol  = st.session_state.get("ss_solar")
    _con  = st.session_state.get("ss_energy")
    _tar  = st.session_state.get("ss_tariff")

    has_solar    = _sol  is not None
    has_energy   = _con  is not None
    has_tariff   = _tar  is not None
    models_ready = sum([has_solar, has_energy, has_tariff])

    # ── Status row ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Model Status</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    def _status_card(ready, label, icon, color):
        bg   = f"rgba({color},0.12)" if ready else "rgba(139,148,158,0.08)"
        bord = f"rgba({color},0.35)" if ready else "rgba(139,148,158,0.25)"
        mark = "✅ Ready" if ready else "⏳ Not run"
        return f"""<div class="metric-card" style="background:{bg};border-color:{bord};padding:1rem;">
            <div style="font-size:1.8rem">{icon}</div>
            <div class="metric-label" style="margin-top:0.4rem">{label}</div>
            <div style="font-size:0.9rem;font-weight:600;color:{'#3FB950' if ready else '#8B949E'};margin-top:0.3rem">{mark}</div>
        </div>"""

    with sc1:
        st.markdown(_status_card(has_solar,  "Solar Forecast",      "☀️", "245,166,35"),  unsafe_allow_html=True)
    with sc2:
        st.markdown(_status_card(has_energy, "Energy Consumption",  "⚡", "88,166,255"),  unsafe_allow_html=True)
    with sc3:
        st.markdown(_status_card(has_tariff, "Tariff Forecast",     "💰", "188,140,255"), unsafe_allow_html=True)

    if models_ready == 0:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:2.5rem;margin-top:1.5rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem">📊</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.2rem;color:#E6EDF3;margin-bottom:0.5rem">
                Run at least one forecast to see results here
            </div>
            <div style="font-size:0.9rem;color:#8B949E">
                Go to the ☀️ Solar, ⚡ Energy, or 💰 Tariff tab and click the generate button.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1 — Next 1-Hour Energy Decision
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;
                    background:linear-gradient(90deg,#F5A623,#58A6FF);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin:0.5rem 0 1rem 0;">
            ⚡ Next 1-Hour Energy Decision
        </div>""", unsafe_allow_html=True)

        # Extract t+1h values
        solar_1h      = _sol["forecasts"][0]["energy_kwh"]      if has_solar  else None
        solar_1h_lbl  = _sol["forecasts"][0]["hour_label"]      if has_solar  else "—"
        energy_1h     = _con["forecasts"][0]["consumption_kwh"] if has_energy else None
        energy_1h_lbl = _con["forecasts"][0]["hour_label"]      if has_energy else "—"

        # Identify best SELL hour from tariff model (peak tariff = highest price → best time to sell)
        sell_hour_label = None
        sell_tariff_val = None
        sell_hour_abs   = None
        if has_tariff:
            _t_list = _tar["tariffs"]
            _sh     = _tar["start_hour"]
            peak_idx        = int(np.argmax(_t_list))
            sell_hour_abs   = (_sh + peak_idx + 1) % 24
            sell_tariff_val = _t_list[peak_idx]
            suffix = "AM" if sell_hour_abs < 12 else "PM"
            h12    = sell_hour_abs % 12 or 12
            sell_hour_label = f"{h12}:00 {suffix}"

        # ── 3 KPI cards ───────────────────────────────────────────────────────
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            if has_solar:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(245,166,35,0.5);text-align:center;">
                    <div class="metric-label">☀️ Solar Generated (t+1h)</div>
                    <div style="font-size:0.78rem;color:#8B949E;margin-bottom:0.3rem;">{solar_1h_lbl}</div>
                    <div class="metric-value" style="color:#F5A623;font-size:2.4rem;">{solar_1h:.4f}</div>
                    <div class="metric-unit">kWh</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card" style="opacity:0.45;text-align:center;">
                    <div style="font-size:1.8rem">☀️</div>
                    <div class="metric-label" style="margin-top:0.4rem">Solar — not run</div>
                </div>""", unsafe_allow_html=True)

        with kpi2:
            if has_energy:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(88,166,255,0.5);text-align:center;">
                    <div class="metric-label">⚡ Energy Needed (t+1h)</div>
                    <div style="font-size:0.78rem;color:#8B949E;margin-bottom:0.3rem;">{energy_1h_lbl}</div>
                    <div class="metric-value" style="color:#58A6FF;font-size:2.4rem;">{energy_1h:.4f}</div>
                    <div class="metric-unit">kWh</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card" style="opacity:0.45;text-align:center;">
                    <div style="font-size:1.8rem">⚡</div>
                    <div class="metric-label" style="margin-top:0.4rem">Consumption — not run</div>
                </div>""", unsafe_allow_html=True)

        with kpi3:
            if has_tariff:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(255,107,107,0.5);text-align:center;">
                    <div class="metric-label">🔴 Best Sell Hour (tariff peak)</div>
                    <div style="font-size:0.78rem;color:#8B949E;margin-bottom:0.3rem;">Identified by BiLSTM tariff model</div>
                    <div class="metric-value" style="color:#FF6B6B;font-size:2.4rem;">{sell_hour_label}</div>
                    <div class="metric-unit">₹{sell_tariff_val:.2f} / kWh (peak)</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card" style="opacity:0.45;text-align:center;">
                    <div style="font-size:1.8rem">💰</div>
                    <div class="metric-label" style="margin-top:0.4rem">Tariff — not run</div>
                </div>""", unsafe_allow_html=True)

        # ── 1-Hour Decision Banner ────────────────────────────────────────────
        if has_solar and has_energy:
            surplus_1h = solar_1h - energy_1h
            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

            if surplus_1h > 0:
                if has_tariff and sell_hour_label:
                    sell_earn   = surplus_1h * sell_tariff_val
                    sell_advice = f'<div style="margin-top:0.7rem;font-size:0.9rem;color:#E6EDF3;">💡 <b>Sell Action:</b> Export <b>{surplus_1h:.4f} kWh</b> to the grid at <b style="color:#FF6B6B;">{sell_hour_label}</b> when the tariff peaks at <b style="color:#FF6B6B;">₹{sell_tariff_val:.2f}/kWh</b>.<br><span style="color:#3FB950;font-weight:600;">Estimated revenue: ₹{sell_earn:.2f}</span> — schedule your grid export / battery discharge at that hour for maximum return.</div>'
                else:
                    sell_advice = f'<div style="margin-top:0.7rem;font-size:0.9rem;color:#E6EDF3;">💡 Run the <b>💰 Tariff Forecast</b> tab to find the best hour to sell the <b>{surplus_1h:.4f} kWh</b> surplus.</div>'

                st.markdown(f"""
                <div class="recommendation-box" style="border-color:#3FB950;background:linear-gradient(135deg,rgba(63,185,80,0.15),rgba(63,185,80,0.05));">
                    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;">
                        <div style="font-size:2rem;">🟢</div>
                        <div>
                            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#3FB950;">
                                SURPLUS — Solar exceeds demand by {surplus_1h:.4f} kWh
                            </div>
                            <div style="font-size:0.85rem;color:#8B949E;margin-top:2px;">
                                ☀️ Solar next hour: <b style="color:#F5A623">{solar_1h:.4f} kWh</b> &nbsp;|&nbsp;
                                ⚡ Energy needed: <b style="color:#58A6FF">{energy_1h:.4f} kWh</b>
                            </div>
                        </div>
                    </div>
                    {sell_advice}
                </div>""", unsafe_allow_html=True)

            else:
                deficit_1h   = abs(surplus_1h)
                grid_cost_txt = ""
                if has_tariff:
                    next_tariff   = _tar["tariffs"][0]
                    grid_cost     = deficit_1h * next_tariff
                    grid_cost_txt = f" | Grid import cost at current tariff: <b style='color:#FF6B6B;'>₹{grid_cost:.2f}</b>"

                cheap_hint = ""
                if has_tariff:
                    cheap_hr  = (_tar["start_hour"] + int(np.argmin(_tar["tariffs"])) + 1) % 24
                    cheap_suf = "AM" if cheap_hr < 12 else "PM"
                    cheap_h12 = cheap_hr % 12 or 12
                    cheap_hint = f"Cheapest grid import window is at <b>{cheap_h12}:00 {cheap_suf}</b> (₹{min(_tar['tariffs']):.2f}/kWh)."
                else:
                    cheap_hint = "Check the tariff tab for the cheapest grid import window."

                st.markdown(f"""
                <div class="warning-box" style="border-color:#FF6B6B;background:linear-gradient(135deg,rgba(255,107,107,0.15),rgba(255,107,107,0.05));">
                    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;">
                        <div style="font-size:2rem;">🔴</div>
                        <div>
                            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#FF6B6B;">
                                DEFICIT — Grid import needed: {deficit_1h:.4f} kWh
                            </div>
                            <div style="font-size:0.85rem;color:#8B949E;margin-top:2px;">
                                ☀️ Solar next hour: <b style="color:#F5A623">{solar_1h:.4f} kWh</b> &nbsp;|&nbsp;
                                ⚡ Energy needed: <b style="color:#58A6FF">{energy_1h:.4f} kWh</b>
                                {grid_cost_txt}
                            </div>
                        </div>
                    </div>
                    <div style="margin-top:0.6rem;font-size:0.9rem;color:#E6EDF3;">
                        💡 <b>Tip:</b> Defer non-critical loads or use stored battery energy. {cheap_hint}
                    </div>
                </div>""", unsafe_allow_html=True)

        elif has_solar and not has_energy:
            st.info("💡 Run the **⚡ Energy Consumption** tab to compare solar generation vs demand.")
        elif has_energy and not has_solar:
            st.info("💡 Run the **☀️ Solar Forecast** tab to compare solar generation vs demand.")

        # ── 1-Hour Mini Bar Chart ─────────────────────────────────────────────
        if has_solar and has_energy:
            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
            bar_df_1h = pd.DataFrame({
                "Source": ["☀️ Solar (t+1h)", "⚡ Demand (t+1h)"],
                "kWh":    [solar_1h, energy_1h],
            }).set_index("Source")
            ch1, ch2 = st.columns([2, 1])
            with ch1:
                st.markdown("**Next 1-Hour: Solar vs Consumption**")
                st.bar_chart(bar_df_1h, height=220)
            with ch2:
                if has_tariff:
                    cur_tariff = _tar["tariffs"][0]
                    st.markdown(f"""
                    <div class="metric-card metric-card-tariff" style="margin-top:1.8rem;text-align:center;">
                        <div class="metric-label">Tariff — Next Hour</div>
                        <div class="metric-value metric-value-tariff" style="font-size:1.8rem;">₹{cur_tariff:.2f}</div>
                        <div class="metric-unit">/kWh</div>
                    </div>
                    <div class="metric-card" style="margin-top:0.6rem;border-color:rgba(255,107,107,0.5);text-align:center;">
                        <div class="metric-label">Best Sell @</div>
                        <div class="metric-value" style="color:#FF6B6B;font-size:1.8rem;">{sell_hour_label}</div>
                        <div class="metric-unit">₹{sell_tariff_val:.2f}/kWh</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2 — 3-Hour Energy Balance
        # ══════════════════════════════════════════════════════════════════════
        if has_solar or has_energy:
            st.markdown('<div class="section-header">⚖️ 3-Hour Energy Balance</div>', unsafe_allow_html=True)

            total_gen   = _sol["total_kwh"] if has_solar else 0.0
            first3h_con = sum(f["consumption_kwh"] for f in _con["forecasts"][:3]) if has_energy else 0.0

            eb1, eb2, eb3 = st.columns(3)
            with eb1:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(245,166,35,0.4);">
                    <div class="metric-label">☀️ Solar Generated</div>
                    <div class="metric-value" style="color:#F5A623">{total_gen:.3f}</div>
                    <div class="metric-unit">kWh (3 h horizon)</div>
                </div>""", unsafe_allow_html=True)
            with eb2:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(88,166,255,0.4);">
                    <div class="metric-label">⚡ Consumption</div>
                    <div class="metric-value" style="color:#58A6FF">{first3h_con:.3f}</div>
                    <div class="metric-unit">kWh (3 h horizon)</div>
                </div>""", unsafe_allow_html=True)
            with eb3:
                net         = total_gen - first3h_con
                net_color   = "#3FB950" if net >= 0 else "#FF6B6B"
                net_label   = "Surplus 🟢" if net >= 0 else "Deficit 🔴"
                st.markdown(f"""
                <div class="metric-card" style="border-color:{net_color}55;">
                    <div class="metric-label">Net Balance (3h)</div>
                    <div class="metric-value" style="color:{net_color}">{net:+.3f}</div>
                    <div class="metric-unit">kWh · {net_label}</div>
                </div>""", unsafe_allow_html=True)

            if has_solar and has_energy:
                first3_labels   = [f["hour_label"] for f in _con["forecasts"][:3]]
                first3_con_vals = [f["consumption_kwh"] for f in _con["forecasts"][:3]]
                solar_vals      = [f["energy_kwh"] for f in _sol["forecasts"]]
                n = max(len(solar_vals), len(first3_con_vals))
                solar_vals      = solar_vals      + [0.0] * (n - len(solar_vals))
                first3_con_vals = first3_con_vals + [0.0] * (n - len(first3_con_vals))
                labels = first3_labels + [f["hour_label"] for f in _sol["forecasts"]][len(first3_labels):]
                labels = labels[:n]
                bal_df = pd.DataFrame({
                    "Hour":               labels[:n],
                    "Solar Gen (kWh)":    solar_vals[:n],
                    "Consumption (kWh)":  first3_con_vals[:n],
                }).set_index("Hour")
                st.markdown("**3-Hour Solar Generation vs Consumption**")
                st.bar_chart(bal_df, height=240)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4 — Unified Timeline Chart
        # ══════════════════════════════════════════════════════════════════════
        if models_ready >= 2:
            st.markdown('<div class="section-header">📈 Unified Forecast Timeline</div>', unsafe_allow_html=True)

            timeline_data = {}

            if has_solar:
                for f in _sol["forecasts"]:
                    lbl = f["hour_label"]
                    timeline_data.setdefault(lbl, {})["Solar Gen (kWh)"] = f["energy_kwh"]

            if has_energy:
                for f in _con["forecasts"]:
                    lbl = f["hour_label"]
                    timeline_data.setdefault(lbl, {})["Consumption (kWh)"] = f["consumption_kwh"]

            if has_tariff:
                _sh = _tar["start_hour"]
                for i, tv in enumerate(_tar["tariffs"]):
                    lbl_h = (_sh + i + 1) % 24
                    lbl = pd.Timestamp.now().normalize() + pd.Timedelta(hours=lbl_h)
                    lbl_str = lbl.strftime("%H:%M")
                    matched = [k for k in timeline_data if k.endswith(lbl_str)]
                    key = matched[0] if matched else f"H{lbl_h:02d}:00"
                    timeline_data.setdefault(key, {})["Tariff (₹/kWh)"] = tv

                if sell_hour_label:
                    st.markdown(f"""
                    <div style="background:rgba(255,107,107,0.1);border:1px solid rgba(255,107,107,0.3);
                                border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.5rem;font-size:0.85rem;">
                        📌 <b>Best sell window:</b>
                        <span style="color:#FF6B6B;font-weight:700;">{sell_hour_label}</span>
                        — tariff peaks at <b>₹{sell_tariff_val:.2f}/kWh</b>
                    </div>""", unsafe_allow_html=True)

            if timeline_data:
                tl_df = pd.DataFrame(timeline_data).T.sort_index().fillna(0)
                st.line_chart(tl_df, height=280)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5 — Combined Smart Recommendations
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">💡 Combined Smart Recommendations</div>', unsafe_allow_html=True)

        recs = []

        # Primary: 1-hour solar vs demand decision + sell recommendation
        if has_solar and has_energy:
            surplus_1h = solar_1h - energy_1h
            if surplus_1h > 0:
                if has_tariff and sell_hour_label:
                    sell_earn = surplus_1h * sell_tariff_val
                    recs.append(("💹 Sell Extra Solar Energy — Primary Decision",
                                  f"Next hour ({solar_1h_lbl}): solar generates <b style='color:#F5A623'>{solar_1h:.4f} kWh</b> "
                                  f"but you only need <b style='color:#58A6FF'>{energy_1h:.4f} kWh</b>. "
                                  f"<b style='color:#3FB950'>Surplus: {surplus_1h:.4f} kWh</b>.<br>"
                                  f"✅ Tariff model says price peaks at "
                                  f"<b style='color:#FF6B6B'>{sell_hour_label} (₹{sell_tariff_val:.2f}/kWh)</b>. "
                                  f"Export at that hour → estimated revenue <b style='color:#3FB950'>₹{sell_earn:.2f}</b>.",
                                  "#3FB950", "recommendation-box"))
                else:
                    recs.append(("🔋 Export Opportunity",
                                  f"Next hour solar surplus: <b>{surplus_1h:.4f} kWh</b>. Run the Tariff tab to find the best sell hour.",
                                  "#3FB950", "recommendation-box"))
            else:
                deficit = abs(surplus_1h)
                recs.append(("⚠️ Grid Import Required",
                              f"Next hour deficit: <b>{deficit:.4f} kWh</b>. "
                              "Solar cannot cover demand — defer non-critical loads.",
                              "#FF6B6B", "warning-box"))

        if has_solar:
            best_gen_idx  = int(np.argmax([f["energy_kwh"] for f in _sol["forecasts"]]))
            best_hour_lbl = _sol["forecasts"][best_gen_idx]["hour_label"]
            best_gen_kwh  = _sol["forecasts"][best_gen_idx]["energy_kwh"]
            recs.append(("☀️ Peak Solar Window (3h)",
                          f"Best solar output at <b>{best_hour_lbl}</b> — <b>{best_gen_kwh:.4f} kWh</b>. Run high-power appliances then.",
                          "#F5A623", "recommendation-box"))

        if has_tariff:
            _t_list     = _tar["tariffs"]
            _sh         = _tar["start_hour"]
            best_t_idx  = int(np.argmin(_t_list))
            worst_t_idx = int(np.argmax(_t_list))
            best_t_hr   = (_sh + best_t_idx  + 1) % 24
            worst_t_hr  = (_sh + worst_t_idx + 1) % 24
            recs.append(("🟢 Cheapest Tariff — Buy / Import Window",
                          f"Lowest tariff at <b>{best_t_hr:02d}:00</b> (₹{_t_list[best_t_idx]:.2f}/kWh). Ideal for EV charging, water heaters, washing machines.",
                          "#3FB950", "recommendation-box"))
            recs.append(("🔴 Peak Tariff — Best Sell / Avoid Buying",
                          f"Highest tariff at <b>{worst_t_hr:02d}:00</b> (₹{_t_list[worst_t_idx]:.2f}/kWh). "
                          "Sell excess solar here for maximum revenue; defer grid-powered loads.",
                          "#FF6B6B", "warning-box"))

        if has_energy and has_tariff:
            _con_vals    = [f["consumption_kwh"] for f in _con["forecasts"][:12]]
            _tar_vals    = _tar["tariffs"]
            costs        = [c * t for c, t in zip(_con_vals, _tar_vals)]
            cheapest_idx = int(np.argmin(costs))
            cheapest_lbl = _con["forecasts"][cheapest_idx]["hour_label"]
            recs.append(("💰 Lowest Cost Hour",
                          f"Minimum electricity cost at <b>{cheapest_lbl}</b> — ₹{costs[cheapest_idx]:.2f} for that hour. Ideal for heavy loads.",
                          "#BC8CFF", "recommendation-box"))

        if recs:
            for title, body, color, box_class in recs:
                st.markdown(f"""
                <div class="{box_class}" style="border-color:{color}55;margin-bottom:0.6rem;">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;color:{color};font-size:0.95rem;margin-bottom:0.4rem">{title}</div>
                    <div style="font-size:0.88rem;color:#E6EDF3;">{body}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Run more models to generate combined recommendations.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#8B949E;font-size:0.8rem;">
        Combined Dashboard · Solar XGBoost · Consumption Hybrid (XGB+LSTM) · Tariff BiLSTM
    </div>""", unsafe_allow_html=True)
