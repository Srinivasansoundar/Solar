"""
Quick end-to-end health check for all 3 SmartSolar models.
Run: python test_models.py
"""
import sys, os, math, requests
from datetime import datetime, timedelta

API = "http://localhost:8000"
PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"

errors = []

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SmartSolar Model Health Check")
print("="*60)

# ── 1. Backend reachability ───────────────────────────────────────────────────
print("\n[1] Backend API")
try:
    r = requests.get(f"{API}/health", timeout=5).json()
    if r.get("models_loaded"):
        print(f"  Solar XGBoost models : {PASS}  (window {r['window']})")
    else:
        print(f"  Solar XGBoost models : {FAIL}  models_loaded=False")
        errors.append("Solar XGBoost not loaded")
except Exception as e:
    print(f"  Backend reachable    : {FAIL}  {e}")
    errors.append(f"Backend offline: {e}")
    sys.exit(1)

try:
    re = requests.get(f"{API}/energy-status", timeout=5).json()
    xgb_ok  = re.get("xgb_loaded", False)
    lstm_ok = re.get("lstm_loaded", False)
    hybrid  = re.get("hybrid_ready", False)
    print(f"  Energy XGB           : {PASS if xgb_ok  else FAIL}")
    print(f"  Energy LSTM          : {PASS if lstm_ok else WARN + '  (XGB-only mode)'}")
    print(f"  Hybrid (XGB+LSTM)    : {PASS if hybrid  else WARN}")
    if not xgb_ok:
        errors.append("Energy XGB not loaded")
except Exception as e:
    print(f"  Energy status        : {FAIL}  {e}")
    errors.append(f"Energy status error: {e}")

# ── 2. Solar /predict ─────────────────────────────────────────────────────────
print("\n[2] Solar Forecast  (POST /predict) — 3h ahead")
now = datetime.now()
rows = []
for i in range(36):
    dt   = now - timedelta(hours=36 - i)
    h    = dt.hour
    elev = max(0, 90 - abs(h - 12) * 7.5)
    ghi  = max(0, elev * 10 + 50)
    rows.append({
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "GHI": ghi, "DNI": ghi * 0.7, "DHI": ghi * 0.25,
        "T_amb": 28.0, "RH": 55.0, "WS": 3.5, "WS_gust": 5.2,
        "WD": 180.0, "WD_std": 15.0, "BP": 1013.0,
    })
try:
    r = requests.post(f"{API}/predict",
                      json={"historical_data": rows,
                            "panel_area_m2": 10.0,
                            "panel_efficiency": 0.20},
                      timeout=30)
    if r.status_code == 200:
        d = r.json()
        print(f"  Status               : {PASS}  (HTTP 200)")
        print(f"  Total energy 3h      : {d['total_energy_kwh']} kWh")
        for f in d["forecasts"]:
            ghi_val = f["ghi_wm2"]
            kwh_val = f["energy_kwh"]
            cat     = f["irradiance_category"]
            print(f"    t+{f['step']}h  {f['hour_label']}  |  GHI={ghi_val:.1f} W/m²  |  {kwh_val:.4f} kWh  |  {cat}")
        # Sanity checks
        if d["total_energy_kwh"] < 0:
            errors.append("Solar: negative energy output")
            print(f"  Sanity check         : {FAIL}  negative energy!")
        else:
            print(f"  Sanity check         : {PASS}  values non-negative")
    else:
        print(f"  Status               : {FAIL}  HTTP {r.status_code}: {r.text[:200]}")
        errors.append(f"Solar predict HTTP {r.status_code}")
except Exception as e:
    print(f"  Status               : {FAIL}  {e}")
    errors.append(f"Solar predict error: {e}")

# ── 3. Energy /predict-energy ─────────────────────────────────────────────────
print("\n[3] Energy Consumption (POST /predict-energy) — 24h ahead")
rows_e = []
for i in range(40):
    dt   = now - timedelta(hours=40 - i)
    h    = dt.hour
    load = 0.5 + 1.5 * (
        0.5 * math.exp(-((h - 8) ** 2) / 8) +
        0.5 * math.exp(-((h - 19) ** 2) / 8)
    )
    rows_e.append({
        "datetime":           dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Global_active_power": round(load, 3),
    })
try:
    r = requests.post(f"{API}/predict-energy",
                      json={"historical_data": rows_e},
                      timeout=60)
    if r.status_code == 200:
        d = r.json()
        print(f"  Status               : {PASS}  (HTTP 200)")
        print(f"  Total 24h consumption: {d['total_consumption_kwh']} kWh")
        vals = [f["consumption_kwh"] for f in d["forecasts"]]
        print(f"  Hours returned       : {len(vals)} (expected 24)")
        print(f"  Min / Max per hour   : {min(vals):.4f} / {max(vals):.4f} kWh")
        print(f"  First 6 hours:")
        for f in d["forecasts"][:6]:
            print(f"    {f['hour_label']}  {f['consumption_kwh']:.4f} kWh")
        if len(vals) != 24:
            errors.append(f"Energy: returned {len(vals)} hours instead of 24")
            print(f"  Hour count check     : {FAIL}")
        elif min(vals) < 0:
            errors.append("Energy: negative consumption value")
            print(f"  Sanity check         : {FAIL}  negative consumption!")
        else:
            print(f"  Sanity check         : {PASS}  24 h, all non-negative")
    else:
        print(f"  Status               : {FAIL}  HTTP {r.status_code}: {r.text[:200]}")
        errors.append(f"Energy predict HTTP {r.status_code}")
except Exception as e:
    print(f"  Status               : {FAIL}  {e}")
    errors.append(f"Energy predict error: {e}")

# ── 4. Tariff BiLSTM ──────────────────────────────────────────────────────────
print("\n[4] Tariff Forecast (BiLSTM — client-side)")
_TARIFF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tariff_forcasting_model"))
sys.path.insert(0, _TARIFF_DIR)
try:
    from lstm_model import predict_tariff
    tariffs, stats = predict_tariff(
        hour_of_day=10.0, day_of_week=4.0, month=3.0,
        temperature=30.0, historical_avg=6.5, demand_index=0.5,
    )
    print(f"  Status               : {PASS}  model executed")
    print(f"  Hours returned       : {len(tariffs)} (expected 12)")
    print(f"  Min / Avg / Max      : ₹{stats['min']:.2f} / ₹{stats['avg']:.2f} / ₹{stats['max']:.2f} per kWh")
    print(f"  Tariffs: {[round(t, 2) for t in tariffs]}")
    if len(tariffs) != 12:
        errors.append(f"Tariff: returned {len(tariffs)} hours instead of 12")
        print(f"  Hour count check     : {FAIL}")
    elif any(t < 0 for t in tariffs):
        errors.append("Tariff: negative value found")
        print(f"  Sanity check         : {FAIL}  negative tariff!")
    else:
        print(f"  Sanity check         : {PASS}  12 h, all non-negative")
except Exception as e:
    print(f"  Status               : {FAIL}  {e}")
    errors.append(f"Tariff model error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"  RESULT: {FAIL}  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    • {e}")
else:
    print(f"  RESULT: {PASS}  All 3 models are working correctly!")
print("="*60 + "\n")
