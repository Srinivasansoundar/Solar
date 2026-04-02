"""
BiLSTM Tariff Forecasting Model Helper
=======================================
Architecture (matches tariff_lstm_weights.pth):
  - 2-layer Bidirectional LSTM
  - input_size  = 14
  - hidden_size = 128
  - output_size = 12  (12-hour forecast horizon)

Public API:
    from lstm_model import predict_tariff
    tariffs, stats = predict_tariff(hour_of_day, day_of_week, month,
                                    temperature, historical_avg, demand_index)
    # tariffs → list[float] len=12, stats → dict with min/max/avg/peak_hour/off_peak_hour
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn

# ── Model definition ──────────────────────────────────────────────────────────

class TariffBiLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM followed by a linear output head.
    Must match the architecture that produced tariff_lstm_weights.pth.
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional → 2 × hidden_size per timestep
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, 2*hidden)
        last    = out[:, -1, :]        # take last timestep
        return self.fc(last)           # (batch, output_size)


# ── Weight loading (cached after first call) ──────────────────────────────────

_model: TariffBiLSTM | None = None

def _load_model() -> TariffBiLSTM:
    global _model
    if _model is not None:
        return _model

    weights_path = os.path.join(os.path.dirname(__file__), "tariff_lstm_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Make sure tariff_lstm_weights.pth is placed in the models/ folder."
        )

    model = TariffBiLSTM(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        output_size=12,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return model


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_input_sequence(
    hour_of_day:    float,
    day_of_week:    float,
    month:          float,
    temperature:    float,
    historical_avg: float,
    demand_index:   float,
    seq_len:        int = 24,
) -> torch.Tensor:
    """
    Build a (1, seq_len, 14) input tensor from the 6 UI parameters.

    Feature layout (14 dims):
      0  hour_sin
      1  hour_cos
      2  day_sin
      3  day_cos
      4  month_sin
      5  month_cos
      6  temperature_norm      (normalised to [-1, 1] over [-10°C, 50°C])
      7  demand_index          (already 0–1)
      8  hist_avg_norm         (normalised to [0, 1] over [0, 20] ₹/kWh)
      9  tariff_lag_1          (synthetic lag from hist_avg ± demand noise)
      10 tariff_lag_2
      11 tariff_lag_3
      12 tariff_lag_6
      13 tariff_lag_12
    """
    rng = np.random.default_rng(seed=int(hour_of_day * 100 + day_of_week))

    frames = []
    for step in range(seq_len):
        # Slide the "current hour" backwards so last frame = current hour
        h = (hour_of_day - (seq_len - 1 - step)) % 24

        h_sin = math.sin(2 * math.pi * h / 24)
        h_cos = math.cos(2 * math.pi * h / 24)
        d_sin = math.sin(2 * math.pi * day_of_week / 7)
        d_cos = math.cos(2 * math.pi * day_of_week / 7)
        m_sin = math.sin(2 * math.pi * month / 12)
        m_cos = math.cos(2 * math.pi * month / 12)

        temp_norm   = (temperature - 20.0) / 30.0          # roughly [-1, 1]
        demand      = float(demand_index)
        hist_norm   = float(historical_avg) / 10.0          # roughly [0, 2]

        # Synthetic lag values: historical avg with small demand-driven noise
        noise = 0.1 * demand + 0.05
        lag1  = hist_norm + rng.normal(0, noise)
        lag2  = hist_norm + rng.normal(0, noise)
        lag3  = hist_norm + rng.normal(0, noise)
        lag6  = hist_norm + rng.normal(0, noise * 1.3)
        lag12 = hist_norm + rng.normal(0, noise * 1.6)

        frames.append([
            h_sin, h_cos,
            d_sin, d_cos,
            m_sin, m_cos,
            temp_norm,
            demand,
            hist_norm,
            lag1, lag2, lag3, lag6, lag12,
        ])

    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 14)
    return x


# ── Public API ────────────────────────────────────────────────────────────────

def predict_tariff(
    hour_of_day:    float,
    day_of_week:    float,
    month:          float,
    temperature:    float,
    historical_avg: float,
    demand_index:   float,
) -> tuple[list[float], dict]:
    """
    Run the BiLSTM tariff model and return predictions for the next 12 hours.

    Returns
    -------
    tariffs : list[float]
        Predicted tariff in ₹/kWh for each of the next 12 hours.
    stats : dict
        Keys: min, max, avg, peak_hour, off_peak_hour
    """
    model = _load_model()

    with torch.no_grad():
        x = _build_input_sequence(
            hour_of_day, day_of_week, month,
            temperature, historical_avg, demand_index,
        )
        raw: torch.Tensor = model(x)      # (1, 12)

    # Post-process: scale back from normalised output to ₹/kWh
    # The model output is normalised relative to historical_avg.
    # We centre around historical_avg and add realistic spread.
    raw_np = raw.squeeze(0).numpy()       # shape (12,)

    # Sigmoid to map to (0, 1), then rescale around historical_avg
    scaled = torch.sigmoid(raw).squeeze(0).numpy()

    # Map to ₹ range: ±50 % around historical_avg; span proportional to demand
    spread = historical_avg * (0.4 + 0.2 * demand_index)
    base   = historical_avg * (0.8 - 0.1 * demand_index)
    tariffs_np = base + scaled * spread

    # Clip to realistic Indian tariff range (₹1–₹25/kWh)
    tariffs_np = np.clip(tariffs_np, 1.0, 25.0)
    tariffs = tariffs_np.tolist()

    peak_hour     = int(np.argmax(tariffs))
    off_peak_hour = int(np.argmin(tariffs))

    stats = {
        "min":          float(np.min(tariffs)),
        "max":          float(np.max(tariffs)),
        "avg":          float(np.mean(tariffs)),
        "peak_hour":    peak_hour,
        "off_peak_hour": off_peak_hour,
    }

    return tariffs, stats
