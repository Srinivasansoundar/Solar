"""
download_models.py
==================
Run once at Render startup (before uvicorn) to pull model files from Google Drive.

Google Drive folder links come from the project README:
  - Solar models     → saved_models/
  - Energy models    → energy_consumption_saved_models/
  - Tariff weights   → tariff_forcasting_model/

Usage (called automatically via render.yaml buildCommand):
    python download_models.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Google Drive folder IDs — taken from README.md sharing links
#   https://drive.google.com/drive/folders/<FOLDER_ID>
# ---------------------------------------------------------------------------
SOLAR_FOLDER_ID   = "1XqG_3399_OTNGNdEJ8jX81tDFSMMao1X"
ENERGY_FOLDER_ID  = "1k_uSKYQq0d2LNQeHJMHbvwZ9Li5PYDzb"
TARIFF_FOLDER_ID  = "1YwLhgDQGljp5xPgyacOXe6hmVzdg6eXQ"

# ---------------------------------------------------------------------------
# Destination directories — relative to smartsolar/
# (this script lives in smartsolar/backend/, so go one level up)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOLAR_DIR  = os.path.join(BASE_DIR, "saved_models")
ENERGY_DIR = os.path.join(BASE_DIR, "energy_consumption_saved_models")
TARIFF_DIR = os.path.join(BASE_DIR, "tariff_forcasting_model")


def download_folder(folder_id: str, dest_dir: str, label: str) -> bool:
    """Download all files in a Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        print(f"[download_models] gdown not installed — skipping {label}", flush=True)
        return False

    os.makedirs(dest_dir, exist_ok=True)

    # Check if folder already has files (avoid re-downloading on hot redeploy)
    existing = [f for f in os.listdir(dest_dir) if not f.startswith(".")]
    if existing:
        print(f"[download_models] ✓ {label} — already present ({len(existing)} files), skipping.", flush=True)
        return True

    print(f"[download_models] ⬇ Downloading {label} → {dest_dir} …", flush=True)
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=dest_dir, quiet=False, use_cookies=False)
        downloaded = [f for f in os.listdir(dest_dir) if not f.startswith(".")]
        print(f"[download_models] ✓ {label} — {len(downloaded)} files downloaded.", flush=True)
        return True
    except Exception as e:
        print(f"[download_models] ✗ {label} download failed: {e}", flush=True)
        return False


def main():
    print("[download_models] Starting model download …", flush=True)

    ok_solar  = download_folder(SOLAR_FOLDER_ID,  SOLAR_DIR,  "Solar models")
    ok_energy = download_folder(ENERGY_FOLDER_ID, ENERGY_DIR, "Energy models")
    ok_tariff = download_folder(TARIFF_FOLDER_ID, TARIFF_DIR, "Tariff weights")

    if not (ok_solar and ok_energy):
        print(
            "[download_models] ⚠ Some models failed to download. "
            "Backend will start but predictions may fail.",
            flush=True,
        )
    else:
        print("[download_models] ✅ All models ready.", flush=True)


if __name__ == "__main__":
    main()
