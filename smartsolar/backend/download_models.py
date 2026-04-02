"""
download_models.py
──────────────────
Run once at Render startup to pull model files from Google Drive.
Called automatically by start.sh before uvicorn launches.

Set these env vars on Render:
  SOLAR_FOLDER_ID      → Google Drive folder ID for saved_models/
  ENERGY_FOLDER_ID     → Google Drive folder ID for energy_consumption_saved_models/
"""

import os
import subprocess
import sys


def pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def download_folder(folder_id: str, dest: str):
    """Download a Google Drive folder using gdown."""
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "gdown",
        "--folder", f"https://drive.google.com/drive/folders/{folder_id}",
        "-O", dest,
        "--quiet",
    ])


def main():
    # Install gdown if not present
    try:
        import gdown  # noqa
    except ImportError:
        print("Installing gdown...")
        pip_install("gdown")

    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))

    solar_id  = os.environ.get("SOLAR_FOLDER_ID")
    energy_id = os.environ.get("ENERGY_FOLDER_ID")

    solar_dest  = os.path.join(root, "saved_models")
    energy_dest = os.path.join(root, "energy_consumption_saved_models")

    if solar_id:
        if os.path.isdir(solar_dest) and len(os.listdir(solar_dest)) >= 7:
            print(f"✓ Solar models already present in {solar_dest} — skipping download.")
        else:
            print(f"⬇ Downloading solar models → {solar_dest} ...")
            download_folder(solar_id, solar_dest)
            print("✓ Solar models downloaded.")
    else:
        print("⚠ SOLAR_FOLDER_ID not set — skipping solar model download.")

    if energy_id:
        if os.path.isdir(energy_dest) and len(os.listdir(energy_dest)) >= 4:
            print(f"✓ Energy models already present in {energy_dest} — skipping download.")
        else:
            print(f"⬇ Downloading energy models → {energy_dest} ...")
            download_folder(energy_id, energy_dest)
            print("✓ Energy models downloaded.")
    else:
        print("⚠ ENERGY_FOLDER_ID not set — skipping energy model download.")


if __name__ == "__main__":
    main()
