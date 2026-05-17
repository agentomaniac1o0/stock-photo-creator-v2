"""
Configuration for Stock Photo Creator v2
"""
import os
from pathlib import Path

# ── Nextcloud ──────────────────────────────────────────────────────────────────

NEXTCLOUD_HOST = os.getenv("NEXTCLOUD_HOST", "https://100.75.220.89").rstrip("/")
NEXTCLOUD_USER = os.getenv("NEXTCLOUD_USER", "nerdclaudeadm")
NEXTCLOUD_APP_PASSWORD = os.getenv("NEXTCLOUD_APP_PASSWORD", "")

NC_BASE_PATH = "Photos/StockFotoCreator"
NC_RAW_PATH = os.getenv("NC_RAW_PATH", f"{NC_BASE_PATH}/RAW")

# ── Bracket Detection ─────────────────────────────────────────────────────────

BRACKET_TIME_TOLERANCE_SEC = 3.0  # 0-3 seconds window
BRACKET_EV_DIFF_THRESHOLD = 0.2  # EV difference to consider "different exposure"

# ── Quality Gate ───────────────────────────────────────────────────────────────

# Overexposure: if > this ratio of pixels are clipped, image is unrecoverable
OVEREXPOSURE_CLIPPING_THRESHOLD = 0.02  # 2% of pixels at max value

# ── File Types ─────────────────────────────────────────────────────────────────

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}
JPEG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
ALL_IMAGE_EXTENSIONS = RAW_EXTENSIONS | JPEG_EXTENSIONS

# ── Local Temp ─────────────────────────────────────────────────────────────────

LOCAL_TEMP_DIR = Path.home() / "stock-pipeline-temp-v2"
