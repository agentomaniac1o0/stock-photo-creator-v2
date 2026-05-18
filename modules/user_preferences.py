"""
User Preference Learning Module.

Learns the user's preferred brightness level from their exemplary
image selections. Provides a suggested EV offset that shifts the
exposure alignment target to match the user's taste.

Usage:
    from modules.user_preferences import suggest_brightness_offset

    # Provide one or more CR2 files the user considers "ideally exposed"
    exemplars = [Path("1551.CR2"), Path("1533.CR2")]
    offset = suggest_brightness_offset(exemplars)
    # -> e.g. 0.35 (meaning: push alignment +0.35 EV brighter)

The offset is cached in ~/.config/stock-photo-creator/user_preferences.json
and can be overridden manually.
"""
import json
import logging
from pathlib import Path

import numpy as np

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from modules.bracket_detector import FileExifData, BracketGroup, GroupType

logger = logging.getLogger(__name__)

PREFS_DIR = Path.home() / ".config" / "stock-photo-creator"
PREFS_FILE = PREFS_DIR / "user_preferences.json"
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}


def measure_raw_luminance(filepath: Path) -> float:
    """
    Measure the mean raw luminance of a CR2/RAW file.

    Uses rawpy to access the linear raw Bayer data (before any
    tone curve or gamma), averages across all pixels, and returns
    a value in [0, 1] representing the fraction of full well.

    For non-RAW files, falls back to PIL luminance estimate in [0, 255].

    Returns:
        Mean luminance (0-1 for RAW via rawpy, 0-255 for PIL fallback)
    """
    ext = filepath.suffix.lower()
    if ext in RAW_EXTENSIONS and HAS_RAWPY:
        try:
            with rawpy.imread(str(filepath)) as raw:
                raw_visible = raw.raw_image_visible
                white_level = float(raw.white_level) if raw.white_level else 16383.0
                mean_raw = float(np.mean(raw_visible))
                return mean_raw / white_level
        except Exception as e:
            logger.warning(f"rawpy failed for {filepath.name}: {e}")

    from PIL import Image
    try:
        img = Image.open(filepath).convert("RGB")
        arr = np.array(img).astype(np.float64)
        luminance = (0.2126 * arr[:, :, 0] +
                     0.7152 * arr[:, :, 1] +
                     0.0722 * arr[:, :, 2])
        return float(np.mean(luminance)) / 255.0
    except Exception as e:
        logger.error(f"Cannot measure luminance for {filepath.name}: {e}")
        return 0.5


def suggest_brightness_offset(
    exemplary_files: list[Path],
    reference_luminance: float = 0.25,
) -> float:
    """
    Suggest an EV brightness offset from user's exemplary images.

    Measures the raw luminance of each exemplar and computes the
    EV shift needed to reach that luminance from the pipeline's
    standard reference level.

    Args:
        exemplary_files: CR2 files the user considers "ideally exposed"
        reference_luminance: The pipeline's standard reference luminance
            fraction (0.25 ≈ 25% of full well for a typical 0EV image)

    Returns:
        Suggested brightness_offset in EV (positive = brighter).
        Returns 0.0 if no exemplars provided or measurement fails.
    """
    if not exemplary_files:
        logger.warning("No exemplary files provided, using offset=0.0")
        return 0.0

    luminances = []
    for fp in exemplary_files:
        lum = measure_raw_luminance(fp)
        if lum > 0.01:
            luminances.append(lum)
            logger.debug(f"  {fp.name}: raw luminance = {lum:.4f} "
                        f"({lum * 100:.1f}% of full well)")

    if not luminances:
        logger.warning("Could not measure any exemplars, using offset=0.0")
        return 0.0

    target_lum = float(np.median(luminances))
    ev_diff = np.log2(target_lum / reference_luminance)
    offset = float(np.clip(ev_diff, -2.0, 2.0))

    logger.info(f"User preference: median raw luminance={target_lum:.4f} "
                f"({target_lum*100:.1f}%), offset={offset:+.3f} EV")

    return offset


def save_preferences(preferences: dict) -> None:
    """Save user preferences to JSON file."""
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(preferences, indent=2))
    logger.info(f"Preferences saved to {PREFS_FILE}")


def load_preferences() -> dict:
    """Load user preferences from JSON file."""
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load preferences: {e}")
    return {}
