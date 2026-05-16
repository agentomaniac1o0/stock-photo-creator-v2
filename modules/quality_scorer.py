"""
Module 05: Quality Scorer (v2 — Metric-Only)

Computes individual quality metrics for each image:
- Exposure (histogram-based: mean, percentiles, shadow/highlight balance)
- Noise (shadow-focused: high-pass energy in dark areas)
- Sharpness (combined Laplacian + Tenengrad)
- Detail (local contrast)
- Defects (chromatic aberration, distortion, etc.)

No overall score — metrics are used individually by the selector
for multi-stage filtering and comparison.

Input:  List of BracketGroup objects
Output: Same groups with quality metrics attached to each file
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import convolve as scipy_convolve

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)


@dataclass
class ImageMetrics:
    """Individual quality metrics for a single image."""
    filepath: Path
    exposure_score: float  # 0-100 (higher = well-exposed)
    noise_score: float  # 0-100 (higher = less noise)
    sharpness_score: float  # 0-100
    detail_score: float  # 0-100
    defect_score: float  # 0-100 (higher = fewer defects)

    @property
    def filename(self) -> str:
        return self.filepath.name

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "exposure_score": round(self.exposure_score, 1),
            "noise_score": round(self.noise_score, 1),
            "sharpness_score": round(self.sharpness_score, 1),
            "detail_score": round(self.detail_score, 1),
            "defect_score": round(self.defect_score, 1),
        }


@dataclass
class MetricsResult:
    """Result of metric computation for all groups."""
    scored_groups: list[BracketGroup]
    metrics: list[ImageMetrics]
    total_scored: int = 0
    total_failed: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  QUALITY METRICS SUMMARY",
            f"{'='*60}",
            f"  Images scored:   {self.total_scored}",
            f"  Images failed:   {self.total_failed}",
        ]
        if self.metrics:
            lines.append("")
            lines.append("  Metrics:")
            for m in sorted(self.metrics, key=lambda x: x.noise_score, reverse=True):
                lines.append(f"    {m.filename}: "
                           f"exp={m.exposure_score:.0f}, noise={m.noise_score:.0f}, "
                           f"sharp={m.sharpness_score:.0f}, detail={m.detail_score:.0f}, "
                           f"defects={m.defect_score:.0f}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}


def load_image_array(filepath: Path) -> np.ndarray:
    """
    Load an image as a numpy array.

    For RAW files (CR2/CR3/etc): uses rawpy to render full-resolution RGB.
    For JPEG/PNG: uses PIL.

    Returns:
        numpy array of shape (H, W, 3) with uint8 values.
    """
    ext = filepath.suffix.lower()

    if ext in RAW_EXTENSIONS and HAS_RAWPY:
        try:
            with rawpy.imread(str(filepath)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=False,
                )
            return rgb
        except Exception as e:
            logger.warning(f"rawpy failed for {filepath.name}: {e}, falling back to PIL")

    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def compute_exposure_score(gray: np.ndarray) -> float:
    """
    Score exposure quality using histogram analysis.

    Ideal: Mean 35-65%, Q1 (25th percentile) > 15%, Q3 (75th) < 90%
    Underexposed: Mean < 25% or Q1 < 10%
    Overexposed: Mean > 75% or Q3 > 95%

    Args:
        gray: Grayscale array (uint8, 0-255)

    Returns:
        Score 0-100 (higher = well-exposed)
    """
    mean_val = float(np.mean(gray)) / 255.0
    q1 = float(np.percentile(gray, 25)) / 255.0
    q3 = float(np.percentile(gray, 75)) / 255.0

    score = 100.0

    if mean_val < 0.15:
        penalty = 80 + ((0.15 - mean_val) / 0.15) * 15
        score -= penalty
    elif mean_val < 0.25:
        penalty = 50 + ((0.25 - mean_val) / 0.10) * 20
        score -= penalty
    elif mean_val < 0.35:
        penalty = ((0.35 - mean_val) / 0.10) * 20
        score -= penalty

    if mean_val > 0.85:
        penalty = 70 + ((mean_val - 0.85) / 0.15) * 20
        score -= penalty
    elif mean_val > 0.75:
        penalty = 40 + ((mean_val - 0.75) / 0.10) * 20
        score -= penalty
    elif mean_val > 0.65:
        penalty = ((mean_val - 0.65) / 0.10) * 15
        score -= penalty

    if q1 < 0.05:
        penalty = 30 + ((0.05 - q1) / 0.05) * 20
        score -= penalty
    elif q1 < 0.10:
        penalty = ((0.10 - q1) / 0.05) * 20
        score -= penalty

    if q3 > 0.98:
        penalty = 30 + ((q3 - 0.98) / 0.02) * 20
        score -= penalty
    elif q3 > 0.95:
        penalty = ((q3 - 0.95) / 0.03) * 20
        score -= penalty

    return max(min(score, 100.0), 0.0)


def compute_shadow_noise(gray: np.ndarray) -> float:
    """
    Score noise level with focus on shadow areas.

    Noise is most visible and hardest to fix in dark areas.
    Uses high-pass filter energy on shadow pixels only.

    Args:
        gray: Grayscale array (uint8, 0-255)

    Returns:
        Score 0-100 (higher = less noise)
    """
    shadow_mask = gray < (255 * 0.30)

    if not np.any(shadow_mask):
        return 80.0

    blurred = scipy_convolve(
        gray.astype(np.float64),
        np.ones((5, 5), dtype=np.float64) / 25.0
    )
    highpass = gray.astype(np.float64) - blurred
    shadow_highpass = highpass[shadow_mask]

    noise_std = float(np.std(shadow_highpass))

    noise_score = max(100.0 - (noise_std * 8), 0.0)
    return noise_score


def compute_tenengrad(gray: np.ndarray) -> float:
    """
    Score sharpness using the Tenengrad method (Sobel-based).

    More robust against noise artifacts than Laplacian variance.
    Measures edge strength via gradient magnitude, ignoring flat areas
    where noise can masquerade as sharpness.

    Args:
        gray: Grayscale array (uint8, 0-255)

    Returns:
        Score 0-100 (higher = sharper)
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = scipy_convolve(gray.astype(np.float64), sobel_x)
    gy = scipy_convolve(gray.astype(np.float64), sobel_y)

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    threshold = np.mean(gradient_magnitude) * 1.5
    edge_mask = gradient_magnitude > threshold

    if not np.any(edge_mask):
        return 5.0

    tenengrad_value = float(np.mean(gradient_magnitude[edge_mask] ** 2))

    score = min(tenengrad_value / 200.0, 100.0)
    return max(score, 0.0)


def compute_combined_sharpness(gray: np.ndarray, noise_score: float) -> float:
    """
    Combined sharpness score from Laplacian and Tenengrad.

    Laplacian (40%): Sensitive to fine detail, dampened by noise.
    Tenengrad (60%): Robust edge detection, less noise-sensitive.

    Args:
        gray: Grayscale array (uint8, 0-255)
        noise_score: Pre-computed noise score (0-100, higher = less noise)

    Returns:
        Combined sharpness score 0-100
    """
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = scipy_convolve(gray, laplacian_kernel)
    lap_var = float(np.var(lap))
    lap_score = min(lap_var / 5.0, 100.0) * (noise_score / 100.0)

    tenengrad_score = compute_tenengrad(gray)

    combined = lap_score * 0.4 + tenengrad_score * 0.6
    return max(min(combined, 100.0), 0.0)


def noise_curve(noise_score: float) -> float:
    """
    Non-linear noise scoring: diminishing returns for clean images,
    steep penalty for heavy noise.

    < 40: halved (heavy noise heavily penalized)
    40-70: linear (moderate noise)
    > 70: diminishing returns (clean images get less bonus)
    """
    if noise_score < 40:
        return noise_score * 0.5
    elif noise_score < 70:
        return noise_score
    else:
        return 70 + (noise_score - 70) * 0.5


def sharpness_curve(sharp_score: float) -> float:
    """
    Non-linear sharpness scoring: penalty for blurry images,
    bonus for tack-sharp images.

    < 15: halved (slightly blurry penalized)
    15-35: linear (acceptable sharpness)
    > 35: 1.5x bonus (sharp images rewarded)
    """
    if sharp_score < 15:
        return sharp_score * 0.5
    elif sharp_score < 35:
        return sharp_score
    else:
        return 35 + (sharp_score - 35) * 1.5


def exposure_correction_penalty(filepath: Path) -> float:
    """
    Penalty for images that were exposure-corrected (brightened).

    Underexposed images that were brightened have statistically higher
    noise and are more likely to be unusable. Only applies to files
    with '_exposure_corrected' in the name.

    Returns:
        Penalty value (0-10) to subtract from comparison score.
    """
    if "_exposure_corrected" in filepath.name:
        return 5.0
    return 0.0


def compute_metrics(filepath: Path) -> ImageMetrics:
    """
    Compute individual quality metrics for a single image.

    No overall score — metrics are used individually by the selector.

    Uses:
    - Histogram analysis for exposure scoring
    - Shadow-focused noise detection
    - Combined Laplacian + Tenengrad for sharpness
    - Local contrast for detail
    - Color channel correlation for defects
    """
    try:
        arr = load_image_array(filepath)
        gray = np.mean(arr.astype(float), axis=2)

        exposure_score = compute_exposure_score(gray)
        noise_score = compute_shadow_noise(gray)
        sharpness_score = compute_combined_sharpness(gray, noise_score)
        detail_score = min(float(np.std(gray)) / 2.0, 100.0)

        r_channel = arr[:, :, 0].astype(float)
        b_channel = arr[:, :, 2].astype(float)
        rb_diff = np.abs(r_channel - b_channel)
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        lap = scipy_convolve(gray, laplacian_kernel)
        edge_mask = np.abs(lap) > np.mean(np.abs(lap)) * 2
        if np.any(edge_mask):
            ca_score = float(np.mean(rb_diff[edge_mask]))
            defect_score = max(100.0 - (ca_score * 2), 0.0)
        else:
            defect_score = 70.0

        return ImageMetrics(
            filepath=filepath,
            exposure_score=exposure_score,
            noise_score=noise_score,
            sharpness_score=sharpness_score,
            detail_score=detail_score,
            defect_score=defect_score,
        )

    except Exception as e:
        logger.error(f"Metric computation error for {filepath.name}: {e}")
        return ImageMetrics(
            filepath=filepath,
            exposure_score=50.0,
            noise_score=50.0,
            sharpness_score=50.0,
            detail_score=50.0,
            defect_score=50.0,
        )


def compute_all_metrics(groups: list[BracketGroup]) -> MetricsResult:
    """
    Compute metrics for all images in all groups.

    For each group:
    1. Compute metrics for every image
    2. Attach metrics to FileExifData (via ImageMetrics)
    3. Return groups with metrics

    Args:
        groups: List of BracketGroup objects

    Returns:
        MetricsResult with scored groups
    """
    scored_groups = []
    metrics = []
    total_scored = 0
    total_failed = 0

    for group in groups:
        logger.info(f"Computing metrics for group #{group.group_id} ({group.group_type.value}, "
                   f"{group.file_count} files)")

        group_metrics = []
        for fd in group.files:
            try:
                m = compute_metrics(fd.filepath)
            except Exception as e:
                logger.error(f"  Unexpected error computing metrics for {fd.filename}: {e}")
                m = ImageMetrics(
                    filepath=fd.filepath,
                    exposure_score=50.0,
                    noise_score=50.0,
                    sharpness_score=50.0,
                    detail_score=50.0,
                    defect_score=50.0,
                )
            metrics.append(m)
            group_metrics.append(m)

            has_default = (
                m.exposure_score == 50.0
                and m.noise_score == 50.0
                and m.sharpness_score == 50.0
                and m.detail_score == 50.0
                and m.defect_score == 50.0
            )
            if has_default:
                total_failed += 1
            else:
                total_scored += 1

            logger.info(f"  {fd.filename}: "
                       f"exp={m.exposure_score:.0f}, noise={m.noise_score:.0f}, "
                       f"sharp={m.sharpness_score:.0f}")

        scored_files = sorted(
            zip(group.files, group_metrics),
            key=lambda x: (
                noise_curve(x[1].noise_score),
                x[1].defect_score,
                sharpness_curve(x[1].sharpness_score),
            ),
            reverse=True
        )

        new_group = BracketGroup(
            group_type=group.group_type,
            files=[fd for fd, _ in scored_files],
            group_id=group.group_id,
        )
        scored_groups.append(new_group)

    result = MetricsResult(
        scored_groups=scored_groups,
        metrics=metrics,
        total_scored=total_scored,
        total_failed=total_failed,
    )

    logger.info(result.summary())
    return result
