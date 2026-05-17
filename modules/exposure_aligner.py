"""
Module 04: Exposure Aligner (v2 — Reference-Based)

Aligns all AEB images to a reference exposure level.

Reference selection:
- 3-image AEB: middle exposure image (by exposure_time)
- <3 images: best histogram (closest to mid-gray)

All images except the reference are corrected to match the reference.
This ensures fair comparison of noise/sharpness/defects at the same exposure.

For Burst groups and Singles: no correction needed (same exposure).

Input:  List of BracketGroup objects (after overexposure check)
Output: Same groups with corrected images (in-place or new files)
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)


@dataclass
class ExposureCorrection:
    """Correction parameters for a single image."""
    filepath: Path
    original_ev: float
    target_ev: float
    brightness_adjustment: float
    highlights_adjustment: float
    shadows_adjustment: float
    corrected_path: Optional[Path] = None
    success: bool = False
    reason: str = ""
    is_reference: bool = False

    @property
    def filename(self) -> str:
        return self.filepath.name


@dataclass
class ExposureAlignResult:
    """Result of the exposure alignment for all groups."""
    aligned_groups: list[BracketGroup]
    corrections: list[ExposureCorrection]
    total_corrected: int = 0
    total_failed: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  EXPOSURE ALIGNMENT SUMMARY",
            f"{'='*60}",
            f"  Images corrected: {self.total_corrected}",
            f"  Images failed:    {self.total_failed}",
        ]
        if self.corrections:
            lines.append("")
            lines.append("  Corrections applied:")
            for c in self.corrections:
                if c.is_reference:
                    lines.append(f"    {c.filename}: REFERENCE (no correction)")
                    continue
                status = "OK" if c.success else "FAILED"
                lines.append(f"    {c.filename}: EV {c.original_ev:+.2f} → {c.target_ev:+.2f} "
                           f"[{status}]")
                if c.reason:
                    lines.append(f"      {c.reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def estimate_image_brightness(filepath: Path) -> float:
    """
    Estimate the average brightness of an image (0-255 scale).

    Uses luminance-weighted average for RGB images.
    """
    try:
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)
        luminance = (0.2126 * arr[:, :, 0] +
                    0.7152 * arr[:, :, 1] +
                    0.0722 * arr[:, :, 2])

        return float(np.mean(luminance))
    except Exception as e:
        logger.error(f"Brightness estimation error for {filepath}: {e}")
        return 128.0


def correct_exposure_pillow(
    filepath: Path,
    output_path: Path,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
) -> bool:
    """
    Correct exposure using Pillow (for JPEGs or when rawpy unavailable).

    Args:
        filepath: Input image path
        output_path: Output image path
        brightness_factor: >1 = brighter, <1 = darker
        contrast_factor: >1 = more contrast, <1 = less
    """
    try:
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if brightness_factor != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        if contrast_factor != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=95)
        return True

    except Exception as e:
        logger.error(f"Pillow exposure correction error for {filepath}: {e}")
        return False


def correct_exposure_rawpy(
    filepath: Path,
    output_path: Path,
    brightness_adjustment: float = 0.0,
) -> bool:
    """
    Correct exposure using rawpy (for RAW files).

    Applies exposure compensation during RAW development.

    Args:
        filepath: Input RAW file path
        output_path: Output JPEG path
        brightness_adjustment: EV compensation value
    """
    try:
        import rawpy

        with rawpy.imread(str(filepath)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
                bright=brightness_adjustment,
            )

        img = Image.fromarray(rgb)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=95)
        return True

    except ImportError:
        logger.warning("rawpy not available, falling back to Pillow")
        return False
    except Exception as e:
        logger.error(f"rawpy exposure correction error for {filepath}: {e}")
        return False


def _parse_exposure_time(exp_value) -> float:
    """Parse exposure time to float seconds."""
    if isinstance(exp_value, (int, float)):
        return float(exp_value)
    exp_str = str(exp_value)
    if "/" in exp_str:
        parts = exp_str.split("/")
        return float(parts[0]) / float(parts[1])
    return float(exp_str)


def compute_toncurve_score(filepath: Path) -> float:
    """
    Score how 'natural' the toncurve of an image is.

    The best reference image has:
    - Brightness near mid-gray (128) — not under/overexposed
    - Low clipping ratio — highlights are preserved
    - Wide histogram — good contrast range

    Returns:
        Score 0-100 (higher = better reference candidate)
    """
    try:
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)
        luminance = (0.2126 * arr[:, :, 0] +
                     0.7152 * arr[:, :, 1] +
                     0.0722 * arr[:, :, 2])

        mean_val = float(np.mean(luminance))

        # Brightness score: ideal = 128 (mid-gray)
        brightness_score = max(0, 100 - abs(mean_val - 128) * 0.8)

        # Clipping penalty: pixels at 250-255
        clipped = float(np.sum(luminance >= 250)) / luminance.size
        clipping_penalty = clipped * 100

        # Dynamic range bonus: wider histogram = more natural
        std_val = float(np.std(luminance))
        dr_bonus = min(std_val * 0.5, 15)

        # Q1/Q3 balance: well-exposed should have detail in shadows AND highlights
        q1 = float(np.percentile(luminance, 10))
        q3 = float(np.percentile(luminance, 90))
        range_score = min((q3 - q1) / 2.55, 60)

        total = brightness_score - clipping_penalty + dr_bonus + range_score * 0.2
        return max(0, min(total, 100))

    except Exception as e:
        logger.warning(f"Toncurve score error for {filepath.name}: {e}")
        return 50.0


def select_reference_image(files: list[FileExifData]) -> FileExifData:
    """
    Select the reference image for exposure alignment.

    Strategy: pick the image with the BEST TONKURVE.
    In AEB brackets, this is almost always the 0EV (normal exposure):
    - Not underexposed (stretched curve, more noise)
    - Not overexposed (clipped highlights)
    - Most natural contrast distribution

    Uses a composite score based on:
    - Brightness closeness to mid-gray
    - Minimal clipping
    - Good contrast range

    Args:
        files: List of FileExifData in the AEB group

    Returns:
        The reference FileExifData (best toncurve)
    """
    scored = []
    for fd in files:
        score = compute_toncurve_score(fd.filepath)
        scored.append((score, fd))
        logger.debug(f"  Toncurve {fd.filename}: {score:.1f}")

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    logger.info(f"  Reference: {best.filename} (toncurve score={scored[0][0]:.1f})")
    return best


def calculate_correction_params(
    file_data: FileExifData,
    reference_fd: FileExifData,
) -> ExposureCorrection:
    """
    Calculate correction parameters to align an image to reference exposure.

    Uses actual ExposureTime ratio instead of ExposureCompensation,
    since AEB groups have identical ExposureCompensation but different
    ExposureTime across the bracket.

    Args:
        file_data: EXIF data of the image to correct
        reference_fd: EXIF data of the reference image

    Returns:
        ExposureCorrection with calculated parameters
    """
    t_ref = _parse_exposure_time(reference_fd.exposure_time)
    t_cur = _parse_exposure_time(file_data.exposure_time)
    brightness_factor = t_ref / t_cur

    ev_diff = math.log2(brightness_factor)
    target_ev = reference_fd.exposure_compensation

    if ev_diff > 0:
        highlights_adjustment = min(ev_diff * 30, 100)
        shadows_adjustment = min(ev_diff * 50, 100)
    else:
        highlights_adjustment = max(ev_diff * 30, -100)
        shadows_adjustment = max(ev_diff * 50, -100)

    return ExposureCorrection(
        filepath=file_data.filepath,
        original_ev=file_data.exposure_compensation,
        target_ev=target_ev,
        brightness_adjustment=brightness_factor,
        highlights_adjustment=highlights_adjustment,
        shadows_adjustment=shadows_adjustment,
    )


def apply_correction(
    correction: ExposureCorrection,
    output_dir: Path = None,
) -> ExposureCorrection:
    """
    Apply exposure correction to an image.

    Args:
        correction: Calculated correction parameters
        output_dir: Directory for corrected files (default: same as input)

    Returns:
        Updated ExposureCorrection with result
    """
    if output_dir is None:
        output_dir = correction.filepath.parent

    stem = correction.filepath.stem
    corrected_name = f"{stem}_exposure_corrected.jpg"
    corrected_path = output_dir / corrected_name
    correction.corrected_path = corrected_path

    is_raw = correction.filepath.suffix.lower() in {
        ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"
    }

    if is_raw:
        success = correct_exposure_rawpy(
            correction.filepath,
            corrected_path,
            brightness_adjustment=correction.brightness_adjustment,
        )
        if not success:
            success = correct_exposure_pillow(
                correction.filepath,
                corrected_path,
                brightness_factor=correction.brightness_adjustment,
            )
    else:
        success = correct_exposure_pillow(
            correction.filepath,
            corrected_path,
            brightness_factor=correction.brightness_adjustment,
        )

    correction.success = success
    if success:
        correction.reason = (f"Corrected: brightness x{correction.brightness_adjustment:.2f}, "
                           f"highlights {correction.highlights_adjustment:+.0f}, "
                           f"shadows {correction.shadows_adjustment:+.0f}")
    else:
        correction.reason = "Correction failed"

    return correction


def align_exposures(
    groups: list[BracketGroup],
    output_dir: Path = None,
) -> ExposureAlignResult:
    """
    Align exposures for all AEB groups to a reference image.

    For each AEB group:
    1. Select reference image (middle for 3-image, best histogram for <3)
    2. Correct ALL other images to match reference exposure
    3. Update the group with corrected file paths

    Burst groups and Singles are passed through unchanged.
    """
    aligned_groups = []
    corrections = []
    total_corrected = 0
    total_failed = 0

    for group in groups:
        if group.group_type != GroupType.AEB:
            aligned_groups.append(group)
            continue

        if len(group.files) < 1:
            continue

        reference_fd = select_reference_image(group.files)
        logger.info(f"AEB group #{group.group_id}: reference = {reference_fd.filename} "
                   f"(EV={reference_fd.exposure_compensation:+.2f}, "
                   f"ExpTime={reference_fd.exposure_time})")

        target_ev = reference_fd.exposure_compensation
        corrected_files = []
        group_had_corrections = False

        for fd in group.files:
            if fd.filepath == reference_fd.filepath:
                corrections.append(ExposureCorrection(
                    filepath=fd.filepath,
                    original_ev=fd.exposure_compensation,
                    target_ev=target_ev,
                    brightness_adjustment=1.0,
                    highlights_adjustment=0,
                    shadows_adjustment=0,
                    success=True,
                    is_reference=True,
                    reason="Reference image (no correction needed)",
                ))
                corrected_files.append(fd)
                continue

            correction = calculate_correction_params(fd, reference_fd)

            if 0.95 < correction.brightness_adjustment < 1.05:
                corrections.append(ExposureCorrection(
                    filepath=fd.filepath,
                    original_ev=fd.exposure_compensation,
                    target_ev=target_ev,
                    brightness_adjustment=1.0,
                    highlights_adjustment=0,
                    shadows_adjustment=0,
                    success=True,
                    reason="No correction needed (exposure matches reference)",
                ))
                corrected_files.append(fd)
                total_corrected += 1
                logger.info(f"  {fd.filename}: exposure matches reference, keeping original")
                continue

            correction = apply_correction(correction, output_dir)
            corrections.append(correction)

            if correction.success:
                total_corrected += 1
                group_had_corrections = True
                corrected_fd = FileExifData(
                    filepath=correction.corrected_path,
                    exposure_compensation=target_ev,
                    timestamp=fd.timestamp,
                    exposure_time=fd.exposure_time,
                    f_number=fd.f_number,
                    iso=fd.iso,
                    model=fd.model,
                )
                corrected_files.append(corrected_fd)
                logger.info(f"  Corrected {fd.filename}: "
                           f"bright x{correction.brightness_adjustment:.2f}")
            else:
                total_failed += 1
                corrected_files.append(fd)
                logger.warning(f"  Correction failed for {fd.filename}, keeping original")

        new_group = BracketGroup(
            group_type=GroupType.AEB if len(corrected_files) > 1 else GroupType.SINGLE,
            files=corrected_files,
            group_id=group.group_id,
        )
        aligned_groups.append(new_group)

        if group_had_corrections:
            logger.info(f"  Group #{group.group_id}: {len(corrected_files)} files after correction")

    result = ExposureAlignResult(
        aligned_groups=aligned_groups,
        corrections=corrections,
        total_corrected=total_corrected,
        total_failed=total_failed,
    )

    logger.info(result.summary())
    return result
