"""
Module 04: Exposure Aligner (v2 — Reference-Based, pp3 Sidecar)

Aligns all AEB images to a reference exposure level using pp3 sidecars
instead of rendered JPEGs.

Reference selection:
- <3 images: best histogram (closest to mid-gray)

All images except the reference get a {stem}_exposure_corrected.pp3 sidecar
with the exposure compensation. The original CR2 is never modified.
This preserves native RAW quality for metrics computation.

For Burst groups and Singles: no correction needed (same exposure).

Input:  List of BracketGroup objects (after overexposure check)
Output: Same groups with pp3 sidecars + exposure_corrected flags
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)


@dataclass
class ExposureCorrection:
    """Correction parameters for a single image."""
    filepath: Path
    original_ev: float
    target_ev: float
    ev_diff: float
    pp3_path: Optional[Path] = None
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
            lines.append("  pp3 sidecars created:")
            for c in self.corrections:
                if c.is_reference:
                    lines.append(f"    {c.filename}: REFERENCE (no correction)")
                    continue
                status = "OK" if c.success else "FAILED"
                lines.append(f"    {c.filename}: EV {c.original_ev:+.2f} → {c.target_ev:+.2f} "
                           f"(Δ{c.ev_diff:+.2f}EV) [{status}]")
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


def _parse_exposure_time(exp_value) -> float:
    """Parse exposure time to float seconds."""
    if isinstance(exp_value, (int, float)):
        return float(exp_value)
    exp_str = str(exp_value)
    if "/" in exp_str:
        parts = exp_str.split("/")
        return float(parts[0]) / float(parts[1])
    return float(exp_str)


def write_exposure_pp3(
    filepath: Path,
    output_path: Path,
    ev_diff: float,
) -> bool:
    """
    Write a .pp3 sidecar file with exposure compensation.

    Instead of rendering a JPEG (which degrades sharpness), we write
    a RawTherapee pp3 sidecar that stores the exposure adjustment.
    The original CR2 stays untouched — metrics are computed on RAW.

    Args:
        filepath: Original RAW file path (unused, only for context)
        output_path: Output .pp3 file path
        ev_diff: Exposure compensation in EV to apply
    """
    try:
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        expcomp = max(-5.0, min(5.0, ev_diff))

        pp3_content = f"""[Version]
Version=1

[Exposure]
Compensation={expcomp:.3f}
Contrast=0
Saturation=0
Brightness=0
Black=0
HighlightCompr=0
ShadowCompr=0
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pp3_content)

        return True

    except Exception as e:
        logger.error(f"pp3 write error for {filepath.name}: {e}")
        return False


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

        brightness_score = max(0, 100 - abs(mean_val - 128) * 0.8)

        clipped = float(np.sum(luminance >= 250)) / luminance.size
        clipping_penalty = clipped * 100

        std_val = float(np.std(luminance))
        dr_bonus = min(std_val * 0.5, 15)

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
    brightness_offset: float = 0.0,
) -> ExposureCorrection:
    """
    Calculate correction parameters to align an image to reference exposure.

    Uses actual ExposureTime ratio to compute EV difference, then
    applies a user-defined brightness_offset to shift the target.

    Args:
        file_data: EXIF data of the image to correct
        reference_fd: EXIF data of the reference image
        brightness_offset: Additional EV to add (user preference).
            Positive = brighter target. Derived from user_preferences.py.

    Returns:
        ExposureCorrection with calculated EV diff
    """
    t_ref = _parse_exposure_time(reference_fd.exposure_time)
    t_cur = _parse_exposure_time(file_data.exposure_time)

    if t_cur <= 0 or t_ref <= 0:
        ev_diff = 0.0
    else:
        ev_diff = math.log2(t_ref / t_cur)

    ev_diff += brightness_offset

    target_ev = reference_fd.exposure_compensation

    return ExposureCorrection(
        filepath=file_data.filepath,
        original_ev=file_data.exposure_compensation,
        target_ev=target_ev,
        ev_diff=ev_diff,
    )


def apply_correction(
    correction: ExposureCorrection,
    output_dir: Path = None,
) -> ExposureCorrection:
    """
    Apply exposure correction by writing a pp3 sidecar.

    Instead of rendering a JPEG (which degrades quality), writes
    a RawTherapee pp3 sidecar with the exposure compensation.

    Args:
        correction: Calculated correction parameters
        output_dir: Directory for pp3 sidecar files

    Returns:
        Updated ExposureCorrection with result
    """
    if output_dir is None:
        output_dir = correction.filepath.parent

    # Skip if EV diff is negligible (within 1/6 stop)
    if abs(correction.ev_diff) < 0.167:
        correction.success = True
        correction.reason = f"Negligible EV diff ({correction.ev_diff:+.3f}), no pp3 needed"
        return correction

    stem = correction.filepath.stem
    pp3_name = f"{stem}_exposure_corrected.pp3"
    pp3_path = output_dir / pp3_name
    correction.pp3_path = pp3_path

    success = write_exposure_pp3(
        correction.filepath,
        pp3_path,
        correction.ev_diff,
    )

    correction.success = success
    if success:
        correction.reason = (f"pp3 sidecar: Δ{correction.ev_diff:+.2f}EV "
                           f"(ExpComp={correction.ev_diff:+.3f})")
    else:
        correction.reason = "pp3 write failed"

    return correction


def align_exposures(
    groups: list[BracketGroup],
    output_dir: Path = None,
    brightness_offset: float = 0.0,
) -> ExposureAlignResult:
    """
    Align exposures for all AEB groups to a reference image.

    For each AEB group:
    1. Select reference image (best toncurve)
    2. Write pp3 sidecars for ALL other images
    3. FileExifData.filepath stays on the original CR2
    4. exposure_corrected flag + pp3_path + ev_diff set on FileExifData

    Args:
        groups: Bracket groups to process
        output_dir: Directory for pp3 sidecar files
        brightness_offset: User preference EV offset (positive = brighter).
            Applied to ALL corrections. See user_preferences.py.

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

        for fd in group.files:
            if fd.filepath == reference_fd.filepath:
                # Reference image — only brightness_offset applies
                ev_diff = brightness_offset
                ref_correction = ExposureCorrection(
                    filepath=fd.filepath,
                    original_ev=fd.exposure_compensation,
                    target_ev=target_ev,
                    ev_diff=ev_diff,
                    success=True,
                    is_reference=True,
                    reason=f"Reference image (offset={ev_diff:+.3f}EV)",
                )
                fd.ev_diff = ev_diff
                corrections.append(ref_correction)
                corrected_files.append(fd)
                continue

            correction = calculate_correction_params(fd, reference_fd, brightness_offset)

            # Check if correction is needed
            if abs(correction.ev_diff) < 0.167:
                noop = ExposureCorrection(
                    filepath=fd.filepath,
                    original_ev=fd.exposure_compensation,
                    target_ev=target_ev,
                    ev_diff=correction.ev_diff,
                    success=True,
                    reason=f"Exposure matches reference (Δ{correction.ev_diff:+.3f}EV)",
                )
                fd.ev_diff = correction.ev_diff
                corrections.append(noop)
                corrected_files.append(fd)
                total_corrected += 1
                logger.info(f"  {fd.filename}: exposure matches reference, keeping original")
                continue

            correction = apply_correction(correction, output_dir)
            fd.ev_diff = correction.ev_diff
            corrections.append(correction)

            if correction.success:
                total_corrected += 1
                # Keep original CR2 filepath — only add pp3 sidecar reference
                fd.exposure_corrected = True
                fd.exposure_corrected_pp3 = correction.pp3_path
                corrected_files.append(fd)
                logger.info(f"  {fd.filename}: pp3 sidecar → Δ{correction.ev_diff:+.2f}EV")
            else:
                total_failed += 1
                corrected_files.append(fd)
                logger.warning(f"  pp3 failed for {fd.filename}, keeping original")

        new_group = BracketGroup(
            group_type=GroupType.AEB if len(corrected_files) > 1 else GroupType.SINGLE,
            files=corrected_files,
            group_id=group.group_id,
        )
        aligned_groups.append(new_group)

        # Log if any corrections were applied
        n_corrected = sum(1 for c in corrections[-len(group.files):] if c.success and not c.is_reference)
        if n_corrected:
            logger.info(f"  Group #{group.group_id}: {n_corrected} pp3 sidecar(s) written")

    result = ExposureAlignResult(
        aligned_groups=aligned_groups,
        corrections=corrections,
        total_corrected=total_corrected,
        total_failed=total_failed,
    )

    logger.info(result.summary())
    return result
