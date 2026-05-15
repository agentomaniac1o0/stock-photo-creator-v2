"""
Module 04: Exposure Aligner

For AEB groups where the brightest image was rejected (unrecoverable clipping),
the remaining darker images are corrected to match proper exposure.

Correction strategy:
- Lift shadows/highlights in underexposed images
- Adjust brightness to match the target exposure level
- Uses rawpy for RAW files, Pillow for JPEGs

For Burst groups and Singles: no correction needed (same exposure).

Input:  List of BracketGroup objects (after overexposure check)
Output: Same groups with corrected images (in-place or new files)
"""
import logging
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
                status = "OK" if c.success else "FAILED"
                lines.append(f"    {c.filename}: EV {c.original_ev:+.1f} → {c.target_ev:+.1f} "
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
        return 128.0  # Default mid-gray


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


def calculate_correction_params(
    file_data: FileExifData,
    target_ev: float = 0.0,
) -> ExposureCorrection:
    """
    Calculate correction parameters to align an image to target exposure.

    Args:
        file_data: EXIF data of the image to correct
        target_ev: Target exposure value (usually 0.0 = middle exposure)

    Returns:
        ExposureCorrection with calculated parameters
    """
    ev_diff = target_ev - file_data.exposure_compensation

    # Brightness adjustment: each EV stop = 2x brightness
    brightness_factor = 2.0 ** ev_diff

    # Highlights/shadows adjustments based on EV difference
    # Negative EV (underexposed) → lift shadows and highlights
    if ev_diff > 0:
        # Image is darker than target → brighten
        highlights_adjustment = min(ev_diff * 30, 100)  # Up to +100
        shadows_adjustment = min(ev_diff * 50, 100)     # Up to +100
    else:
        # Image is brighter than target → darken (rare case)
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

    # Create corrected filename
    stem = correction.filepath.stem
    suffix = correction.filepath.suffix
    corrected_name = f"{stem}_exposure_corrected.jpg"
    corrected_path = output_dir / corrected_name
    correction.corrected_path = corrected_path

    # Determine if RAW or JPEG
    is_raw = correction.filepath.suffix.lower() in {
        ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"
    }

    if is_raw:
        # Try rawpy first, fallback to Pillow
        ev_diff = correction.target_ev - correction.original_ev
        success = correct_exposure_rawpy(
            correction.filepath,
            corrected_path,
            brightness_adjustment=ev_diff,
        )
        if not success:
            success = correct_exposure_pillow(
                correction.filepath,
                corrected_path,
                brightness_factor=correction.brightness_adjustment,
            )
    else:
        # JPEG → use Pillow
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
    Align exposures for AEB groups where the brightest image was rejected.

    For each AEB group:
    1. Find the darkest image (lowest EV)
    2. Calculate correction to bring it closer to target EV (0.0)
    3. Apply correction
    4. Update the group with corrected file path

    Burst groups and Singles are passed through unchanged.
    """
    aligned_groups = []
    corrections = []
    total_corrected = 0
    total_failed = 0

    for group in groups:
        if group.group_type != GroupType.AEB:
            # Non-AEB groups pass through unchanged
            aligned_groups.append(group)
            continue

        if len(group.files) < 1:
            continue

        # Sort by EV (darkest first)
        sorted_files = sorted(
            group.files,
            key=lambda f: f.exposure_compensation,
        )

        # Target: middle exposure (EV=0)
        target_ev = 0.0

        logger.info(f"AEB group #{group.group_id}: aligning exposures to EV={target_ev:+.1f}")

        corrected_files = []
        group_had_corrections = False

        for fd in sorted_files:
            # Only correct images that are significantly underexposed
            if fd.exposure_compensation < -0.5:
                correction = calculate_correction_params(fd, target_ev)
                correction = apply_correction(correction, output_dir)
                corrections.append(correction)

                if correction.success:
                    total_corrected += 1
                    group_had_corrections = True
                    # Create new FileExifData with corrected path
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
                              f"EV {fd.exposure_compensation:+.1f} → {target_ev:+.1f}")
                else:
                    total_failed += 1
                    # Keep original file even if correction failed
                    corrected_files.append(fd)
                    logger.warning(f"  Correction failed for {fd.filename}, keeping original")
            else:
                # Image is close enough to target → keep as-is
                corrected_files.append(fd)

        # Create updated group
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
