"""
Module 04: Sky Recovery — DRC Highlight Recovery

After exposure alignment, some images may still have clipped highlights
(blown-out sky). This module applies Dynamic Range Compression (DRC)
via highlight tonemapping to recover detail.

DRC is attempted when:
  - Clipping ratio > 2% (significant clipped pixels)
  - Highlight detail ratio >= 0.1 (some structure remains to recover)

Returns:
  - drc_success: true if DRC was applied and improved the image
  - drc_applied: true if DRC was attempted
  - highlight_detail_ratio_after: metric for selector bonus
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)

DRC_CLIPPING_THRESHOLD = 0.02
DRC_HIGHLIGHT_DETAIL_MIN = 0.1
DRC_MAX_STRENGTH = 20.0


@dataclass
class DRCResult:
    filepath: Path
    output_path: Optional[Path]
    drc_applied: bool
    drc_success: bool
    clipping_before: float
    highlight_detail_before: float
    highlight_detail_after: float
    strength_used: float
    reason: str = ""

    @property
    def filename(self) -> str:
        return self.filepath.name


@dataclass
class SkyRecoveryResult:
    results: list[DRCResult]
    drc_applied_count: int = 0
    drc_success_count: int = 0
    drc_fail_count: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  SKY RECOVERY (DRC) SUMMARY",
            f"{'='*60}",
            f"  Images checked:    {len(self.results)}",
            f"  DRC applied:       {self.drc_applied_count}",
            f"  DRC successful:    {self.drc_success_count}",
            f"  DRC failed:        {self.drc_fail_count}",
        ]
        for r in self.results:
            status = "OK" if r.drc_applied and r.drc_success else \
                     "NO DRC NEEDED" if not r.drc_applied else \
                     "DRC FAILED"
            lines.append(f"  {r.filename}: [{status}] {r.reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def analyze_highlights(filepath: Path) -> tuple[float, float]:
    """
    Analyze highlight clipping in an image.

    Returns:
        (clipping_ratio, highlight_detail_ratio)
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

        total = luminance.size
        if total == 0:
            return 0.0, 0.0

        clipped_mask = luminance >= 250
        clipping_ratio = float(np.sum(clipped_mask)) / total

        bright_mask = luminance >= 230
        bright_pixels = luminance[bright_mask]

        if len(bright_pixels) > 50:
            highlight_variance = float(np.var(bright_pixels))
            highlight_detail = min(highlight_variance / 100.0, 1.0)
        else:
            highlight_detail = 0.0

        return clipping_ratio, highlight_detail

    except Exception as e:
        logger.warning(f"Highlight analysis error for {filepath.name}: {e}")
        return 0.0, 0.0


def apply_drc_jpeg(
    filepath: Path,
    output_path: Path,
    strength: float = 20.0,
) -> bool:
    """
    Apply DRC to a JPEG via tone curve manipulation.

    Compresses the top portion of the brightness range to recover
    highlight detail while preserving midtones and shadows.
    strength: 0-20, higher = more highlight compression.
    """
    try:
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img).astype(np.float32)

        factor = strength / 20.0

        highlight_threshold = 200
        shoulder = 255 - highlight_threshold
        compression = 1.0 - (factor * 0.4)

        mask = arr > highlight_threshold
        if not np.any(mask):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, "JPEG", quality=95)
            return True

        compressed = arr.copy()
        overflow = compressed[mask] - highlight_threshold
        compressed[mask] = highlight_threshold + overflow * compression

        compressed = np.clip(compressed, 0, 255).astype(np.uint8)

        out = Image.fromarray(compressed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path, "JPEG", quality=95)
        return True

    except Exception as e:
        logger.error(f"JPEG DRC error for {filepath.name}: {e}")
        return False


def apply_drc_raw(
    filepath: Path,
    output_path: Path,
    strength: float = 20.0,
) -> bool:
    """
    Apply DRC to a RAW file using rawpy highlight recovery.

    strength: 0-20, mapped to rawpy's bright parameter.
    """
    if not HAS_RAWPY:
        logger.warning(f"rawpy not available, falling back to JPEG DRC for {filepath.name}")
        return apply_drc_jpeg(filepath, output_path, strength)

    try:
        with rawpy.imread(str(filepath)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
                highlight_mode=2,
                bright=strength / 100.0,
            )

        img = Image.fromarray(rgb)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=95)
        return True

    except Exception as e:
        logger.error(f"RAW DRC error for {filepath.name}: {e}")
        return False


def try_drc_recovery(
    filepath: Path,
    output_dir: Path,
    max_strength: float = DRC_MAX_STRENGTH,
) -> DRCResult:
    """
    Try to recover blown-out highlights using DRC.

    Strategy:
    1. Analyze highlight clipping
    2. If clipping > threshold AND highlight detail exists → apply DRC
    3. If no clipping → no DRC needed
    4. If clipping but no detail → DRC won't help

    Returns:
        DRCResult with outcome
    """
    clipping_ratio, detail_before = analyze_highlights(filepath)

    if clipping_ratio < DRC_CLIPPING_THRESHOLD:
        return DRCResult(
            filepath=filepath,
            output_path=filepath,
            drc_applied=False,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            highlight_detail_after=detail_before,
            strength_used=0,
            reason=f"No DRC needed ({clipping_ratio*100:.1f}% clipped)",
        )

    if detail_before < DRC_HIGHLIGHT_DETAIL_MIN:
        return DRCResult(
            filepath=filepath,
            output_path=filepath,
            drc_applied=False,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            highlight_detail_after=detail_before,
            strength_used=0,
            reason=f"DRC skipped: no detail in highlights ({detail_before:.3f})",
        )

    stem = filepath.stem
    suffix = filepath.suffix.lower()
    is_raw = suffix in {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}

    drc_path = output_dir / f"{stem}_drc.jpg"

    if is_raw and HAS_RAWPY:
        success = apply_drc_raw(filepath, drc_path, max_strength)
    else:
        success = apply_drc_jpeg(filepath, drc_path, max_strength)

    if not success:
        return DRCResult(
            filepath=filepath,
            output_path=filepath,
            drc_applied=True,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            highlight_detail_after=detail_before,
            strength_used=max_strength,
            reason=f"DRC failed (technical error)",
        )

    _, detail_after = analyze_highlights(drc_path)

    if detail_after <= detail_before:
        return DRCResult(
            filepath=filepath,
            output_path=filepath,
            drc_applied=True,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            highlight_detail_after=detail_after,
            strength_used=max_strength,
            reason=f"DRC unsuccessful: detail didn't improve ({detail_before:.3f}→{detail_after:.3f})",
        )

    return DRCResult(
        filepath=filepath,
        output_path=drc_path,
        drc_applied=True,
        drc_success=True,
        clipping_before=clipping_ratio,
        highlight_detail_before=detail_before,
        highlight_detail_after=detail_after,
        strength_used=max_strength,
        reason=f"DRC OK: detail {detail_before:.3f}→{detail_after:.3f}",
    )


def recover_highlights(
    groups: list[BracketGroup],
    output_dir: Path,
    max_strength: float = DRC_MAX_STRENGTH,
) -> SkyRecoveryResult:
    """
    Apply DRC sky recovery to all images across all groups.

    For each image:
    1. Analyze highlights
    2. If clipping > threshold → try DRC
    3. Update the group's file paths based on outcome

    Returns:
        SkyRecoveryResult with per-file DRC outcomes
    """
    results = []
    drc_applied = 0
    drc_success = 0
    drc_fail = 0

    for group in groups:
        updated_files = []
        for fd in group.files:
            result = try_drc_recovery(fd.filepath, output_dir, max_strength)
            results.append(result)

            if result.drc_applied:
                drc_applied += 1
            if result.drc_success:
                drc_success += 1
            elif result.drc_applied and not result.drc_success:
                drc_fail += 1

            new_fd = FileExifData(
                filepath=result.output_path or fd.filepath,
                exposure_compensation=fd.exposure_compensation,
                timestamp=fd.timestamp,
                exposure_time=fd.exposure_time,
                f_number=fd.f_number,
                iso=fd.iso,
                model=fd.model,
                drc_applied=result.drc_applied,
                drc_success=result.drc_success,
            )
            updated_files.append(new_fd)

        group.files = updated_files

    return SkyRecoveryResult(
        results=results,
        drc_applied_count=drc_applied,
        drc_success_count=drc_success,
        drc_fail_count=drc_fail,
    )
