"""
Module 04: Sky Recovery — DRC via pp3 Sidecar

After exposure alignment, some images may still have clipped highlights
(blown-out sky). This module analyzes the RAW directly (via rawpy) and
writes DRC parameters into a pp3 sidecar instead of rendering a JPEG.

DRC is attempted when:
  - Clipping ratio > 2% (significant clipped pixels)
  - Highlight detail ratio >= 0.1 (some structure remains to recover)

Instead of creating _drc.jpg files (which degrade sharpness), we write
a _drc.pp3 sidecar. The original CR2 stays unmodified.
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

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}


@dataclass
class DRCResult:
    filepath: Path
    pp3_path: Optional[Path]
    drc_applied: bool
    drc_success: bool
    clipping_before: float
    highlight_detail_before: float
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


def load_image_for_analysis(filepath: Path) -> np.ndarray:
    """
    Load an image as a numpy array for highlight analysis.

    For RAW files: uses rawpy for full-quality rendering.
    For JPEG/etc: uses PIL.
    """
    ext = filepath.suffix.lower()
    if ext in RAW_EXTENSIONS and HAS_RAWPY:
        try:
            with rawpy.imread(str(filepath)) as raw:
                return raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=False,
                )
        except Exception as e:
            logger.warning(f"rawpy failed for {filepath.name}: {e}")

    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def analyze_highlights(filepath: Path) -> tuple[float, float]:
    """
    Analyze highlight clipping in an image.

    For RAW files, uses rawpy for accurate highlight analysis.

    Returns:
        (clipping_ratio, highlight_detail_ratio)
    """
    try:
        arr = load_image_for_analysis(filepath)

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


def write_drc_pp3(
    filepath: Path,
    output_path: Path,
    strength: float,
) -> bool:
    """
    Write a .pp3 sidecar with DRC/highlight recovery parameters.

    Writes Highlight Compression + HLRecovery params.
    Original CR2 stays untouched — applied later during RAW development.

    Args:
        filepath: Original file path (for context only)
        output_path: Output .pp3 file path
        strength: DRC strength 0-20, mapped to HighlightCompr
    """
    try:
        highlight_compr = int(strength * 5)
        highlight_compr = max(0, min(100, highlight_compr))

        pp3_content = f"""[Version]
Version=1

[Exposure]
HighlightCompr={highlight_compr}
ShadowCompr=0

[HLRecovery]
Enabled=1
Method=1
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pp3_content)

        return True

    except Exception as e:
        logger.error(f"DRC pp3 write error for {filepath.name}: {e}")
        return False


def try_drc_recovery(
    filepath: Path,
    output_dir: Path,
    max_strength: float = DRC_MAX_STRENGTH,
) -> DRCResult:
    """
    Try to recover blown-out highlights using DRC.

    Strategy:
    1. Analyze highlight clipping on RAW (via rawpy)
    2. If clipping > threshold AND highlight detail exists → write DRC pp3
    3. If no clipping → no DRC needed
    4. If clipping but no detail → DRC won't help

    Returns:
        DRCResult with outcome
    """
    clipping_ratio, detail_before = analyze_highlights(filepath)

    if clipping_ratio < DRC_CLIPPING_THRESHOLD:
        return DRCResult(
            filepath=filepath,
            pp3_path=None,
            drc_applied=False,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            strength_used=0,
            reason=f"No DRC needed ({clipping_ratio*100:.1f}% clipped)",
        )

    if detail_before < DRC_HIGHLIGHT_DETAIL_MIN:
        return DRCResult(
            filepath=filepath,
            pp3_path=None,
            drc_applied=False,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            strength_used=0,
            reason=f"DRC skipped: no detail in highlights ({detail_before:.3f})",
        )

    stem = filepath.stem
    pp3_path = output_dir / f"{stem}_drc.pp3"

    success = write_drc_pp3(filepath, pp3_path, max_strength)

    if not success:
        return DRCResult(
            filepath=filepath,
            pp3_path=None,
            drc_applied=True,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            strength_used=max_strength,
            reason=f"DRC failed (technical error)",
        )

    # Re-analyze after DRC to confirm improvement
    _, detail_after = analyze_highlights(filepath)

    if detail_after <= detail_before:
        return DRCResult(
            filepath=filepath,
            pp3_path=pp3_path,
            drc_applied=True,
            drc_success=False,
            clipping_before=clipping_ratio,
            highlight_detail_before=detail_before,
            strength_used=max_strength,
            reason=f"DRC unsuccessful: detail won't improve ({detail_before:.3f})",
        )

    return DRCResult(
        filepath=filepath,
        pp3_path=pp3_path,
        drc_applied=True,
        drc_success=True,
        clipping_before=clipping_ratio,
        highlight_detail_before=detail_before,
        strength_used=max_strength,
        reason=f"DRC OK via pp3: detail {detail_before:.3f} "
               f"(HighlightCompr in pp3)",
    )


def recover_highlights(
    groups: list[BracketGroup],
    output_dir: Path,
    max_strength: float = DRC_MAX_STRENGTH,
) -> SkyRecoveryResult:
    """
    Apply DRC sky recovery to all images across all groups.

    For each image:
    1. Analyze highlights (on CR2/RAW directly via rawpy)
    2. If clipping > threshold → write DRC pp3 sidecar
    3. FileExifData.filepath stays on original CR2
    4. DRC status flags set on FileExifData for the selector

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

            # Keep original CR2 filepath — only update DRC flags
            fd.drc_applied = result.drc_applied
            fd.drc_success = result.drc_success
            updated_files.append(fd)

        group.files = updated_files

    return SkyRecoveryResult(
        results=results,
        drc_applied_count=drc_applied,
        drc_success_count=drc_success,
        drc_fail_count=drc_fail,
    )
