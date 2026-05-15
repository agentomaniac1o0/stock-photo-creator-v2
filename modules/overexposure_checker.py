"""
Module 03: Overexposure Checker

Checks the brightest image in each AEB group for unrecoverable clipping.
If highlights are blown beyond correction, the image is marked for deletion.

For AEB groups:
  - Check the +EV (brightest) image for clipping
  - If unrecoverable → mark for deletion
  - Remaining images stay for further processing

For Burst groups and Singles:
  - All images are passed through (quality check happens in Module 05)

Input:  List of BracketGroup objects
Output: Filtered groups + list of rejected files with reasons
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config.settings import OVEREXPOSURE_CLIPPING_THRESHOLD
from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)


@dataclass
class ClippingResult:
    """Result of clipping analysis for a single image."""
    filepath: Path
    is_clipped: bool
    clipping_ratio: float
    highlight_detail_ratio: float
    recoverable: bool
    reason: str = ""

    @property
    def filename(self) -> str:
        return self.filepath.name


@dataclass
class OverexposureCheckResult:
    """Result of the overexposure check for all groups."""
    checked_groups: list[BracketGroup]
    rejected_files: list[tuple[Path, str]]  # (filepath, reason)
    total_checked: int = 0
    total_rejected: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  OVEREXPOSURE CHECK SUMMARY",
            f"{'='*60}",
            f"  Images checked:  {self.total_checked}",
            f"  Images rejected: {self.total_rejected}",
            f"  Images kept:     {self.total_checked - self.total_rejected}",
        ]
        if self.rejected_files:
            lines.append("")
            lines.append("  Rejected files:")
            for filepath, reason in self.rejected_files:
                lines.append(f"    {filepath.name}: {reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def analyze_clipping(filepath: Path) -> ClippingResult:
    """
    Analyze an image for unrecoverable clipping.

    Strategy:
    1. Open the image (JPEG preview from RAW or the file itself)
    2. Calculate the ratio of pixels at or near maximum brightness
    3. Check if there's any detail left in highlights (gradient analysis)
    4. Determine if clipping is recoverable

    Returns:
        ClippingResult with analysis details
    """
    try:
        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)

        # Convert to luminance (perceived brightness)
        # Using standard Rec. 709 coefficients
        if arr.ndim == 3:
            luminance = (0.2126 * arr[:, :, 0] +
                        0.7152 * arr[:, :, 1] +
                        0.0722 * arr[:, :, 2])
        else:
            luminance = arr.astype(float)

        total_pixels = luminance.size

        # Count clipped pixels (at or near maximum: 250-255)
        clipped_mask = luminance >= 250
        clipped_count = int(np.sum(clipped_mask))
        clipping_ratio = clipped_count / total_pixels

        # Check for highlight detail: analyze the gradient in bright areas
        # If there's structure/detail in highlights, it might be recoverable
        bright_mask = luminance >= 230
        bright_pixels = luminance[bright_mask]

        if len(bright_pixels) > 100:
            # Calculate local variance in bright areas
            # Low variance = flat white (no detail) = unrecoverable
            # Higher variance = some detail left = potentially recoverable
            highlight_variance = float(np.var(bright_pixels))
            highlight_detail_ratio = min(highlight_variance / 100.0, 1.0)
        else:
            highlight_detail_ratio = 0.0

        # Determine if clipping is recoverable
        # Criteria:
        # 1. If clipping ratio > threshold → unrecoverable
        # 2. If highlight detail is very low AND clipping is significant → unrecoverable
        is_clipped = clipping_ratio > OVEREXPOSURE_CLIPPING_THRESHOLD

        if is_clipped and highlight_detail_ratio < 0.1:
            recoverable = False
            reason = (f"Unrecoverable clipping: {clipping_ratio*100:.1f}% of pixels "
                     f"clipped, no highlight detail (detail_ratio={highlight_detail_ratio:.3f})")
        elif is_clipped:
            recoverable = True
            reason = (f"Clipping detected ({clipping_ratio*100:.1f}%) but some "
                     f"highlight detail remains (detail_ratio={highlight_detail_ratio:.3f})")
        else:
            recoverable = True
            reason = f"No significant clipping ({clipping_ratio*100:.2f}% clipped)"

        return ClippingResult(
            filepath=filepath,
            is_clipped=is_clipped,
            clipping_ratio=clipping_ratio,
            highlight_detail_ratio=highlight_detail_ratio,
            recoverable=recoverable,
            reason=reason,
        )

    except Exception as e:
        logger.error(f"Clipping analysis error for {filepath.name}: {e}")
        return ClippingResult(
            filepath=filepath,
            is_clipped=False,
            clipping_ratio=0.0,
            highlight_detail_ratio=0.0,
            recoverable=True,
            reason=f"Analysis error: {e}",
        )


def _parse_exposure_time(et_str: str) -> float:
    """Parse exposure time string like '1/200' or '0.5' into seconds."""
    if not et_str:
        return 0.0
    et_str = str(et_str).strip()
    if '/' in et_str:
        parts = et_str.split('/')
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(et_str)
    except ValueError:
        return 0.0


def check_overexposure(groups: list[BracketGroup]) -> OverexposureCheckResult:
    """
    Check all AEB groups for overexposure.

    For each AEB group:
    1. Find the brightest image (highest EV OR longest exposure time)
    2. Analyze it for clipping
    3. If unrecoverable → remove it from the group
    4. If the group becomes empty → mark all original files as rejected

    Burst groups and Singles are passed through unchanged.
    """
    checked_groups = []
    rejected_files: list[tuple[Path, str]] = []
    total_checked = 0
    total_rejected = 0

    for group in groups:
        if group.group_type != GroupType.AEB:
            # Non-AEB groups pass through (quality check in Module 05)
            checked_groups.append(group)
            continue

        # Sort by brightness: highest EV first, or longest exposure time
        def brightness_key(f):
            ev = f.exposure_compensation
            et = _parse_exposure_time(f.exposure_time)
            # Combine: higher EV = brighter, longer exposure = brighter
            return (ev, et)

        sorted_files = sorted(group.files, key=brightness_key, reverse=True)

        brightest = sorted_files[0]
        total_checked += 1
        result = analyze_clipping(brightest.filepath)

        logger.info(f"AEB group #{group.group_id}: checking brightest image "
                   f"({brightest.filename}, EV={brightest.exposure_compensation:+.2f}, "
                   f"ExpTime={brightest.exposure_time})")
        logger.info(f"  {result.filename}: {result.reason}")

        if not result.recoverable:
            # Unrecoverable → reject this image
            rejected_files.append((brightest.filepath, result.reason))
            total_rejected += 1

            # Remove from group
            remaining = [f for f in sorted_files[1:]]

            if not remaining:
                # All images in this group rejected
                logger.warning(f"  All images in AEB group #{group.group_id} rejected")
                rejected_files.extend([
                    (f.filepath, "All images in group rejected (brightest was unrecoverable)")
                    for f in sorted_files[1:]
                ])
                total_rejected += len(sorted_files) - 1
                continue

            # Create new group with remaining images
            # Re-classify: if only 1 image left → Single, else check EV variation
            if len(remaining) == 1:
                new_type = GroupType.SINGLE
            else:
                ev_values = [f.exposure_compensation for f in remaining]
                if max(ev_values) - min(ev_values) > 0.2:
                    new_type = GroupType.AEB
                else:
                    new_type = GroupType.BURST

            new_group = BracketGroup(
                group_type=new_type,
                files=remaining,
                group_id=group.group_id
            )
            checked_groups.append(new_group)
            logger.info(f"  Brightest image rejected. Group now: {new_type.value}, "
                       f"{len(remaining)} file(s)")
        else:
            # Image is OK (or recoverable) → keep the group as-is
            checked_groups.append(group)

    result = OverexposureCheckResult(
        checked_groups=checked_groups,
        rejected_files=rejected_files,
        total_checked=total_checked,
        total_rejected=total_rejected,
    )

    logger.info(result.summary())
    return result
