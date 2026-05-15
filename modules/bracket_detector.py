"""
Module 02: Bracket Detector

Groups RAW files into:
- AEB groups (different exposure compensation, same scene)
- Burst groups (same exposure compensation, same scene, rapid fire)
- Single files (no grouping possible)

Uses EXIF ExposureCompensation + timestamp (0-3s window).

Input:  List of local RAW file paths
Output: List of Group objects with type, files, and EXIF data
"""
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from config.settings import BRACKET_TIME_TOLERANCE_SEC, BRACKET_EV_DIFF_THRESHOLD

logger = logging.getLogger(__name__)


class GroupType(Enum):
    AEB = "aeb"           # Different EV values, same scene
    BURST = "burst"       # Same EV values, same scene, rapid fire
    SINGLE = "single"     # No grouping possible


@dataclass
class FileExifData:
    """Parsed EXIF data for a single file."""
    filepath: Path
    exposure_compensation: float = 0.0
    timestamp: Optional[datetime] = None
    exposure_time: Optional[str] = None
    f_number: Optional[str] = None
    iso: Optional[str] = None
    model: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.filepath.name


@dataclass
class BracketGroup:
    """A group of related images."""
    group_type: GroupType
    files: list[FileExifData]
    group_id: int = 0

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def has_exposure_variation(self) -> bool:
        """True if files in this group have different exposure values."""
        if len(self.files) < 2:
            return False
        # First try ExposureCompensation
        ev_values = [f.exposure_compensation for f in self.files]
        ev_range = max(ev_values) - min(ev_values)
        if ev_range > BRACKET_EV_DIFF_THRESHOLD:
            return True
        # Fallback: compare ExposureTime strings
        exp_times = [f.exposure_time for f in self.files if f.exposure_time]
        if len(exp_times) >= 2:
            unique_times = set(str(t) for t in exp_times)
            if len(unique_times) > 1:
                return True
        return False

    @property
    def best_exposure_file(self) -> Optional[FileExifData]:
        """File closest to middle exposure (lowest absolute EV, or middle ExposureTime)."""
        if not self.files:
            return None
        # Try EV first
        ev_values = [f.exposure_compensation for f in self.files]
        if max(ev_values) - min(ev_values) > BRACKET_EV_DIFF_THRESHOLD:
            return min(self.files, key=lambda f: abs(f.exposure_compensation))
        # Fallback: middle ExposureTime
        def exposure_time_sort_key(f):
            et = f.exposure_time
            if not et:
                return 0.0
            et_str = str(et)
            if '/' in et_str:
                try:
                    parts = et_str.split('/')
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    return 0.0
            try:
                return float(et_str)
            except ValueError:
                return 0.0
        times = [exposure_time_sort_key(f) for f in self.files]
        median_time = sorted(times)[len(times) // 2]
        return min(self.files, key=lambda f: abs(exposure_time_sort_key(f) - median_time))

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "file_count": self.file_count,
            "has_exposure_variation": self.has_exposure_variation,
            "files": [f.filename for f in self.files],
            "ev_values": [f.exposure_compensation for f in self.files],
        }


def read_exif_fields(filepath: Path) -> dict:
    """Read relevant EXIF fields from a RAW/JPEG file."""
    try:
        result = subprocess.run(
            ["exiftool", "-json",
             "-ExposureCompensation",
             "-DateTimeOriginal",
             "-ExposureTime",
             "-FNumber",
             "-ISO",
             "-Model",
             str(filepath)],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        if data and isinstance(data, list):
            return data[0]
        return {}
    except Exception as e:
        logger.error(f"EXIF read error for {filepath.name}: {e}")
        return {}


def _parse_fraction(value: str) -> float:
    """Parse a fraction string like '-1/3' or '+2' into a float."""
    if not value:
        return 0.0
    value = str(value).strip()
    if '/' in value:
        parts = value.split('/')
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_exif_data(filepath: Path) -> FileExifData:
    """Parse EXIF data into a FileExifData object."""
    exif = read_exif_fields(filepath)

    ev_str = exif.get("ExposureCompensation", "")
    ev_val = _parse_fraction(ev_str)

    dt_str = exif.get("DateTimeOriginal", "")
    try:
        timestamp = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S") if dt_str else None
    except ValueError:
        timestamp = None

    return FileExifData(
        filepath=filepath,
        exposure_compensation=ev_val,
        timestamp=timestamp,
        exposure_time=exif.get("ExposureTime", ""),
        f_number=exif.get("FNumber", ""),
        iso=exif.get("ISO", ""),
        model=exif.get("Model", ""),
    )


def detect_brackets(filepaths: list[Path]) -> list[BracketGroup]:
    """
    Group files into AEB, Burst, or Single groups.

    Algorithm:
    1. Read EXIF for all files
    2. Sort by timestamp
    3. Group files within time window (0-3s)
    4. Classify each group as AEB (different EV) or Burst (same EV)
    """
    if not filepaths:
        return []

    # Parse EXIF for all files
    file_data = []
    for fp in filepaths:
        fd = parse_exif_data(fp)
        file_data.append(fd)
        logger.debug(f"  {fd.filename}: EV={fd.exposure_compensation:+.1f}, "
                     f"time={fd.timestamp or 'unknown'}")

    # Sort by timestamp (files without timestamp go to end)
    with_time = sorted(
        [f for f in file_data if f.timestamp is not None],
        key=lambda f: f.timestamp
    )
    without_time = [f for f in file_data if f.timestamp is None]

    groups = []
    used = set()
    group_id = 1

    # Group files with timestamps
    for i, fd in enumerate(with_time):
        if id(fd) in used:
            continue

        group = [fd]
        used.add(id(fd))

        # Look ahead for files within time window
        for j in range(i + 1, len(with_time)):
            fdj = with_time[j]
            if id(fdj) in used:
                continue

            time_diff = abs((fdj.timestamp - fd.timestamp).total_seconds())
            if time_diff > BRACKET_TIME_TOLERANCE_SEC:
                break

            group.append(fdj)
            used.add(id(fdj))

        # Classify group
        if len(group) == 1:
            groups.append(BracketGroup(
                group_type=GroupType.SINGLE,
                files=group,
                group_id=group_id
            ))
        else:
            # Check if exposure values differ (EV or ExposureTime)
            ev_values = [f.exposure_compensation for f in group]
            ev_range = max(ev_values) - min(ev_values)

            # Also check ExposureTime
            exp_times = [f.exposure_time for f in group if f.exposure_time]
            unique_times = set(str(t) for t in exp_times) if exp_times else set()

            if ev_range > BRACKET_EV_DIFF_THRESHOLD or len(unique_times) > 1:
                group_type = GroupType.AEB
                time_info = f", {len(unique_times)} different exposure times" if len(unique_times) > 1 else ""
                logger.info(f"  AEB group #{group_id}: {len(group)} files, "
                          f"EV range: {min(ev_values):+.2f} to {max(ev_values):+.2f}{time_info}")
            else:
                group_type = GroupType.BURST
                logger.info(f"  Burst group #{group_id}: {len(group)} files, "
                          f"EV: {ev_values[0]:+.2f}")

            groups.append(BracketGroup(
                group_type=group_type,
                files=group,
                group_id=group_id
            ))

        group_id += 1

    # Files without timestamps are singles
    for fd in without_time:
        groups.append(BracketGroup(
            group_type=GroupType.SINGLE,
            files=[fd],
            group_id=group_id
        ))
        group_id += 1

    # Summary
    aeb_count = sum(1 for g in groups if g.group_type == GroupType.AEB)
    burst_count = sum(1 for g in groups if g.group_type == GroupType.BURST)
    single_count = sum(1 for g in groups if g.group_type == GroupType.SINGLE)

    logger.info(f"Detection complete: {len(groups)} groups total "
               f"({aeb_count} AEB, {burst_count} Burst, {single_count} Single)")

    return groups


def print_group_summary(groups: list[BracketGroup]) -> str:
    """Print a human-readable summary of detected groups."""
    lines = [f"\n{'='*60}", "  BRACKET DETECTION SUMMARY", f"{'='*60}"]

    aeb_groups = [g for g in groups if g.group_type == GroupType.AEB]
    burst_groups = [g for g in groups if g.group_type == GroupType.BURST]
    single_groups = [g for g in groups if g.group_type == GroupType.SINGLE]

    lines.append(f"  AEB groups:    {len(aeb_groups)}")
    lines.append(f"  Burst groups:  {len(burst_groups)}")
    lines.append(f"  Single files:  {len(single_groups)}")
    lines.append(f"  Total files:   {sum(g.file_count for g in groups)}")
    lines.append("")

    for g in groups:
        ev_str = ", ".join(f"{f.exposure_compensation:+.1f}" for f in g.files)
        files_str = ", ".join(f.filename for f in g.files)
        lines.append(f"  [{g.group_type.value.upper()} #{g.group_id}] "
                    f"EV=[{ev_str}]")
        lines.append(f"    Files: {files_str}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)
