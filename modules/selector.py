"""
Module 06: Selector + Upload (v2 — Multi-Stage Filter Pipeline)

Multi-stage selection pipeline:
1. HARD FILTERS: Reject too-blurry (sharp < 10) and unrecoverable overexposed
2. EXPOSURE ALIGN: All AEB images aligned to reference (done in exposure_aligner)
3. COMPARE & SELECT:
   - AEB groups: 1 image wins (least noise + fewest defects)
   - Burst normal: 1 image wins (least noise + fewest defects)
   - Burst action (ExpTime < 1/250s): keep all that pass filters
   - Singles: keep if not too blurry + moderate noise
4. UPLOAD: selected → Nextcloud, rejected → Nextcloud

Input:  BracketGroup objects with ImageMetrics attached
Output: Uploaded files + selection report
"""
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import NC_RAW_PATH
from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.nextcloud_client import NextcloudClient
from modules.quality_scorer import (
    ImageMetrics,
    MetricsResult,
    noise_curve,
    sharpness_curve,
    exposure_correction_penalty,
    ATMO_SHARPNESS_GATE,
)

logger = logging.getLogger(__name__)

# Quality thresholds
SHARPNESS_GATE_MIN = 7.0
LOW_LIGHT_SHARPNESS_GATE = 5.0
NOISE_GATE_MIN = 30.0
LOW_LIGHT_NOISE_GATE = 15.0
BURST_ACTION_EXP_TIME = 1 / 250

# DRC bonus/malus for comparison scoring
DRC_NO_NEED_BONUS = 10.0
DRC_SUCCESS_BONUS = 5.0
DRC_FAIL_PENALTY = 20.0

# Tie detection: if top scores are within this %, mark for manual review
TIE_THRESHOLD_PCT = 2.0


@dataclass
class SelectionDecision:
    """Selection decision for a single image."""
    filepath: Path
    decision: str  # "keep" or "reject"
    reason: str
    destination: Optional[str] = None


@dataclass
class SelectionResult:
    """Result of the selection and upload process."""
    decisions: list[SelectionDecision]
    kept_files: list[Path]
    rejected_files: list[Path]
    upload_success: int = 0
    upload_failed: int = 0
    report_path: Optional[Path] = None

    @property
    def kept_count(self) -> int:
        return len(self.kept_files)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_files)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  SELECTION & UPLOAD SUMMARY",
            f"{'='*60}",
            f"  Images kept:     {self.kept_count}",
            f"  Images rejected: {self.rejected_count}",
            f"  Upload success:  {self.upload_success}",
            f"  Upload failed:   {self.upload_failed}",
        ]
        if self.decisions:
            lines.append("")
            lines.append("  Decisions:")
            for d in self.decisions:
                status = "KEPT" if d.decision == "keep" else "REJECTED"
                lines.append(f"    {d.filepath.name}: [{status}] - {d.reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def detect_low_light_scene(group: BracketGroup, metrics_list: list[ImageMetrics]) -> tuple[bool, str]:
    """
    Detect if a group is a low-light scene (indoor, church, night, etc.).

    Low-light scenes should use relaxed thresholds because noise
    is inherently higher and sharpness is harder to achieve.

    Detection criteria (either qualifies):
    - ISO > 1600 (from EXIF)
    - Average group exposure_score < 40

    Returns:
        (is_low_light, reason)
    """
    iso_trigger = False
    exp_trigger = False

    for fd in group.files:
        if fd.iso:
            try:
                iso_val = int(str(fd.iso))
                if iso_val > 1600:
                    iso_trigger = True
                    break
            except (ValueError, TypeError):
                pass

    if metrics_list:
        avg_exp = sum(m.exposure_score for m in metrics_list) / len(metrics_list)
        if avg_exp < 40:
            exp_trigger = True

    reasons = []
    if iso_trigger:
        reasons.append("ISO>1600")
    if exp_trigger:
        reasons.append(f"avg exposure<40")

    if reasons:
        return True, f"Low-light detected: {', '.join(reasons)}"
    return False, ""


def passes_hard_filters(metrics: ImageMetrics, low_light: bool = False) -> tuple[bool, str]:
    """
    Step 1: Hard filters — reject too-blurry or unrecoverable images.

    Sharpness gates (descending priority):
    1. Atmo scene (sunset/mood):           gate = 3  (mood > sharpness)
    2. Low-light (ISO>1600 or exp<40):     gate = 5
    3. Normal:                              gate = 7

    Returns:
        (passes, reason) - reason is rejection reason if fails
    """
    if not metrics.blur_recoverable:
        return False, (f"Unrecoverable blur ({metrics.blur_type}: "
                       f"sharpness={metrics.sharpness_score:.0f})")
    if metrics.atmo_scene:
        gate = ATMO_SHARPNESS_GATE
        label = "atmo"
    elif low_light:
        gate = LOW_LIGHT_SHARPNESS_GATE
        label = "low-light"
    else:
        gate = SHARPNESS_GATE_MIN
        label = "normal"
    if metrics.sharpness_score < gate:
        return False, (f"Too blurry ({label}: sharpness={metrics.sharpness_score:.0f} < {gate:.0f})")
    return True, ""


def compute_drc_bonus(fd: FileExifData) -> float:
    """Calculate DRC bonus based on recovery outcome."""
    if not fd.drc_applied and not fd.drc_success:
        return DRC_NO_NEED_BONUS
    elif fd.drc_applied and fd.drc_success:
        return DRC_SUCCESS_BONUS
    elif fd.drc_applied and not fd.drc_success:
        return -DRC_FAIL_PENALTY
    return 0.0


NOISE_WEIGHT = 1.0
SHARPNESS_WEIGHT = 1.5
DEFECT_WEIGHT = 0.1
EXPOSURE_WEIGHT = 0.1
SHAKE_PENALTY = 30.0
MOTION_PENALTY = 20.0


def compute_comparison_score(metrics: ImageMetrics, fd: FileExifData = None) -> float:
    """
    Step 3: Compute comparison score for ranking (higher = better).

    Weighted sum:
      noise_curve × NOISE_WEIGHT
      + sharpness_curve × SHARPNESS_WEIGHT
      + defect_score × DEFECT_WEIGHT
      + exposure_score × EXPOSURE_WEIGHT
      + DRC bonus/malus
      − exposure correction penalty
      − unrecoverable blur penalty

    Sharpness is weighted 1.5× over noise so that visible blur
    outweighs minor noise differences — unless the blur is
    recoverable (defocus/soft) and the scene benefits from it.
    """
    ev_diff_val = fd.ev_diff if fd else 0.0
    penalty = exposure_correction_penalty(ev_diff_val)

    score = (
        noise_curve(metrics.noise_score) * NOISE_WEIGHT
        + sharpness_curve(metrics.sharpness_score) * SHARPNESS_WEIGHT
        + metrics.defect_score * DEFECT_WEIGHT
        + metrics.exposure_score * EXPOSURE_WEIGHT
        - penalty
    )

    # DRC bonus/malus
    drc_bonus = compute_drc_bonus(fd) if fd else 0.0
    score += drc_bonus

    # Unrecoverable blur penalty (safety net beyond hard filters)
    if not metrics.blur_recoverable:
        if metrics.blur_type == "shake":
            score -= SHAKE_PENALTY
        elif metrics.blur_type == "motion":
            score -= MOTION_PENALTY

    return score


def select_best_in_group(
    files: list[FileExifData],
    metrics_list: list[ImageMetrics],
) -> tuple[FileExifData, ImageMetrics, str]:
    """
    Select the best image from a group based on ranking.

    Ranking:
    1. noise_curve (primary)
    2. defect_score (secondary)
    3. sharpness_curve (tertiary)
    4. exposure_score (quartary)
    + DRC bonus

    When top-2 candidates are within TIE_THRESHOLD_PCT,
    the image is flagged for manual review.

    Returns:
        (best_file, best_metrics, tie_warning)
    """
    paired = list(zip(files, metrics_list))
    scored = [(compute_comparison_score(m, fd), fd, m) for fd, m in paired]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_fd, best_m = scored[0][1], scored[0][2]
    tie_warning = ""

    if len(scored) >= 2:
        top_score = scored[0][0]
        second_score = scored[1][0]
        if top_score > 0 and second_score > 0:
            diff_pct = abs(top_score - second_score) / max(top_score, second_score) * 100
            if diff_pct < TIE_THRESHOLD_PCT:
                tie_warning = (
                    f"TIE: #{scored[0][1].filename} vs #{scored[1][1].filename} "
                    f"({diff_pct:.1f}% difference) — manual review recommended"
                )

    return best_fd, best_m, tie_warning


def select_from_aeb_group(
    group: BracketGroup,
    metrics_list: list[ImageMetrics],
    low_light: bool = False,
    tie_warnings: list[str] = None,
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select from AEB group:
    1. Detect low-light scene → relaxed thresholds
    2. Apply hard filters (sharp < 10 normal, < 5 low-light)
    3. Reject DRC failures
    4. From remaining, pick best by noise → defects → sharpness → exposure + DRC
    5. Generate tie-warnings when scores are close
    """
    decisions = []
    kept = []
    rejected = []
    local_ties = tie_warnings if tie_warnings is not None else []

    if not group.files:
        return kept, decisions

    candidates = []
    for fd, m in zip(group.files, metrics_list):
        if fd.drc_applied and not fd.drc_success:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="reject",
                reason=f"DRC failed: sky not recoverable",
            ))
            continue

        passes, reason = passes_hard_filters(m, low_light=low_light)
        if passes:
            candidates.append((fd, m))
        else:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="reject",
                reason=f"Failed hard filter: {reason}",
            ))

    if not candidates:
        # Last resort: pick the best among failed-filter images
        # rather than rejecting the entire group
        fallback = [(fd, m) for fd, m in zip(group.files, metrics_list)
                     if not (fd.drc_applied and not fd.drc_success)]
        if fallback:
            best_fd, best_m, tie = select_best_in_group(
                [fd for fd, _ in fallback],
                [m for _, m in fallback],
            )
            kept.append(best_fd.filepath)
            decisions.append(SelectionDecision(
                filepath=best_fd.filepath,
                decision="keep",
                reason=(f"LAST RESORT in AEB group #{group.group_id} "
                        f"(sharp={best_m.sharpness_score:.0f}, noise={best_m.noise_score:.0f})"
                        f" — only candidate"),
            ))
            for fd, m in fallback:
                if fd.filepath != best_fd.filepath:
                    rejected.append(fd.filepath)
            logger.info(f"AEB group #{group.group_id}: last resort kept {best_fd.filename} "
                       f"(sharp={best_m.sharpness_score:.0f})")
        else:
            logger.info(f"AEB group #{group.group_id}: ALL images failed, rejecting all")
        return kept, decisions

    best_fd, best_m, tie = select_best_in_group(
        [fd for fd, _ in candidates],
        [m for _, m in candidates],
    )
    if tie:
        local_ties.append(f"AEB #{group.group_id}: {tie}")

    drc_label = ""
    if best_fd.drc_applied and best_fd.drc_success:
        drc_label = " [DRC sky recovered]"
    elif not best_fd.drc_applied:
        drc_label = " [natural highlights]"

    kept.append(best_fd.filepath)
    decisions.append(SelectionDecision(
        filepath=best_fd.filepath,
        decision="keep",
        reason=(f"Best quality in AEB group #{group.group_id} "
                f"(noise={best_m.noise_score:.0f}, defects={best_m.defect_score:.0f})"
                f"{drc_label}"),
    ))

    for fd, m in candidates:
        if fd.filepath != best_fd.filepath:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="reject",
                reason=f"Lower quality than best in AEB group #{group.group_id}",
            ))

    logger.info(f"AEB group #{group.group_id}: kept {best_fd.filename} "
               f"(noise={best_m.noise_score:.0f}, sharp={best_m.sharpness_score:.0f})"
               f"{drc_label}, rejected {len(group.files) - len(kept)} other(s)")

    return kept, decisions


def select_from_burst_group(
    group: BracketGroup,
    metrics_list: list[ImageMetrics],
    low_light: bool = False,
    tie_warnings: list[str] = None,
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select from burst group:
    - Action sequence (ExpTime < 1/250s): keep all that pass hard filters
    - Normal burst: keep best by noise → defects → sharpness → exposure + DRC
    """
    decisions = []
    kept = []
    rejected = []
    local_ties = tie_warnings if tie_warnings is not None else []

    if not group.files:
        return kept, decisions

    passed = []
    for fd, m in zip(group.files, metrics_list):
        if fd.drc_applied and not fd.drc_success:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="reject",
                reason=f"DRC failed: sky not recoverable",
            ))
            continue

        passes, reason = passes_hard_filters(m, low_light=low_light)
        if passes:
            passed.append((fd, m))
        else:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="reject",
                reason=f"Failed hard filter: {reason}",
            ))

    if not passed:
        # Last resort: pick best among failed-filter images
        fallback = [(fd, m) for fd, m in zip(group.files, metrics_list)
                     if not (fd.drc_applied and not fd.drc_success)]
        if fallback:
            best_fd, best_m, _ = select_best_in_group(
                [fd for fd, _ in fallback],
                [m for _, m in fallback],
            )
            kept.append(best_fd.filepath)
            decisions.append(SelectionDecision(
                filepath=best_fd.filepath,
                decision="keep",
                reason=f"LAST RESORT in burst group #{group.group_id} "
                       f"(sharp={best_m.sharpness_score:.0f}) — only candidate",
            ))
            logger.info(f"Burst group #{group.group_id}: last resort kept {best_fd.filename}")
        else:
            logger.info(f"Burst group #{group.group_id}: ALL images failed")
        return kept, decisions

    if group.is_action_sequence:
        for fd, m in passed:
            drc_label = ""
            if fd.drc_applied and fd.drc_success:
                drc_label = " [DRC]"
            kept.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="keep",
                reason=f"Action sequence — passed filters "
                       f"(noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f})"
                       f"{drc_label}",
            ))
        logger.info(f"Burst group #{group.group_id} (action): kept {len(kept)}")
    else:
        best_fd, best_m, tie = select_best_in_group(
            [fd for fd, _ in passed],
            [m for _, m in passed],
        )
        if tie:
            local_ties.append(f"Burst #{group.group_id}: {tie}")

        drc_label = ""
        if best_fd.drc_applied and best_fd.drc_success:
            drc_label = " [DRC sky recovered]"
        elif not best_fd.drc_applied:
            drc_label = " [natural highlights]"

        kept.append(best_fd.filepath)
        decisions.append(SelectionDecision(
            filepath=best_fd.filepath,
            decision="keep",
            reason=f"Best quality in burst group #{group.group_id} "
                   f"(noise={best_m.noise_score:.0f}, defects={best_m.defect_score:.0f})"
                   f"{drc_label}",
        ))

        for fd, m in passed:
            if fd.filepath != best_fd.filepath:
                rejected.append(fd.filepath)
                decisions.append(SelectionDecision(
                    filepath=fd.filepath,
                    decision="reject",
                    reason=f"Lower quality than best in burst group #{group.group_id}",
                ))
        logger.info(f"Burst group #{group.group_id}: kept {best_fd.filename}")

    return kept, decisions


def select_from_single(
    group: BracketGroup,
    metrics_list: list[ImageMetrics],
    low_light: bool = False,
    tie_warnings: list[str] = None,
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select or reject a single image:
    - Reject DRC failures
    - Keep if passes hard filters AND noise is not too high
    - Low-light scenes use relaxed noise gate (15 instead of 30)
    """
    decisions = []
    kept = []
    rejected = []

    if not group.files:
        return kept, decisions

    fd = group.files[0]
    m = metrics_list[0] if metrics_list else ImageMetrics(
        filepath=fd.filepath,
        exposure_score=50.0,
        noise_score=50.0,
        sharpness_score=50.0,
        detail_score=50.0,
        defect_score=50.0,
    )

    if fd.drc_applied and not fd.drc_success:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"DRC failed: sky not recoverable",
        ))
        logger.info(f"Single {fd.filename}: rejected (DRC failed)")
        return kept, decisions

    noise_gate = LOW_LIGHT_NOISE_GATE if low_light else NOISE_GATE_MIN
    gate_label = "low-light" if low_light else "normal"

    passes, reason = passes_hard_filters(m, low_light=low_light)
    is_drc_fail = fd.drc_applied and not fd.drc_success
    if not passes and not is_drc_fail:
        # Last resort: keep single image even if slightly blurry,
        # as long as noise is acceptable — this is the only
        # representation of this subject/scene
        if m.noise_score >= noise_gate:
            drc_label = ""
            if fd.drc_applied and fd.drc_success:
                drc_label = " [DRC sky recovered]"
            elif not fd.drc_applied:
                drc_label = " [natural highlights]"
            kept.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="keep",
                reason=f"LAST RESORT single (sharp={m.sharpness_score:.0f}, "
                       f"noise={m.noise_score:.0f}) — only candidate{drc_label}",
            ))
            logger.info(f"Single {fd.filename}: last resort kept "
                       f"(sharp={m.sharpness_score:.0f})")
            return kept, decisions
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"Failed hard filter + too noisy: {reason}",
        ))
    elif not passes:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"Failed hard filter: {reason}",
        ))
    elif m.noise_score < noise_gate:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"Too noisy ({gate_label}: noise={m.noise_score:.0f} < {noise_gate:.0f})",
        ))
    else:
        drc_label = ""
        if fd.drc_applied and fd.drc_success:
            drc_label = " [DRC sky recovered]"
        elif not fd.drc_applied:
            drc_label = " [natural highlights]"
        kept.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="keep",
            reason=f"Passed filters (noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f})"
                   f"{drc_label}",
        ))

    logger.info(f"Single {fd.filename}: {'kept' if kept else 'rejected'} "
               f"(noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f}, "
               f"{gate_label} gate={noise_gate:.0f})")

    return kept, decisions


def find_pp3_sidecars(filepath: Path, temp_dir: Path = None) -> list[Path]:
    """
    Find associated pp3 sidecar files for a CR2.
    Checks: source dir, _drc.pp3, _exposure_corrected.pp3
    """
    found = []
    stem = filepath.stem
    base = filepath.parent

    search_paths = [
        base / f"{stem}.pp3",
        base / f"{stem}_drc.pp3",
        base / f"{stem}_exposure_corrected.pp3",
    ]
    if temp_dir:
        search_paths.extend([
            temp_dir / "drc" / f"{stem}_drc.pp3",
            temp_dir / "corrected" / f"{stem}_exposure_corrected.pp3",
        ])
    for sp in search_paths:
        if sp.exists():
            found.append(sp)
    return found


def upload_files(
    files: list[Path],
    nc_client: NextcloudClient,
    remote_dir: str,
    temp_dir: Path = None,
) -> tuple[int, int]:
    """
    Upload files (CR2 + associated pp3 sidecars) to Nextcloud.

    Returns:
        (success_count, failed_count)
    """
    success = 0
    failed = 0

    nc_client.mkdir(remote_dir)

    for filepath in files:
        if not filepath.exists():
            logger.warning(f"File not found for upload: {filepath}")
            failed += 1
            continue

        remote_path = f"{remote_dir}/{filepath.name}"
        if nc_client.upload_file(filepath, remote_path):
            logger.info(f"  Uploaded: {filepath.name}")
            success += 1
        else:
            logger.error(f"  Upload failed: {filepath.name}")
            failed += 1

        for pp3_path in find_pp3_sidecars(filepath, temp_dir):
            remote_pp3 = f"{remote_dir}/{pp3_path.name}"
            if nc_client.upload_file(pp3_path, remote_pp3):
                logger.info(f"  Uploaded: {pp3_path.name}")
                success += 1
            else:
                logger.error(f"  Upload failed: {pp3_path.name}")
                failed += 1

    return success, failed


def generate_selection_report(
    decisions: list[SelectionDecision],
    batch_name: str,
    output_dir: Path,
) -> Path:
    """
    Generate a JSON report of all selection decisions.
    """
    tie_warnings = [d for d in decisions if "TIE" in d.reason.upper()]

    report = {
        "batch_name": batch_name,
        "timestamp": datetime.now().isoformat(),
        "total_decisions": len(decisions),
        "kept": sum(1 for d in decisions if d.decision == "keep"),
        "rejected": sum(1 for d in decisions if d.decision == "reject"),
        "ties": len(tie_warnings),
        "tie_details": [d.reason for d in tie_warnings],
        "decisions": [
            {
                "filename": d.filepath.name,
                "decision": d.decision,
                "reason": d.reason,
            }
            for d in decisions
        ],
    }

    report_path = output_dir / "selection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Selection report written: {report_path}")
    return report_path


def select_and_upload(
    metrics_result: MetricsResult,
    nc_client: NextcloudClient,
    batch_name: str,
    temp_dir: Path,
) -> SelectionResult:
    """
    Select best images and upload to Nextcloud.

    Args:
        metrics_result: Result from metric computation
        nc_client: Nextcloud client
        batch_name: Batch folder name
        temp_dir: Local temp directory with files

    Returns:
        SelectionResult with decisions and upload stats
    """
    all_decisions = []
    all_kept = []
    all_rejected = []
    tie_warnings: list[str] = []

    metrics_map = {}
    for m in metrics_result.metrics:
        metrics_map[m.filepath] = m

    for group in metrics_result.scored_groups:
        group_metrics = [metrics_map.get(fd.filepath) for fd in group.files]
        group_metrics = [m for m in group_metrics if m is not None]

        low_light, ll_reason = detect_low_light_scene(group, group_metrics)
        if low_light:
            logger.info(f"Group #{group.group_id} [{group.group_type.value}]: {ll_reason}")

        if group.group_type == GroupType.AEB:
            kept, decisions = select_from_aeb_group(
                group, group_metrics, low_light=low_light, tie_warnings=tie_warnings)
        elif group.group_type == GroupType.BURST:
            kept, decisions = select_from_burst_group(
                group, group_metrics, low_light=low_light, tie_warnings=tie_warnings)
        else:
            kept, decisions = select_from_single(
                group, group_metrics, low_light=low_light, tie_warnings=tie_warnings)

        all_decisions.extend(decisions)
        all_kept.extend(kept)
        all_rejected.extend([d.filepath for d in decisions if d.decision == "reject"])

    report_path = generate_selection_report(
        all_decisions, f"{batch_name}_phase_1", temp_dir)

    if tie_warnings:
        logger.info(f"\n{'='*60}")
        logger.info("  TIE WARNINGS — Manual review recommended:")
        for tw in tie_warnings:
            logger.info(f"    ⚠ {tw}")
        logger.info(f"{'='*60}")

    nc_selected_dir = f"{NC_RAW_PATH}/{batch_name}/selected-phase_1"
    upload_success, upload_failed = upload_files(all_kept, nc_client, nc_selected_dir, temp_dir)

    nc_rejected_dir = f"{NC_RAW_PATH}/{batch_name}/rejected-phase_1"
    rej_success, rej_failed = upload_files(all_rejected, nc_client, nc_rejected_dir, temp_dir)
    upload_success += rej_success
    upload_failed += rej_failed

    nc_client.upload_file(report_path, f"{NC_RAW_PATH}/{batch_name}/phase_1_report.json")

    for d in all_decisions:
        if d.decision == "keep":
            d.destination = f"{nc_selected_dir}/{d.filepath.name}"
        else:
            d.destination = f"{nc_rejected_dir}/{d.filepath.name}"

    result = SelectionResult(
        decisions=all_decisions,
        kept_files=all_kept,
        rejected_files=all_rejected,
        upload_success=upload_success,
        upload_failed=upload_failed,
        report_path=report_path,
    )

    logger.info(result.summary())
    return result
