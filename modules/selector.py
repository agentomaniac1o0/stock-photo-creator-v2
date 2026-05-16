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
)

logger = logging.getLogger(__name__)

# Quality thresholds
SHARPNESS_GATE_MIN = 10.0
NOISE_GATE_MIN = 30.0
BURST_ACTION_EXP_TIME = 1/250


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


def passes_hard_filters(metrics: ImageMetrics) -> tuple[bool, str]:
    """
    Step 1: Hard filters — reject too-blurry or unrecoverable images.

    Returns:
        (passes, reason) - reason is rejection reason if fails
    """
    if metrics.sharpness_score < SHARPNESS_GATE_MIN:
        return False, f"Too blurry (sharpness={metrics.sharpness_score:.0f} < {SHARPNESS_GATE_MIN:.0f})"
    return True, ""


def compute_comparison_score(metrics: ImageMetrics) -> tuple[float, float, float]:
    """
    Step 3: Compute comparison score for ranking.

    Primary: noise_curve (higher = less noise)
    Secondary: defect_score (higher = fewer defects)
    Tertiary: sharpness_curve (higher = sharper)

    Returns:
        (noise_ranked, defect_score, sharpness_ranked) for sorting
    """
    penalty = exposure_correction_penalty(metrics.filepath)
    noise_ranked = noise_curve(metrics.noise_score) - penalty
    sharpness_ranked = sharpness_curve(metrics.sharpness_score)
    return (noise_ranked, metrics.defect_score, sharpness_ranked)


def select_best_in_group(
    files: list[FileExifData],
    metrics_list: list[ImageMetrics],
) -> tuple[FileExifData, ImageMetrics]:
    """
    Select the best image from a group based on noise + defects.

    Returns:
        (best_file, best_metrics)
    """
    paired = list(zip(files, metrics_list))
    best_fd, best_m = max(
        paired,
        key=lambda x: compute_comparison_score(x[1])
    )
    return best_fd, best_m


def select_from_aeb_group(
    group: BracketGroup,
    metrics_list: list[ImageMetrics],
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select from AEB group:
    1. Apply hard filters (sharp < 10 → reject)
    2. From remaining, pick best by noise + defects
    3. If all fail filters → reject all
    """
    decisions = []
    kept = []
    rejected = []

    if not group.files:
        return kept, decisions

    candidates = []
    for fd, m in zip(group.files, metrics_list):
        passes, reason = passes_hard_filters(m)
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
        logger.info(f"AEB group #{group.group_id}: ALL images failed hard filters, rejecting all")
        return kept, decisions

    best_fd, best_m = select_best_in_group(
        [fd for fd, _ in candidates],
        [m for _, m in candidates],
    )
    kept.append(best_fd.filepath)
    decisions.append(SelectionDecision(
        filepath=best_fd.filepath,
        decision="keep",
        reason=f"Best quality in AEB group #{group.group_id} "
               f"(noise={best_m.noise_score:.0f}, defects={best_m.defect_score:.0f})",
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
               f"(noise={best_m.noise_score:.0f}, sharp={best_m.sharpness_score:.0f}), "
               f"rejected {len(group.files) - len(kept)} other(s)")

    return kept, decisions


def select_from_burst_group(
    group: BracketGroup,
    metrics_list: list[ImageMetrics],
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select from burst group:
    - Action sequence (ExpTime < 1/250s): keep all that pass hard filters
    - Normal burst: keep best by noise + defects
    """
    decisions = []
    kept = []
    rejected = []

    if not group.files:
        return kept, decisions

    passed = []
    for fd, m in zip(group.files, metrics_list):
        passes, reason = passes_hard_filters(m)
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
        logger.info(f"Burst group #{group.group_id}: ALL images failed hard filters")
        return kept, decisions

    if group.is_action_sequence:
        for fd, m in passed:
            kept.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                decision="keep",
                reason=f"Action sequence — passed filters "
                       f"(noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f})",
            ))
        logger.info(f"Burst group #{group.group_id} (action): kept {len(kept)}")
    else:
        best_fd, best_m = select_best_in_group(
            [fd for fd, _ in passed],
            [m for _, m in passed],
        )
        kept.append(best_fd.filepath)
        decisions.append(SelectionDecision(
            filepath=best_fd.filepath,
            decision="keep",
            reason=f"Best quality in burst group #{group.group_id} "
                   f"(noise={best_m.noise_score:.0f}, defects={best_m.defect_score:.0f})",
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
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select or reject a single image:
    - Keep if passes hard filters AND noise is not too high
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

    passes, reason = passes_hard_filters(m)
    if not passes:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"Failed hard filter: {reason}",
        ))
    elif m.noise_score < NOISE_GATE_MIN:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="reject",
            reason=f"Too noisy (noise={m.noise_score:.0f} < {NOISE_GATE_MIN:.0f})",
        ))
    else:
        kept.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            decision="keep",
            reason=f"Passed filters (noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f})",
        ))

    logger.info(f"Single {fd.filename}: {'kept' if kept else 'rejected'} "
               f"(noise={m.noise_score:.0f}, sharp={m.sharpness_score:.0f})")

    return kept, decisions


def upload_files(
    files: list[Path],
    nc_client: NextcloudClient,
    remote_dir: str,
) -> tuple[int, int]:
    """
    Upload files to Nextcloud.

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

    return success, failed


def generate_selection_report(
    decisions: list[SelectionDecision],
    batch_name: str,
    output_dir: Path,
) -> Path:
    """
    Generate a JSON report of all selection decisions.
    """
    report = {
        "batch_name": batch_name,
        "timestamp": datetime.now().isoformat(),
        "total_decisions": len(decisions),
        "kept": sum(1 for d in decisions if d.decision == "keep"),
        "rejected": sum(1 for d in decisions if d.decision == "reject"),
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

    metrics_map = {}
    for m in metrics_result.metrics:
        metrics_map[m.filepath] = m

    for group in metrics_result.scored_groups:
        group_metrics = [metrics_map.get(fd.filepath) for fd in group.files]
        group_metrics = [m for m in group_metrics if m is not None]

        if group.group_type == GroupType.AEB:
            kept, decisions = select_from_aeb_group(group, group_metrics)
        elif group.group_type == GroupType.BURST:
            kept, decisions = select_from_burst_group(group, group_metrics)
        else:
            kept, decisions = select_from_single(group, group_metrics)

        all_decisions.extend(decisions)
        all_kept.extend(kept)
        all_rejected.extend([d.filepath for d in decisions if d.decision == "reject"])

    report_path = generate_selection_report(all_decisions, f"{batch_name}_phase_1", temp_dir)

    nc_selected_dir = f"{NC_RAW_PATH}/{batch_name}/selected-phase_1"
    upload_success, upload_failed = upload_files(all_kept, nc_client, nc_selected_dir)

    nc_rejected_dir = f"{NC_RAW_PATH}/{batch_name}/rejected-phase_1"
    rej_success, rej_failed = upload_files(all_rejected, nc_client, nc_rejected_dir)
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
