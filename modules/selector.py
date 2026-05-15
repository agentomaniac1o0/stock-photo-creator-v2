"""
Module 06: Selector + Upload

Selects the best images from scored groups and uploads them to Nextcloud.

Selection logic:
- AEB groups: Keep only the best quality image (highest score)
- Burst groups: Keep all images above quality threshold (user selects later)
- Singles: Keep if score meets minimum threshold

Uploads:
- Selected images → Nextcloud RAW/{batch}/cleaned/
- Rejected images → Nextcloud RAW/{batch}/rejected/
- Summary report → Nextcloud RAW/{batch}/selection_report.json

Input:  List of scored BracketGroup objects
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
from modules.quality_scorer import QualityScore, QualityScorerResult

logger = logging.getLogger(__name__)

# Quality thresholds
AEB_KEEP_BEST_ONLY = True
BURST_MIN_SCORE = 30.0
SINGLE_MIN_SCORE = 30.0


@dataclass
class SelectionDecision:
    """Selection decision for a single image."""
    filepath: Path
    score: float
    decision: str  # "keep" or "reject"
    reason: str
    destination: Optional[str] = None  # Nextcloud remote path


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
                lines.append(f"    {d.filepath.name}: {d.score:.0f}/100 [{status}] - {d.reason}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def select_from_aeb_group(
    group: BracketGroup,
    scores: list[QualityScore],
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select the best image from an AEB group.

    Only the highest-scoring image is kept.
    All others are rejected.
    """
    decisions = []
    kept = []
    rejected = []

    # Files are already sorted by score (best first) from quality scorer
    if not group.files:
        return kept, decisions

    best_file = group.files[0]
    best_score = scores[0].overall_score if scores else 0

    # Keep the best
    kept.append(best_file.filepath)
    decisions.append(SelectionDecision(
        filepath=best_file.filepath,
        score=best_score,
        decision="keep",
        reason=f"Best quality in AEB group #{group.group_id}",
    ))

    # Reject the rest
    for fd, score in zip(group.files[1:], scores[1:]):
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            score=score.overall_score,
            decision="reject",
            reason=f"Lower quality than best in AEB group #{group.group_id}",
        ))

    logger.info(f"AEB group #{group.group_id}: kept {best_file.filename} "
               f"(score={best_score:.0f}), rejected {len(group.files) - 1} other(s)")

    return kept, decisions


def select_from_burst_group(
    group: BracketGroup,
    scores: list[QualityScore],
    min_score: float = BURST_MIN_SCORE,
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select images from a burst group.

    All images above the quality threshold are kept.
    User makes final selection later.
    """
    decisions = []
    kept = []
    rejected = []

    for fd, score in zip(group.files, scores):
        if score.overall_score >= min_score:
            kept.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                score=score.overall_score,
                decision="keep",
                reason=f"Above quality threshold ({min_score:.0f}) in burst group #{group.group_id}",
            ))
        else:
            rejected.append(fd.filepath)
            decisions.append(SelectionDecision(
                filepath=fd.filepath,
                score=score.overall_score,
                decision="reject",
                reason=f"Below quality threshold ({min_score:.0f}) in burst group #{group.group_id}",
            ))

    logger.info(f"Burst group #{group.group_id}: kept {len(kept)}, "
               f"rejected {len(rejected)} (threshold={min_score:.0f})")

    return kept, decisions


def select_from_single(
    group: BracketGroup,
    scores: list[QualityScore],
    min_score: float = SINGLE_MIN_SCORE,
) -> tuple[list[Path], list[SelectionDecision]]:
    """
    Select or reject a single image.

    Kept if score meets minimum threshold.
    """
    decisions = []
    kept = []
    rejected = []

    if not group.files:
        return kept, decisions

    fd = group.files[0]
    score = scores[0] if scores else QualityScore(
        filepath=fd.filepath,
        overall_score=0,
        sharpness_score=0,
        noise_score=0,
        detail_score=0,
        defect_score=0,
    )

    if score.overall_score >= min_score:
        kept.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            score=score.overall_score,
            decision="keep",
            reason=f"Above quality threshold ({min_score:.0f})",
        ))
    else:
        rejected.append(fd.filepath)
        decisions.append(SelectionDecision(
            filepath=fd.filepath,
            score=score.overall_score,
            decision="reject",
            reason=f"Below quality threshold ({min_score:.0f})",
        ))

    logger.info(f"Single {fd.filename}: {'kept' if kept else 'rejected'} "
               f"(score={score.overall_score:.0f}, threshold={min_score:.0f})")

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
                "score": round(d.score, 1),
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
    quality_result: QualityScorerResult,
    nc_client: NextcloudClient,
    batch_name: str,
    temp_dir: Path,
    burst_min_score: float = BURST_MIN_SCORE,
    single_min_score: float = SINGLE_MIN_SCORE,
) -> SelectionResult:
    """
    Select best images and upload to Nextcloud.

    Args:
        quality_result: Result from quality scoring
        nc_client: Nextcloud client
        batch_name: Batch folder name
        temp_dir: Local temp directory with files
        burst_min_score: Minimum score for burst images
        single_min_score: Minimum score for single images

    Returns:
        SelectionResult with decisions and upload stats
    """
    all_decisions = []
    all_kept = []
    all_rejected = []

    # Map scores to files for lookup
    score_map = {}
    for score in quality_result.scores:
        score_map[score.filepath] = score

    for group in quality_result.scored_groups:
        # Get scores for this group's files
        group_scores = [score_map.get(fd.filepath) for fd in group.files]
        group_scores = [s for s in group_scores if s is not None]

        if group.group_type == GroupType.AEB:
            kept, decisions = select_from_aeb_group(group, group_scores)
        elif group.group_type == GroupType.BURST:
            kept, decisions = select_from_burst_group(
                group, group_scores, min_score=burst_min_score
            )
        else:
            kept, decisions = select_from_single(
                group, group_scores, min_score=single_min_score
            )

        all_decisions.extend(decisions)
        all_kept.extend(kept)
        all_rejected.extend([d.filepath for d in decisions if d.decision == "reject"])

    # Generate report locally
    report_path = generate_selection_report(all_decisions, batch_name, temp_dir)

    # Upload kept files to cleaned/
    nc_cleaned_dir = f"{NC_RAW_PATH}/{batch_name}/cleaned"
    upload_success, upload_failed = upload_files(all_kept, nc_client, nc_cleaned_dir)

    # Upload rejected files to rejected/
    nc_rejected_dir = f"{NC_RAW_PATH}/{batch_name}/rejected"
    rej_success, rej_failed = upload_files(all_rejected, nc_client, nc_rejected_dir)
    upload_success += rej_success
    upload_failed += rej_failed

    # Upload report
    nc_client.upload_file(report_path, f"{NC_RAW_PATH}/{batch_name}/selection_report.json")

    # Update decisions with remote paths
    for d in all_decisions:
        if d.decision == "keep":
            d.destination = f"{nc_cleaned_dir}/{d.filepath.name}"
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
