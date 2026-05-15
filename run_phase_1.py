#!/usr/bin/env python3
"""
Phase 1: Full Batch Processing — Selection & Cleanup

Processes ALL RAW files in a batch, selects the best images,
and uploads to selected-phase_1/ and rejected-phase_1/.

Usage:
    python3 run_phase_1.py SW-England-May26-01
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NC_RAW_PATH
from modules.nextcloud_client import init_nextcloud
from modules.importer import import_raw_files
from modules.bracket_detector import detect_brackets, print_group_summary
from modules.overexposure_checker import check_overexposure
from modules.exposure_aligner import align_exposures
from modules.quality_scorer import score_all_images
from modules.selector import (
    select_from_aeb_group,
    select_from_burst_group,
    select_from_single,
    generate_selection_report,
    upload_files,
    SelectionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Full Batch Selection")
    parser.add_argument("batch", help="Batch folder name in Nextcloud RAW/")
    args = parser.parse_args()

    BATCH_NAME = args.batch
    nc_client = init_nextcloud()
    if nc_client is None:
        logger.error("Nextcloud credentials not found.")
        sys.exit(1)

    selected_dir = f"{NC_RAW_PATH}/{BATCH_NAME}/selected-phase_1"
    rejected_dir = f"{NC_RAW_PATH}/{BATCH_NAME}/rejected-phase_1"

    logger.info("=" * 60)
    logger.info("  PHASE 1: FULL BATCH SELECTION")
    logger.info(f"  Batch: {BATCH_NAME}")
    logger.info(f"  Selected → {selected_dir}")
    logger.info(f"  Rejected → {rejected_dir}")
    logger.info("=" * 60)

    # Step 1: Import
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Import RAW files")
    logger.info("=" * 60)

    import_result = import_raw_files(
        nc_client=nc_client,
        batch_name=BATCH_NAME,
        max_images=0,  # No limit — process all files
    )

    if import_result.file_count == 0:
        logger.error("No RAW files found.")
        sys.exit(1)

    logger.info(f"Imported {import_result.file_count} files")

    # Step 2: Bracket Detection
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Bracket Detection")
    logger.info("=" * 60)

    groups = detect_brackets(import_result.raw_files)
    print(print_group_summary(groups))

    # Step 3: Overexposure Check
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Overexposure Check")
    logger.info("=" * 60)

    overexposure_result = check_overexposure(groups)
    print(overexposure_result.summary())

    groups = overexposure_result.checked_groups

    if not groups:
        logger.error("All images rejected.")
        import_result.cleanup()
        sys.exit(1)

    # Step 4: Exposure Alignment
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Exposure Alignment")
    logger.info("=" * 60)

    exposure_result = align_exposures(
        groups,
        output_dir=import_result.temp_dir / "corrected",
    )
    print(exposure_result.summary())

    groups = exposure_result.aligned_groups

    # Step 5: Quality Scoring
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Quality Scoring")
    logger.info("=" * 60)

    quality_result = score_all_images(groups)
    print(quality_result.summary())

    groups = quality_result.scored_groups

    # Step 6: Selection + Upload to selected-phase_1 / rejected-phase_1
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Selection + Upload")
    logger.info("=" * 60)

    all_decisions = []
    all_kept = []
    all_rejected = []

    score_map = {s.filepath: s for s in quality_result.scores}

    for group in quality_result.scored_groups:
        group_scores = [score_map.get(fd.filepath) for fd in group.files]
        group_scores = [s for s in group_scores if s is not None]

        if group.group_type.value == "aeb":
            kept, decisions = select_from_aeb_group(group, group_scores)
        elif group.group_type.value == "burst":
            kept, decisions = select_from_burst_group(group, group_scores, min_score=50.0)
        else:
            kept, decisions = select_from_single(group, group_scores, min_score=50.0)

        all_decisions.extend(decisions)
        all_kept.extend(kept)
        all_rejected.extend([d.filepath for d in decisions if d.decision == "reject"])

    # Generate report
    report_path = generate_selection_report(
        all_decisions, f"{BATCH_NAME}_phase_1", import_result.temp_dir
    )

    # Upload to selected-phase_1 / rejected-phase_1
    upload_success, upload_failed = upload_files(all_kept, nc_client, selected_dir)
    rej_success, rej_failed = upload_files(all_rejected, nc_client, rejected_dir)
    upload_success += rej_success
    upload_failed += rej_failed

    # Upload report
    nc_client.upload_file(report_path, f"{NC_RAW_PATH}/{BATCH_NAME}/phase_1_report.json")

    # Update destinations
    for d in all_decisions:
        if d.decision == "keep":
            d.destination = f"{selected_dir}/{d.filepath.name}"
        else:
            d.destination = f"{rejected_dir}/{d.filepath.name}"

    result = SelectionResult(
        decisions=all_decisions,
        kept_files=all_kept,
        rejected_files=all_rejected,
        upload_success=upload_success,
        upload_failed=upload_failed,
        report_path=report_path,
    )

    print(result.summary())

    # Cleanup
    import_result.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Selected:     {selected_dir}/")
    logger.info(f"  Rejected:     {rejected_dir}/")
    logger.info(f"  Report:       {NC_RAW_PATH}/{BATCH_NAME}/phase_1_report.json")


if __name__ == "__main__":
    main()
