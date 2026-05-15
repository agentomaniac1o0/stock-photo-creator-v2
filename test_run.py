#!/usr/bin/env python3
"""
Real-Life Test Run: Process RAW files from SW-England-May26-01.
Auto-increments test run number and uploads to test_run_NN/ subdirectory.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NC_RAW_PATH, LOCAL_TEMP_DIR
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
    SelectionDecision,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_NAME = "SW-England-May26-01"
MAX_IMAGES = 50


def get_next_test_run_number(nc_client) -> int:
    """
    Check existing test_run_NN directories and return the next number.
    """
    batch_path = f"{NC_RAW_PATH}/{BATCH_NAME}"
    existing = nc_client.list_dir(batch_path)

    max_num = 0
    for entry in existing:
        name = entry.get("name", "")
        if name.startswith("test_run_"):
            try:
                num = int(name.replace("test_run_", ""))
                max_num = max(max_num, num)
            except ValueError:
                pass

    return max_num + 1


def main():
    nc_client = init_nextcloud()
    if nc_client is None:
        logger.error("Nextcloud credentials not found.")
        sys.exit(1)

    test_run_num = get_next_test_run_number(nc_client)
    test_run_label = f"test_run_{test_run_num:02d}"
    test_output = f"{NC_RAW_PATH}/{BATCH_NAME}/{test_run_label}"

    logger.info("=" * 60)
    logger.info("  REAL-LIFE TEST RUN")
    logger.info(f"  Batch: {BATCH_NAME}")
    logger.info(f"  Max images: {MAX_IMAGES}")
    logger.info(f"  Output: {test_output}")
    logger.info("=" * 60)

    # Step 1: Import
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Import RAW files")
    logger.info("=" * 60)

    import_result = import_raw_files(
        nc_client=nc_client,
        batch_name=BATCH_NAME,
        max_images=MAX_IMAGES,
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
    logger.info("STEP 5: Quality Scoring (AI)")
    logger.info("=" * 60)

    quality_result = score_all_images(groups)
    print(quality_result.summary())

    groups = quality_result.scored_groups

    # Step 6: Selection + Upload to test_run_NN
    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 6: Selection + Upload to {test_run_label}")
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
        all_decisions, f"{BATCH_NAME}_{test_run_label}", import_result.temp_dir
    )

    # Upload to test_run_NN
    nc_selected = f"{test_output}/selected"
    nc_rejected = f"{test_output}/rejected"

    upload_success, upload_failed = upload_files(all_kept, nc_client, nc_selected)
    rej_success, rej_failed = upload_files(all_rejected, nc_client, nc_rejected)
    upload_success += rej_success
    upload_failed += rej_failed

    # Upload report
    nc_client.upload_file(report_path, f"{test_output}/selection_report.json")

    # Update destinations
    for d in all_decisions:
        if d.decision == "keep":
            d.destination = f"{nc_selected}/{d.filepath.name}"
        else:
            d.destination = f"{nc_rejected}/{d.filepath.name}"

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
    logger.info("  TEST RUN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Selected:     {test_output}/selected/")
    logger.info(f"  Rejected:     {test_output}/rejected/")
    logger.info(f"  Report:       {test_output}/selection_report.json")
    logger.info("")
    logger.info("  Bitte prüfe die Ergebnisse in Nextcloud und gib Feedback.")


if __name__ == "__main__":
    main()
