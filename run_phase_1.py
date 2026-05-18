#!/usr/bin/env python3
"""
Phase 1: Full Batch Processing — Selection & Cleanup (v3)

Multi-stage pipeline (analysis-driven order):
1. Import RAW files from Nextcloud
2. Bracket Detection (AEB/Burst/Single + action flag)
3. Exposure Alignment (align ALL to best-toncurve reference)
4. DRC Sky Recovery (highlight recovery via tone compression)
5. Overexposure Check (reject unrecoverable after alignment+DRC)
6. Quality Metrics (noise, sharp, defects, exposure, detail)
7. Multi-Stage Selection (hard filters → compare → low-light → tie-report)
8. Upload to selected-phase_1/ and rejected-phase_1/

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
from modules.exposure_aligner import align_exposures
from modules.sky_recovery import recover_highlights
from modules.overexposure_checker import check_overexposure
from modules.quality_scorer import compute_all_metrics
from modules.selector import select_and_upload
from modules.user_preferences import suggest_brightness_offset, load_preferences, save_preferences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Full Batch Selection")
    parser.add_argument("batch", help="Batch folder name in Nextcloud RAW/")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit number of images (for testing)")
    parser.add_argument("--brightness-offset", type=float, default=None,
                        help="Override learned brightness offset (EV). "
                             "Positive = brighter target. Overrides user_preferences.json.")
    parser.add_argument("--exemplars", nargs="*", default=None,
                        help="CR2 file(s) representing your ideal brightness. "
                             "Used to compute brightness_offset automatically.")
    args = parser.parse_args()

    BATCH_NAME = args.batch
    nc_client = init_nextcloud()
    if nc_client is None:
        logger.error("Nextcloud credentials not found.")
        sys.exit(1)

    selected_dir = f"{NC_RAW_PATH}/{BATCH_NAME}/selected-phase_1"
    rejected_dir = f"{NC_RAW_PATH}/{BATCH_NAME}/rejected-phase_1"

    logger.info("=" * 60)
    logger.info("  PHASE 1: FULL BATCH SELECTION (v3)")
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
        max_images=args.max_images,
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

    # Determine brightness offset (user preference)
    brightness_offset = 0.0
    if args.brightness_offset is not None:
        brightness_offset = args.brightness_offset
        logger.info(f"Using manual brightness offset: {brightness_offset:+.3f} EV")
    elif args.exemplars:
        exemplar_paths = [Path(p) for p in args.exemplars]
        brightness_offset = suggest_brightness_offset(exemplar_paths)
        save_preferences({"brightness_offset": brightness_offset})
    else:
        prefs = load_preferences()
        brightness_offset = prefs.get("brightness_offset", 0.0)
        if brightness_offset:
            logger.info(f"Using saved brightness offset: {brightness_offset:+.3f} EV")

    if brightness_offset:
        logger.info(f"→ Images will be aligned {abs(brightness_offset):.2f} EV "
                    f"{'brighter' if brightness_offset > 0 else 'darker'} than standard reference")

    # Step 3: Exposure Alignment (align ALL to best-toncurve reference)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Exposure Alignment")
    logger.info("=" * 60)

    exposure_result = align_exposures(
        groups,
        output_dir=import_result.temp_dir / "corrected",
        brightness_offset=brightness_offset,
    )
    print(exposure_result.summary())

    groups = exposure_result.aligned_groups

    # Step 4: DRC Sky Recovery (highlight recovery via tone compression)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: DRC Sky Recovery")
    logger.info("=" * 60)

    sky_result = recover_highlights(
        groups,
        output_dir=import_result.temp_dir / "drc",
    )
    print(sky_result.summary())

    # Step 5: Overexposure Check (now: after alignment+DRC)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Overexposure Check (after alignment + DRC)")
    logger.info("=" * 60)

    overexposure_result = check_overexposure(groups)
    print(overexposure_result.summary())

    groups = overexposure_result.checked_groups

    if not groups:
        logger.error("All images rejected.")
        import_result.cleanup()
        sys.exit(1)

    # Step 6: Quality Metrics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Quality Metrics")
    logger.info("=" * 60)

    metrics_result = compute_all_metrics(groups)
    print(metrics_result.summary())

    # Step 7: Selection + Upload
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Selection + Upload (low-light, DRC, tie-aware)")
    logger.info("=" * 60)

    result = select_and_upload(
        metrics_result=metrics_result,
        nc_client=nc_client,
        batch_name=BATCH_NAME,
        temp_dir=import_result.temp_dir,
    )

    print(result.summary())

    # Step 8: Cleanup
    import_result.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Selected:     {selected_dir}/")
    logger.info(f"  Rejected:     {rejected_dir}/")
    logger.info(f"  Report:       {NC_RAW_PATH}/{BATCH_NAME}/phase_1_report.json")


if __name__ == "__main__":
    main()
