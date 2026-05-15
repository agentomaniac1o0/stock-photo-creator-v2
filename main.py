#!/usr/bin/env python3
"""
Stock Photo Creator v2 — CLI Entry Point

Usage:
    python3 main.py <batch_name> [--max-images N] [--dry-run] [--local]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NC_BASE_PATH
from modules.nextcloud_client import init_nextcloud
from modules.importer import import_raw_files
from modules.bracket_detector import detect_brackets, print_group_summary
from modules.overexposure_checker import check_overexposure
from modules.exposure_aligner import align_exposures
from modules.quality_scorer import score_all_images
from modules.selector import select_and_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stock Photo Creator v2")
    parser.add_argument("batch", help="Batch folder name in Nextcloud RAW/")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit number of images (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without processing")
    parser.add_argument("--local", action="store_true",
                        help="Use local directories instead of Nextcloud")
    args = parser.parse_args()

    logger.info(f"Stock Photo Creator v2 — Batch: {args.batch}")

    if args.local:
        logger.info("Local mode not yet implemented. Use Nextcloud mode.")
        sys.exit(1)

    # Initialize Nextcloud
    nc_client = init_nextcloud()
    if nc_client is None:
        logger.error("Nextcloud credentials not found. Set NEXTCLOUD_HOST, "
                     "NEXTCLOUD_USER, NEXTCLOUD_APP_PASSWORD in env or ~/.env")
        sys.exit(1)

    # Step 1: Import
    logger.info("=" * 60)
    logger.info("STEP 1: Import RAW files from Nextcloud")
    logger.info("=" * 60)

    import_result = import_raw_files(
        nc_client=nc_client,
        batch_name=args.batch,
        max_images=args.max_images,
    )

    if import_result.file_count == 0:
        logger.error("No RAW files found. Exiting.")
        sys.exit(1)

    logger.info(f"Imported {import_result.file_count} files")

    if args.dry_run:
        logger.info("DRY RUN — stopping after import")
        import_result.cleanup()
        return

    # Step 2: Bracket Detection
    logger.info("=" * 60)
    logger.info("STEP 2: Detect bracket groups")
    logger.info("=" * 60)

    groups = detect_brackets(import_result.raw_files)
    print(print_group_summary(groups))

    if args.dry_run:
        logger.info("DRY RUN — stopping after bracket detection")
        import_result.cleanup()
        return

    # Step 3: Overexposure Check
    logger.info("=" * 60)
    logger.info("STEP 3: Check for unrecoverable clipping")
    logger.info("=" * 60)

    overexposure_result = check_overexposure(groups)
    print(overexposure_result.summary())

    if overexposure_result.total_rejected > 0:
        logger.warning(f"{overexposure_result.total_rejected} image(s) rejected due to "
                      f"unrecoverable clipping")

    groups = overexposure_result.checked_groups

    if not groups:
        logger.error("All images rejected. Exiting.")
        import_result.cleanup()
        sys.exit(1)

    # Step 4: Exposure Alignment
    logger.info("=" * 60)
    logger.info("STEP 4: Align exposures for underexposed images")
    logger.info("=" * 60)

    exposure_result = align_exposures(groups, output_dir=import_result.temp_dir / "corrected")
    print(exposure_result.summary())

    if exposure_result.total_failed > 0:
        logger.warning(f"{exposure_result.total_failed} image(s) failed exposure correction")

    groups = exposure_result.aligned_groups

    # Step 5: Quality Scoring
    logger.info("=" * 60)
    logger.info("STEP 5: AI Quality Scoring")
    logger.info("=" * 60)

    quality_result = score_all_images(groups)
    print(quality_result.summary())

    if quality_result.total_failed > 0:
        logger.warning(f"{quality_result.total_failed} image(s) failed quality scoring")

    groups = quality_result.scored_groups

    # Step 6: Selection + Upload
    logger.info("=" * 60)
    logger.info("STEP 6: Select best images and upload to Nextcloud")
    logger.info("=" * 60)

    selection_result = select_and_upload(
        quality_result=quality_result,
        nc_client=nc_client,
        batch_name=args.batch,
        temp_dir=import_result.temp_dir,
    )
    print(selection_result.summary())

    if selection_result.upload_failed > 0:
        logger.warning(f"{selection_result.upload_failed} file(s) failed to upload")

    logger.info(f"\n  Cleaned files uploaded to: Photos/StockFotoCreator/RAW/{args.batch}/cleaned/")
    logger.info(f"  Rejected files uploaded to: Photos/StockFotoCreator/RAW/{args.batch}/rejected/")
    logger.info(f"  Selection report: Photos/StockFotoCreator/RAW/{args.batch}/selection_report.json")

    # Cleanup temp
    import_result.cleanup()

    logger.info("\n  Durchlauf 1 (Bereinigung) abgeschlossen.")
    logger.info("  Prüfe die bereinigten RAWs in Nextcloud und starte dann Durchlauf 2.")

    # Cleanup temp
    import_result.cleanup()


if __name__ == "__main__":
    main()
