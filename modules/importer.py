"""
Module 01: Importer

Downloads RAW files from Nextcloud RAW/{batch}/ to local temp directory.

Input:  Nextcloud path + batch name
Output: List of local file paths to downloaded RAW files
"""
import logging
import shutil
from pathlib import Path

from config.settings import NC_RAW_PATH, LOCAL_TEMP_DIR, RAW_EXTENSIONS, JPEG_EXTENSIONS
from modules.nextcloud_client import NextcloudClient

logger = logging.getLogger(__name__)


class ImportResult:
    """Result of the import operation."""
    def __init__(self, raw_files: list[Path], temp_dir: Path, batch_name: str):
        self.raw_files = raw_files
        self.temp_dir = temp_dir
        self.batch_name = batch_name

    @property
    def file_count(self) -> int:
        return len(self.raw_files)

    def cleanup(self):
        """Remove temp directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")


def import_raw_files(
    nc_client: NextcloudClient,
    batch_name: str,
    temp_dir: Path = None,
    max_images: int = 0,
    skip_images: int = 0,
) -> ImportResult:
    """
    Download RAW files from Nextcloud to local temp directory.

    Args:
        nc_client: Initialized Nextcloud client
        batch_name: Name of the batch folder (e.g. "Barcelona_Trip")
        temp_dir: Optional custom temp directory
        max_images: Limit number of images (0 = no limit, for testing)
        skip_images: Skip first N images (for incremental processing)

    Returns:
        ImportResult with list of local file paths
    """
    if temp_dir is None:
        temp_dir = LOCAL_TEMP_DIR / f"import_{batch_name}"

    raw_dir = temp_dir / "RAW"
    raw_dir.mkdir(parents=True, exist_ok=True)

    nc_path = f"{NC_RAW_PATH}/{batch_name}" if batch_name else NC_RAW_PATH
    logger.info(f"Downloading from Nextcloud: {nc_path}")

    items = nc_client.list_dir(nc_path)
    if not items:
        nc_path = f"{NC_RAW_PATH}/{batch_name}"
        items = nc_client.list_dir(nc_path)

    downloaded = []
    skipped = 0
    raw_extensions = RAW_EXTENSIONS | JPEG_EXTENSIONS

    for item in items:
        name = item["name"]
        ext = Path(name).suffix.lower()
        if ext not in raw_extensions:
            continue

        if skip_images > 0 and skipped < skip_images:
            skipped += 1
            continue

        remote_file = f"{nc_path}/{name}"
        local_file = raw_dir / name

        if nc_client.download_file(remote_file, local_file):
            downloaded.append(local_file)
            logger.info(f"  Downloaded: {name}")

        if max_images > 0 and len(downloaded) >= max_images:
            break

    if not downloaded:
        for item in items:
            if "/" in item["name"]:
                continue
            sub_path = f"{nc_path}/{item['name']}"
            sub_items = nc_client.list_dir(sub_path)
            for sub_item in sub_items:
                name = sub_item["name"]
                ext = Path(name).suffix.lower()
                if ext not in raw_extensions:
                    continue
                if skip_images > 0 and skipped < skip_images:
                    skipped += 1
                    continue
                remote_file = f"{sub_path}/{name}"
                local_file = raw_dir / name
                if nc_client.download_file(remote_file, local_file):
                    downloaded.append(local_file)
                    logger.info(f"  Downloaded from subfolder: {name}")
                if max_images > 0 and len(downloaded) >= max_images:
                    break
            if max_images > 0 and len(downloaded) >= max_images:
                break

    if skip_images > 0:
        logger.info(f"Skipped {skipped} file(s)")

    logger.info(f"Import complete: {len(downloaded)} files → {raw_dir}")
    return ImportResult(raw_files=downloaded, temp_dir=temp_dir, batch_name=batch_name)
