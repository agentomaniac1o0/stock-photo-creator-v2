"""
Tests for Module 01: Importer
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.nextcloud_client import NextcloudClient
from modules.importer import ImportResult


class TestNextcloudClient(unittest.TestCase):
    """Test the Nextcloud client URL generation."""

    def setUp(self):
        self.client = NextcloudClient(
            host="https://example.com",
            user="testuser",
            password="testpass"
        )

    def test_url_generation(self):
        url = self.client._url("Photos/StockFotoCreator/RAW")
        self.assertEqual(
            url,
            "https://example.com/remote.php/dav/files/testuser/Photos/StockFotoCreator/RAW"
        )

    def test_url_with_spaces(self):
        url = self.client._url("Photos/StockFotoCreator/RAW/Barcelona Trip")
        self.assertIn("Barcelona%20Trip", url)

    def test_auth(self):
        auth = self.client._auth()
        self.assertEqual(auth, ("testuser", "testpass"))


class TestImportResult(unittest.TestCase):
    """Test the ImportResult data class."""

    @patch('modules.importer.shutil.rmtree')
    def test_cleanup(self, mock_rmtree):
        temp_dir = Path("/tmp/test_temp_importer")
        temp_dir.mkdir(parents=True, exist_ok=True)

        result = ImportResult(
            raw_files=[Path("/tmp/test/IMG_001.CR2")],
            temp_dir=temp_dir,
            batch_name="test_batch"
        )

        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.batch_name, "test_batch")

        result.cleanup()
        mock_rmtree.assert_called_once()

        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
