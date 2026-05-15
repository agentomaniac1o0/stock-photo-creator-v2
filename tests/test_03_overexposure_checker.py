"""
Tests for Module 03: Overexposure Checker
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.overexposure_checker import (
    ClippingResult,
    OverexposureCheckResult,
    analyze_clipping,
    check_overexposure,
)


class TestClippingResult(unittest.TestCase):

    def test_filename(self):
        result = ClippingResult(
            filepath=Path("/tmp/IMG_001.CR2"),
            is_clipped=True,
            clipping_ratio=0.05,
            highlight_detail_ratio=0.0,
            recoverable=False,
        )
        self.assertEqual(result.filename, "IMG_001.CR2")


class TestAnalyzeClipping(unittest.TestCase):

    def _create_test_image(self, path: Path, content: np.ndarray):
        """Helper to create a test JPEG image."""
        img = Image.fromarray(content.astype(np.uint8))
        img.save(path, "JPEG")

    def test_heavily_clipped_image(self):
        """Image with >2% pixels at max brightness → unrecoverable."""
        # Create a 100x100 image where 10% of pixels are white (255)
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        arr[:10, :] = 255  # Top 10 rows = white (10% of pixels)

        with patch('modules.overexposure_checker.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.mode = "RGB"
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            # We need to patch np.array too since Image.open returns a mock
            with patch('modules.overexposure_checker.np.array') as mock_array:
                mock_array.return_value = arr

                result = analyze_clipping(Path("test.CR2"))

                self.assertTrue(result.is_clipped)
                self.assertFalse(result.recoverable)
                self.assertGreater(result.clipping_ratio, 0.02)

    def test_clean_image(self):
        """Image with no clipping → recoverable."""
        # Create a 100x100 image with mid-range values
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)

        with patch('modules.overexposure_checker.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_img.mode = "RGB"
            mock_img.size = (100, 100)
            mock_open.return_value = mock_img

            with patch('modules.overexposure_checker.np.array') as mock_array:
                mock_array.return_value = arr

                result = analyze_clipping(Path("test.CR2"))

                self.assertFalse(result.is_clipped)
                self.assertTrue(result.recoverable)
                self.assertEqual(result.clipping_ratio, 0.0)

    def test_analysis_error(self):
        """If analysis fails, image should be passed through (not rejected)."""
        with patch('modules.overexposure_checker.Image.open', side_effect=Exception("test error")):
            result = analyze_clipping(Path("test.CR2"))

            self.assertFalse(result.is_clipped)
            self.assertTrue(result.recoverable)
            self.assertIn("Analysis error", result.reason)


class TestCheckOverexposure(unittest.TestCase):

    def _make_aeb_group(self, group_id=1) -> BracketGroup:
        return BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("bright.CR2"), exposure_compensation=2.0),
                FileExifData(filepath=Path("mid.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0),
            ],
            group_id=group_id,
        )

    @patch('modules.overexposure_checker.analyze_clipping')
    def test_unrecoverable_clipping_removes_brightest(self, mock_analyze):
        """If brightest image is unrecoverable, it should be removed."""
        mock_analyze.return_value = ClippingResult(
            filepath=Path("bright.CR2"),
            is_clipped=True,
            clipping_ratio=0.15,
            highlight_detail_ratio=0.0,
            recoverable=False,
            reason="Unrecoverable clipping",
        )

        group = self._make_aeb_group()
        result = check_overexposure([group])

        self.assertEqual(result.total_rejected, 1)
        self.assertEqual(len(result.checked_groups), 1)

        # Remaining group should have 2 files (mid + dark)
        remaining_group = result.checked_groups[0]
        self.assertEqual(remaining_group.file_count, 2)
        self.assertNotIn(Path("bright.CR2"), [f.filepath for f in remaining_group.files])

    @patch('modules.overexposure_checker.analyze_clipping')
    def test_recoverable_clipping_keeps_group(self, mock_analyze):
        """If clipping is recoverable, group should be kept as-is."""
        mock_analyze.return_value = ClippingResult(
            filepath=Path("bright.CR2"),
            is_clipped=True,
            clipping_ratio=0.03,
            highlight_detail_ratio=0.5,
            recoverable=True,
            reason="Clipping but recoverable",
        )

        group = self._make_aeb_group()
        result = check_overexposure([group])

        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(len(result.checked_groups), 1)
        self.assertEqual(result.checked_groups[0].file_count, 3)

    @patch('modules.overexposure_checker.analyze_clipping')
    def test_no_clipping_keeps_group(self, mock_analyze):
        """If no clipping, group should be kept as-is."""
        mock_analyze.return_value = ClippingResult(
            filepath=Path("bright.CR2"),
            is_clipped=False,
            clipping_ratio=0.0,
            highlight_detail_ratio=0.0,
            recoverable=True,
            reason="No clipping",
        )

        group = self._make_aeb_group()
        result = check_overexposure([group])

        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(result.checked_groups[0].file_count, 3)

    def test_burst_group_passes_through(self):
        """Burst groups should pass through unchanged."""
        group = BracketGroup(
            group_type=GroupType.BURST,
            files=[
                FileExifData(filepath=Path("shot1.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("shot2.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
        )

        result = check_overexposure([group])

        self.assertEqual(result.total_checked, 0)
        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(len(result.checked_groups), 1)
        self.assertEqual(result.checked_groups[0].file_count, 2)

    def test_single_file_passes_through(self):
        """Single files should pass through unchanged."""
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )

        result = check_overexposure([group])

        self.assertEqual(result.total_checked, 0)
        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(len(result.checked_groups), 1)

    @patch('modules.overexposure_checker.analyze_clipping')
    def test_empty_group_after_rejection(self, mock_analyze):
        """If all files in a group are rejected, group should be removed."""
        mock_analyze.return_value = ClippingResult(
            filepath=Path("bright.CR2"),
            is_clipped=True,
            clipping_ratio=0.15,
            highlight_detail_ratio=0.0,
            recoverable=False,
            reason="Unrecoverable clipping",
        )

        # AEB group with only 1 file (edge case)
        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[FileExifData(filepath=Path("only.CR2"), exposure_compensation=2.0)],
            group_id=1,
        )

        result = check_overexposure([group])

        self.assertEqual(result.total_rejected, 1)
        self.assertEqual(len(result.checked_groups), 0)

    def test_empty_input(self):
        """Empty input should return empty result."""
        result = check_overexposure([])

        self.assertEqual(result.total_checked, 0)
        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(len(result.checked_groups), 0)

    @patch('modules.overexposure_checker.analyze_clipping')
    def test_remaining_single_becomes_single_type(self, mock_analyze):
        """If AEB group has 2 files and brightest is rejected, remaining becomes Single."""
        mock_analyze.return_value = ClippingResult(
            filepath=Path("bright.CR2"),
            is_clipped=True,
            clipping_ratio=0.15,
            highlight_detail_ratio=0.0,
            recoverable=False,
        )

        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("bright.CR2"), exposure_compensation=2.0),
                FileExifData(filepath=Path("mid.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
        )

        result = check_overexposure([group])

        self.assertEqual(len(result.checked_groups), 1)
        self.assertEqual(result.checked_groups[0].group_type, GroupType.SINGLE)
        self.assertEqual(result.checked_groups[0].file_count, 1)

    def test_summary_output(self):
        """Summary should contain meaningful information."""
        result = OverexposureCheckResult(
            checked_groups=[],
            rejected_files=[(Path("test.CR2"), "Clipped")],
            total_checked=5,
            total_rejected=1,
        )

        summary = result.summary()
        self.assertIn("5", summary)
        self.assertIn("1", summary)
        self.assertIn("test.CR2", summary)


if __name__ == "__main__":
    unittest.main()
