"""
Tests for Module 02: Bracket Detector
"""
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.bracket_detector import (
    FileExifData,
    BracketGroup,
    GroupType,
    parse_exif_data,
    detect_brackets,
    print_group_summary,
)


class TestFileExifData(unittest.TestCase):

    def test_filename(self):
        fd = FileExifData(filepath=Path("/tmp/IMG_001.CR2"))
        self.assertEqual(fd.filename, "IMG_001.CR2")


class TestBracketGroup(unittest.TestCase):

    def test_single_file_group(self):
        fd = FileExifData(filepath=Path("test.CR2"), exposure_compensation=0.0)
        group = BracketGroup(group_type=GroupType.SINGLE, files=[fd], group_id=1)
        self.assertEqual(group.file_count, 1)
        self.assertFalse(group.has_exposure_variation)
        self.assertEqual(group.best_exposure_file, fd)

    def test_aeb_group_has_exposure_variation(self):
        files = [
            FileExifData(filepath=Path("a.CR2"), exposure_compensation=-2.0),
            FileExifData(filepath=Path("b.CR2"), exposure_compensation=0.0),
            FileExifData(filepath=Path("c.CR2"), exposure_compensation=2.0),
        ]
        group = BracketGroup(group_type=GroupType.AEB, files=files, group_id=1)
        self.assertTrue(group.has_exposure_variation)

    def test_burst_group_no_exposure_variation(self):
        files = [
            FileExifData(filepath=Path("a.CR2"), exposure_compensation=0.0),
            FileExifData(filepath=Path("b.CR2"), exposure_compensation=0.0),
        ]
        group = BracketGroup(group_type=GroupType.BURST, files=files, group_id=1)
        self.assertFalse(group.has_exposure_variation)

    def test_best_exposure_file(self):
        files = [
            FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0),
            FileExifData(filepath=Path("mid.CR2"), exposure_compensation=0.0),
            FileExifData(filepath=Path("bright.CR2"), exposure_compensation=2.0),
        ]
        group = BracketGroup(group_type=GroupType.AEB, files=files, group_id=1)
        best = group.best_exposure_file
        self.assertEqual(best.filename, "mid.CR2")

    def test_to_dict(self):
        files = [
            FileExifData(filepath=Path("a.CR2"), exposure_compensation=-1.0),
            FileExifData(filepath=Path("b.CR2"), exposure_compensation=0.0),
        ]
        group = BracketGroup(group_type=GroupType.AEB, files=files, group_id=1)
        d = group.to_dict()
        self.assertEqual(d["group_type"], "aeb")
        self.assertEqual(d["file_count"], 2)
        self.assertTrue(d["has_exposure_variation"])


class TestDetectBrackets(unittest.TestCase):

    @patch('modules.bracket_detector.read_exif_fields')
    def test_detect_aeb_group(self, mock_exif):
        """Three files with different EV values within 2 seconds → AEB group."""
        base_time = "2026:05:15 10:00:00"
        mock_exif.side_effect = [
            {"ExposureCompensation": "-2", "DateTimeOriginal": base_time},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:01"},
            {"ExposureCompensation": "2", "DateTimeOriginal": "2026:05:15 10:00:02"},
        ]

        files = [Path("dark.CR2"), Path("mid.CR2"), Path("bright.CR2")]
        groups = detect_brackets(files)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].group_type, GroupType.AEB)
        self.assertEqual(groups[0].file_count, 3)

    @patch('modules.bracket_detector.read_exif_fields')
    def test_detect_burst_group(self, mock_exif):
        """Three files with same EV within 2 seconds → Burst group."""
        mock_exif.side_effect = [
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:00"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:01"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:02"},
        ]

        files = [Path("shot1.CR2"), Path("shot2.CR2"), Path("shot3.CR2")]
        groups = detect_brackets(files)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].group_type, GroupType.BURST)
        self.assertEqual(groups[0].file_count, 3)

    @patch('modules.bracket_detector.read_exif_fields')
    def test_detect_single_file(self, mock_exif):
        """Single file with no nearby files → Single group."""
        mock_exif.return_value = {
            "ExposureCompensation": "0",
            "DateTimeOriginal": "2026:05:15 10:00:00",
        }

        files = [Path("solo.CR2")]
        groups = detect_brackets(files)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].group_type, GroupType.SINGLE)

    @patch('modules.bracket_detector.read_exif_fields')
    def test_detect_mixed_groups(self, mock_exif):
        """AEB group + burst group + single file, separated by time."""
        mock_exif.side_effect = [
            # AEB group at 10:00:00-02
            {"ExposureCompensation": "-2", "DateTimeOriginal": "2026:05:15 10:00:00"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:01"},
            {"ExposureCompensation": "2", "DateTimeOriginal": "2026:05:15 10:00:02"},
            # Burst group at 10:01:00-02
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:01:00"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:01:01"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:01:02"},
            # Single at 10:05:00
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:05:00"},
        ]

        files = [
            Path("aeb1.CR2"), Path("aeb2.CR2"), Path("aeb3.CR2"),
            Path("burst1.CR2"), Path("burst2.CR2"), Path("burst3.CR2"),
            Path("solo.CR2"),
        ]
        groups = detect_brackets(files)

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0].group_type, GroupType.AEB)
        self.assertEqual(groups[1].group_type, GroupType.BURST)
        self.assertEqual(groups[2].group_type, GroupType.SINGLE)

    @patch('modules.bracket_detector.read_exif_fields')
    def test_files_beyond_time_window_not_grouped(self, mock_exif):
        """Files 10 seconds apart should NOT be grouped."""
        mock_exif.side_effect = [
            {"ExposureCompensation": "-2", "DateTimeOriginal": "2026:05:15 10:00:00"},
            {"ExposureCompensation": "0", "DateTimeOriginal": "2026:05:15 10:00:10"},
        ]

        files = [Path("a.CR2"), Path("b.CR2")]
        groups = detect_brackets(files)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].group_type, GroupType.SINGLE)
        self.assertEqual(groups[1].group_type, GroupType.SINGLE)

    def test_empty_file_list(self):
        """Empty input should return empty output."""
        groups = detect_brackets([])
        self.assertEqual(len(groups), 0)


class TestPrintGroupSummary(unittest.TestCase):

    def test_print_summary(self):
        files = [
            FileExifData(filepath=Path("a.CR2"), exposure_compensation=-1.0),
            FileExifData(filepath=Path("b.CR2"), exposure_compensation=0.0),
        ]
        group = BracketGroup(group_type=GroupType.AEB, files=files, group_id=1)
        summary = print_group_summary([group])
        self.assertIn("AEB group", summary)
        self.assertIn("a.CR2", summary)
        self.assertIn("b.CR2", summary)


if __name__ == "__main__":
    unittest.main()
