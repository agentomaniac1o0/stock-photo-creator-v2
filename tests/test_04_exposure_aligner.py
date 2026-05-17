"""
Tests for Module 04: Exposure Aligner
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.exposure_aligner import (
    ExposureCorrection,
    ExposureAlignResult,
    estimate_image_brightness,
    compute_toncurve_score,
    calculate_correction_params,
    align_exposures,
    select_reference_image,
)


class TestExposureCorrection(unittest.TestCase):

    def test_filename(self):
        correction = ExposureCorrection(
            filepath=Path("/tmp/IMG_001.CR2"),
            original_ev=-2.0,
            target_ev=0.0,
            brightness_adjustment=4.0,
            highlights_adjustment=60.0,
            shadows_adjustment=100.0,
        )
        self.assertEqual(correction.filename, "IMG_001.CR2")


class TestEstimateImageBrightness(unittest.TestCase):

    @patch('modules.exposure_aligner.Image.open')
    def test_brightness_estimation(self, mock_open):
        """Should return average luminance."""
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_open.return_value = mock_img

        with patch('modules.exposure_aligner.np.array') as mock_array:
            mock_array.return_value = np.full((100, 100, 3), 128, dtype=np.uint8)
            brightness = estimate_image_brightness(Path("test.CR2"))
            self.assertAlmostEqual(brightness, 128.0, delta=1.0)

    @patch('modules.exposure_aligner.Image.open', side_effect=Exception("test error"))
    def test_brightness_estimation_error(self, mock_open):
        """Should return default mid-gray on error."""
        brightness = estimate_image_brightness(Path("test.CR2"))
        self.assertEqual(brightness, 128.0)


class TestComputeToncurveScore(unittest.TestCase):

    @patch('modules.exposure_aligner.Image.open')
    def test_error_returns_default(self, mock_open):
        mock_open.side_effect = Exception("error")
        score = compute_toncurve_score(Path("test.CR2"))
        self.assertEqual(score, 50.0)

    @patch('modules.exposure_aligner.Image.open')
    def test_midgray_gets_high_score(self, mock_open):
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_open.return_value = mock_img

        with patch('modules.exposure_aligner.np.array') as mock_array:
            mock_array.return_value = np.full((100, 100, 3), 128, dtype=np.uint8)
            score = compute_toncurve_score(Path("test.CR2"))
            self.assertGreater(score, 70)


class TestCalculateCorrectionParams(unittest.TestCase):

    def test_underexposed_image_correction(self):
        """Image at -2 EV should need significant brightening."""
        fd = FileExifData(
            filepath=Path("dark.CR2"),
            exposure_compensation=-2.0,
            exposure_time="1/500",
        )
        ref_fd = FileExifData(
            filepath=Path("mid.CR2"),
            exposure_compensation=0.0,
            exposure_time="1/125",
        )
        correction = calculate_correction_params(fd, ref_fd)

        self.assertEqual(correction.original_ev, -2.0)
        self.assertGreater(correction.brightness_adjustment, 1.0)
        self.assertGreater(correction.highlights_adjustment, 0)
        self.assertGreater(correction.shadows_adjustment, 0)

    def test_well_exposed_image_no_correction(self):
        """Image at 0 EV with same exposure time as ref should need minimal."""
        fd = FileExifData(
            filepath=Path("mid.CR2"),
            exposure_compensation=0.0,
            exposure_time="1/125",
        )
        ref_fd = FileExifData(
            filepath=Path("ref.CR2"),
            exposure_compensation=0.0,
            exposure_time="1/125",
        )
        correction = calculate_correction_params(fd, ref_fd)

        self.assertAlmostEqual(correction.brightness_adjustment, 1.0, delta=0.01)
        self.assertEqual(correction.highlights_adjustment, 0)
        self.assertEqual(correction.shadows_adjustment, 0)


class TestSelectReferenceImage(unittest.TestCase):

    @patch('modules.exposure_aligner.compute_toncurve_score')
    def test_picks_highest_toncurve_score(self, mock_score):
        """Uses the toncurve image (0EV) as reference."""
        files = [
            FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0, exposure_time="1/500"),
            FileExifData(filepath=Path("mid.CR2"), exposure_compensation=0.0, exposure_time="1/125"),
            FileExifData(filepath=Path("bright.CR2"), exposure_compensation=2.0, exposure_time="1/30"),
        ]
        # mid.CR2 has best toncurve (highest score)
        mock_score.side_effect = lambda p: {"dark.CR2": 50, "mid.CR2": 90, "bright.CR2": 30}[p.name]

        ref = select_reference_image(files)
        self.assertEqual(ref.filepath, Path("mid.CR2"))

    @patch('modules.exposure_aligner.compute_toncurve_score')
    def test_two_images_falls_back_to_best_toncurve(self, mock_score):
        files = [
            FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-1.0, exposure_time="1/250"),
            FileExifData(filepath=Path("bright.CR2"), exposure_compensation=1.0, exposure_time="1/60"),
        ]
        mock_score.side_effect = lambda p: {"dark.CR2": 70, "bright.CR2": 40}[p.name]

        ref = select_reference_image(files)
        self.assertEqual(ref.filepath, Path("dark.CR2"))

    def test_all_equal_scores_returns_first(self):
        files = [
            FileExifData(filepath=Path("a.CR2"), exposure_compensation=-2.0, exposure_time=None),
            FileExifData(filepath=Path("b.CR2"), exposure_compensation=0.0, exposure_time=None),
            FileExifData(filepath=Path("c.CR2"), exposure_compensation=2.0, exposure_time=None),
        ]
        ref = select_reference_image(files)
        # All toncurve scores default to 50.0 (file errors), first wins
        self.assertEqual(ref.filepath, Path("a.CR2"))


class TestAlignExposures(unittest.TestCase):

    def _make_aeb_group(self, group_id=1) -> BracketGroup:
        return BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0, exposure_time="1/500"),
                FileExifData(filepath=Path("mid.CR2"), exposure_compensation=0.0, exposure_time="1/125"),
                FileExifData(filepath=Path("bright.CR2"), exposure_compensation=2.0, exposure_time="1/30"),
            ],
            group_id=group_id,
        )

    @patch('modules.exposure_aligner.compute_toncurve_score')
    @patch('modules.exposure_aligner.apply_correction')
    def test_corrects_all_non_reference_images(self, mock_apply, mock_score):
        """All non-reference images should be corrected."""
        mock_score.side_effect = lambda p: 50.0  # all equal

        mock_apply.return_value = ExposureCorrection(
            filepath=Path("dark.CR2"),
            original_ev=-2.0,
            target_ev=0.0,
            brightness_adjustment=4.0,
            highlights_adjustment=60.0,
            shadows_adjustment=100.0,
            corrected_path=Path("dark_exposure_corrected.jpg"),
            success=True,
            reason="Corrected",
        )

        group = self._make_aeb_group()
        result = align_exposures([group])

        # 2 non-reference images should be corrected
        self.assertEqual(result.total_corrected, 2)
        self.assertEqual(result.total_failed, 0)

    @patch('modules.exposure_aligner.compute_toncurve_score')
    @patch('modules.exposure_aligner.apply_correction')
    def test_failed_correction_keeps_original(self, mock_apply, mock_score):
        """If correction fails, original file should be kept."""
        mock_score.side_effect = lambda p: 50.0

        mock_apply.return_value = ExposureCorrection(
            filepath=Path("dark.CR2"),
            original_ev=-2.0,
            target_ev=0.0,
            brightness_adjustment=4.0,
            highlights_adjustment=60.0,
            shadows_adjustment=100.0,
            success=False,
            reason="Correction failed",
        )

        group = self._make_aeb_group()
        result = align_exposures([group])

        self.assertEqual(result.total_corrected, 0)
        self.assertEqual(result.total_failed, 2)

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
        result = align_exposures([group])
        self.assertEqual(result.total_corrected, 0)
        self.assertEqual(len(result.aligned_groups), 1)
        self.assertEqual(result.aligned_groups[0].file_count, 2)

    def test_single_file_passes_through(self):
        """Single files should pass through unchanged."""
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        result = align_exposures([group])
        self.assertEqual(result.total_corrected, 0)
        self.assertEqual(len(result.aligned_groups), 1)

    def test_empty_input(self):
        """Empty input should return empty result."""
        result = align_exposures([])
        self.assertEqual(result.total_corrected, 0)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.aligned_groups), 0)

    def test_summary_output(self):
        """Summary should contain meaningful information."""
        result = ExposureAlignResult(
            aligned_groups=[],
            corrections=[ExposureCorrection(
                filepath=Path("dark.CR2"),
                original_ev=-2.0,
                target_ev=0.0,
                brightness_adjustment=4.0,
                highlights_adjustment=60.0,
                shadows_adjustment=100.0,
                success=True,
            )],
            total_corrected=1,
            total_failed=0,
        )
        summary = result.summary()
        self.assertIn("1", summary)
        self.assertIn("dark.CR2", summary)


if __name__ == "__main__":
    unittest.main()
