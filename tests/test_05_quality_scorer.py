"""
Tests for Module 05: Quality Scorer
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.quality_scorer import (
    QualityScore,
    QualityScorerResult,
    score_image_fallback,
    score_all_images,
)


class TestQualityScore(unittest.TestCase):

    def test_filename(self):
        score = QualityScore(
            filepath=Path("/tmp/IMG_001.CR2"),
            overall_score=85.0,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
            assessment="Good image",
            model_used="test",
        )
        self.assertEqual(score.filename, "IMG_001.CR2")

    def test_to_dict(self):
        score = QualityScore(
            filepath=Path("test.CR2"),
            overall_score=85.5,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
            assessment="Good image",
            model_used="test",
        )
        d = score.to_dict()
        self.assertEqual(d["filename"], "test.CR2")
        self.assertEqual(d["overall_score"], 85.5)
        self.assertEqual(d["sharpness_score"], 90.0)
        self.assertEqual(d["assessment"], "Good image")


class TestScoreImageFallback(unittest.TestCase):

    @patch('modules.quality_scorer.scipy_convolve')
    @patch('modules.quality_scorer.np.mean')
    @patch('modules.quality_scorer.np.abs')
    @patch('modules.quality_scorer.np.any')
    @patch('modules.quality_scorer.np.std')
    @patch('modules.quality_scorer.np.percentile')
    @patch('modules.quality_scorer.np.var')
    @patch('modules.quality_scorer.np.array')
    @patch('modules.quality_scorer.Image.open')
    def test_fallback_scoring_returns_valid_score(
        self, mock_open, mock_array, mock_var, mock_percentile,
        mock_std, mock_any, mock_abs, mock_mean, mock_convolve
    ):
        """Fallback scoring should return a valid QualityScore."""
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_open.return_value = mock_img

        import numpy as np
        mock_array.return_value = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_var.return_value = 50.0
        mock_percentile.return_value = 2.0
        mock_std.return_value = 30.0
        mock_mean.return_value = 10.0
        mock_abs.return_value = np.full((100, 100), 5.0)
        mock_any.return_value = False
        mock_convolve.return_value = np.full((100, 100), 10.0)

        score = score_image_fallback(Path("test.CR2"))

        self.assertIsInstance(score, QualityScore)
        self.assertGreaterEqual(score.overall_score, 0)
        self.assertLessEqual(score.overall_score, 100)
        self.assertEqual(score.model_used, "algorithmic-fallback")

    @patch('modules.quality_scorer.Image.open', side_effect=Exception("test error"))
    def test_fallback_scoring_error_returns_default(self, mock_open):
        """Fallback scoring should return default score on error."""
        score = score_image_fallback(Path("test.CR2"))

        self.assertEqual(score.overall_score, 50.0)
        self.assertEqual(score.model_used, "error")


class TestScoreAllImages(unittest.TestCase):

    def _make_group(self, group_id=1, group_type=GroupType.SINGLE) -> BracketGroup:
        return BracketGroup(
            group_type=group_type,
            files=[
                FileExifData(filepath=Path("img1.CR2"), exposure_compensation=0.0),
            ],
            group_id=group_id,
        )

    @patch('modules.quality_scorer.score_image')
    def test_scores_all_images(self, mock_score):
        """All images should be scored."""
        mock_score.return_value = QualityScore(
            filepath=Path("img1.CR2"),
            overall_score=85.0,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
            assessment="Good",
            model_used="test",
        )

        group = self._make_group()
        result = score_all_images([group])

        self.assertEqual(result.total_scored, 1)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.scores), 1)

    @patch('modules.quality_scorer.score_image')
    def test_sorts_by_score(self, mock_score):
        """Images should be sorted by score (best first)."""
        call_count = [0]
        def side_effect(filepath):
            call_count[0] += 1
            return QualityScore(
                filepath=filepath,
                overall_score=90.0 if "good" in str(filepath) else 50.0,
                sharpness_score=90.0,
                noise_score=80.0,
                detail_score=85.0,
                defect_score=95.0,
                assessment="Test",
                model_used="test",
            )

        mock_score.side_effect = side_effect

        group = BracketGroup(
            group_type=GroupType.BURST,
            files=[
                FileExifData(filepath=Path("bad.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("good.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
        )

        result = score_all_images([group])

        # Best image should be first
        self.assertEqual(result.scored_groups[0].files[0].filepath, Path("good.CR2"))

    @patch('modules.quality_scorer.score_image')
    def test_failed_scoring_counted(self, mock_score):
        """Failed scoring should be counted."""
        mock_score.return_value = QualityScore(
            filepath=Path("img1.CR2"),
            overall_score=50.0,
            sharpness_score=50.0,
            noise_score=50.0,
            detail_score=50.0,
            defect_score=50.0,
            assessment="Failed",
            model_used="error",
        )

        group = self._make_group()
        result = score_all_images([group])

        self.assertEqual(result.total_scored, 0)
        self.assertEqual(result.total_failed, 1)

    def test_empty_input(self):
        """Empty input should return empty result."""
        result = score_all_images([])

        self.assertEqual(result.total_scored, 0)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.scored_groups), 0)

    def test_summary_output(self):
        """Summary should contain meaningful information."""
        result = QualityScorerResult(
            scored_groups=[],
            scores=[QualityScore(
                filepath=Path("test.CR2"),
                overall_score=85.0,
                sharpness_score=90.0,
                noise_score=80.0,
                detail_score=85.0,
                defect_score=95.0,
                assessment="Good",
                model_used="test",
            )],
            total_scored=1,
            total_failed=0,
        )

        summary = result.summary()
        self.assertIn("85", summary)
        self.assertIn("test.CR2", summary)


if __name__ == "__main__":
    unittest.main()
