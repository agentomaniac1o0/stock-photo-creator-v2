"""
Tests for Module 05: Quality Scorer
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.quality_scorer import (
    QualityScore,
    QualityScorerResult,
    compute_exposure_score,
    compute_shadow_noise,
    score_image_fallback,
    score_all_images,
)


class TestQualityScore(unittest.TestCase):

    def test_filename(self):
        score = QualityScore(
            filepath=Path("/tmp/IMG_001.CR2"),
            overall_score=85.0,
            exposure_score=80.0,
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
            exposure_score=80.0,
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
        self.assertEqual(d["exposure_score"], 80.0)
        self.assertEqual(d["sharpness_score"], 90.0)
        self.assertEqual(d["assessment"], "Good image")


class TestExposureScore(unittest.TestCase):

    def test_well_exposed_image(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = compute_exposure_score(gray)
        self.assertGreater(score, 80)

    def test_underexposed_image(self):
        gray = np.full((100, 100), 30, dtype=np.uint8)
        score = compute_exposure_score(gray)
        self.assertLess(score, 40)

    def test_overexposed_image(self):
        gray = np.full((100, 100), 240, dtype=np.uint8)
        score = compute_exposure_score(gray)
        self.assertLess(score, 60)

    def test_ideal_midtone(self):
        gray = np.full((100, 100), 140, dtype=np.uint8)
        score = compute_exposure_score(gray)
        self.assertGreater(score, 90)

    def test_crushed_shadows(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[50:, :] = 200
        score = compute_exposure_score(gray)
        self.assertLess(score, 70)

    def test_score_range(self):
        for brightness in [0, 50, 100, 150, 200, 255]:
            gray = np.full((100, 100), brightness, dtype=np.uint8)
            score = compute_exposure_score(gray)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)


class TestShadowNoise(unittest.TestCase):

    def test_clean_image(self):
        gray = np.full((100, 100), 50, dtype=np.uint8)
        score = compute_shadow_noise(gray)
        self.assertGreater(score, 80)

    def test_noisy_image(self):
        np.random.seed(42)
        gray = np.random.randint(0, 80, (100, 100), dtype=np.uint8)
        score = compute_shadow_noise(gray)
        self.assertLess(score, 70)

    def test_no_shadows_returns_default(self):
        gray = np.full((100, 100), 200, dtype=np.uint8)
        score = compute_shadow_noise(gray)
        self.assertEqual(score, 80.0)

    def test_score_range(self):
        np.random.seed(42)
        for noise_level in [0, 10, 30, 50]:
            base = np.full((100, 100), 40, dtype=np.uint8)
            if noise_level > 0:
                noise = np.random.randint(0, noise_level, (100, 100), dtype=np.uint8)
                gray = np.clip(base.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
            else:
                gray = base
            score = compute_shadow_noise(gray)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)


class TestScoreImageFallback(unittest.TestCase):

    @patch('modules.quality_scorer.load_image_array')
    def test_fallback_scoring_returns_valid_score(self, mock_load):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_load.return_value = arr

        score = score_image_fallback(Path("test.CR2"))

        self.assertIsInstance(score, QualityScore)
        self.assertGreaterEqual(score.overall_score, 0)
        self.assertLessEqual(score.overall_score, 100)
        self.assertGreaterEqual(score.exposure_score, 0)
        self.assertLessEqual(score.exposure_score, 100)
        self.assertGreaterEqual(score.noise_score, 0)
        self.assertLessEqual(score.noise_score, 100)
        self.assertEqual(score.model_used, "algorithmic-fallback")

    @patch('modules.quality_scorer.load_image_array', side_effect=Exception("test error"))
    def test_fallback_scoring_error_returns_default(self, mock_load):
        score = score_image_fallback(Path("test.CR2"))

        self.assertEqual(score.overall_score, 50.0)
        self.assertEqual(score.exposure_score, 50.0)
        self.assertEqual(score.noise_score, 50.0)
        self.assertEqual(score.model_used, "error")

    @patch('modules.quality_scorer.load_image_array')
    def test_underexposed_gets_low_exposure_score(self, mock_load):
        arr = np.full((100, 100, 3), 30, dtype=np.uint8)
        mock_load.return_value = arr

        score = score_image_fallback(Path("test.CR2"))

        self.assertLess(score.exposure_score, 40)

    @patch('modules.quality_scorer.load_image_array')
    def test_well_exposed_gets_high_exposure_score(self, mock_load):
        arr = np.full((100, 100, 3), 140, dtype=np.uint8)
        mock_load.return_value = arr

        score = score_image_fallback(Path("test.CR2"))

        self.assertGreater(score.exposure_score, 80)

    @patch('modules.quality_scorer.load_image_array')
    def test_new_weights_applied(self, mock_load):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_load.return_value = arr

        score = score_image_fallback(Path("test.CR2"))

        expected = (
            score.exposure_score * 0.35 +
            score.noise_score * 0.30 +
            score.sharpness_score * 0.15 +
            score.detail_score * 0.10 +
            score.defect_score * 0.10
        )
        self.assertAlmostEqual(score.overall_score, expected, places=1)


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
        mock_score.return_value = QualityScore(
            filepath=Path("img1.CR2"),
            overall_score=85.0,
            exposure_score=80.0,
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
        call_count = [0]
        def side_effect(filepath):
            call_count[0] += 1
            return QualityScore(
                filepath=filepath,
                overall_score=90.0 if "good" in str(filepath) else 50.0,
                exposure_score=80.0,
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

        self.assertEqual(result.scored_groups[0].files[0].filepath, Path("good.CR2"))

    @patch('modules.quality_scorer.score_image')
    def test_failed_scoring_counted(self, mock_score):
        mock_score.return_value = QualityScore(
            filepath=Path("img1.CR2"),
            overall_score=50.0,
            exposure_score=50.0,
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
        result = score_all_images([])

        self.assertEqual(result.total_scored, 0)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.scored_groups), 0)

    def test_summary_output(self):
        result = QualityScorerResult(
            scored_groups=[],
            scores=[QualityScore(
                filepath=Path("test.CR2"),
                overall_score=85.0,
                exposure_score=80.0,
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
        self.assertIn("exp=80", summary)


if __name__ == "__main__":
    unittest.main()
