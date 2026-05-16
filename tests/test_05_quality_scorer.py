"""
Tests for Module 05: Quality Scorer (v2 — Metric-Only)
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.quality_scorer import (
    ImageMetrics,
    MetricsResult,
    compute_exposure_score,
    compute_shadow_noise,
    compute_tenengrad,
    compute_combined_sharpness,
    noise_curve,
    sharpness_curve,
    exposure_correction_penalty,
    compute_metrics,
    compute_all_metrics,
)


class TestImageMetrics(unittest.TestCase):

    def test_filename(self):
        m = ImageMetrics(
            filepath=Path("/tmp/IMG_001.CR2"),
            exposure_score=80.0,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
        )
        self.assertEqual(m.filename, "IMG_001.CR2")

    def test_to_dict(self):
        m = ImageMetrics(
            filepath=Path("test.CR2"),
            exposure_score=80.0,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
        )
        d = m.to_dict()
        self.assertEqual(d["filename"], "test.CR2")
        self.assertEqual(d["exposure_score"], 80.0)
        self.assertEqual(d["sharpness_score"], 90.0)
        self.assertNotIn("overall_score", d)


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


class TestTenengrad(unittest.TestCase):

    def test_sharp_image_high_score(self):
        gray = np.zeros((100, 100), dtype=np.float64)
        gray[:, 50] = 255
        score = compute_tenengrad(gray.astype(np.uint8))
        self.assertGreater(score, 20)

    def test_blurry_image_low_score(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        score = compute_tenengrad(gray)
        self.assertLess(score, 10)

    def test_noise_vs_sharp_edges(self):
        np.random.seed(42)
        noisy = np.random.randint(100, 156, (100, 100), dtype=np.uint8)
        gray = np.full((100, 100), 128, dtype=np.uint8)
        gray[:, 50] = 255
        noisy_score = compute_tenengrad(noisy)
        edge_score = compute_tenengrad(gray)
        self.assertGreater(edge_score, noisy_score)

    def test_score_range(self):
        for brightness in [0, 50, 100, 150, 200, 255]:
            gray = np.full((100, 100), brightness, dtype=np.uint8)
            score = compute_tenengrad(gray)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)


class TestCombinedSharpness(unittest.TestCase):

    def test_combined_weights_tenengrad_higher(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        noise_score = 80.0
        score = compute_combined_sharpness(gray, noise_score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    @patch('modules.quality_scorer.load_image_array')
    def test_sharp_clean_image_gets_high_score(self, mock_load):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        arr[40:60, 40:60, :] = 200
        mock_load.return_value = arr

        m = compute_metrics(Path("test.CR2"))
        self.assertGreater(m.sharpness_score, 5)

    @patch('modules.quality_scorer.load_image_array')
    def test_flat_blurry_image_gets_low_score(self, mock_load):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_load.return_value = arr

        m = compute_metrics(Path("test.CR2"))
        self.assertLess(m.sharpness_score, 15)


class TestNonLinearCurves(unittest.TestCase):

    def test_noise_curve_heavy_noise(self):
        self.assertAlmostEqual(noise_curve(30), 15.0)

    def test_noise_curve_moderate(self):
        self.assertAlmostEqual(noise_curve(50), 50.0)

    def test_noise_curve_clean_diminishing(self):
        self.assertAlmostEqual(noise_curve(90), 80.0)

    def test_noise_curve_boundary_40(self):
        self.assertAlmostEqual(noise_curve(40), 40.0)

    def test_noise_curve_boundary_70(self):
        self.assertAlmostEqual(noise_curve(70), 70.0)

    def test_sharpness_curve_blurry_penalty(self):
        self.assertAlmostEqual(sharpness_curve(10), 5.0)

    def test_sharpness_curve_acceptable(self):
        self.assertAlmostEqual(sharpness_curve(25), 25.0)

    def test_sharpness_curve_sharp_bonus(self):
        self.assertAlmostEqual(sharpness_curve(50), 57.5)

    def test_sharpness_curve_boundary_15(self):
        self.assertAlmostEqual(sharpness_curve(15), 15.0)

    def test_sharpness_curve_boundary_35(self):
        self.assertAlmostEqual(sharpness_curve(35), 35.0)

    def test_exposure_penalty_corrected_file(self):
        self.assertAlmostEqual(exposure_correction_penalty(Path("IMG_1567_exposure_corrected.jpg")), 5.0)

    def test_exposure_penalty_raw_file(self):
        self.assertAlmostEqual(exposure_correction_penalty(Path("IMG_1567.CR2")), 0.0)


class TestComputeMetrics(unittest.TestCase):

    @patch('modules.quality_scorer.load_image_array')
    def test_compute_metrics_returns_valid(self, mock_load):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_load.return_value = arr

        m = compute_metrics(Path("test.CR2"))

        self.assertIsInstance(m, ImageMetrics)
        self.assertGreaterEqual(m.exposure_score, 0)
        self.assertLessEqual(m.exposure_score, 100)
        self.assertGreaterEqual(m.noise_score, 0)
        self.assertLessEqual(m.noise_score, 100)
        self.assertGreaterEqual(m.sharpness_score, 0)
        self.assertLessEqual(m.sharpness_score, 100)

    @patch('modules.quality_scorer.load_image_array', side_effect=Exception("test error"))
    def test_compute_metrics_error_returns_default(self, mock_load):
        m = compute_metrics(Path("test.CR2"))

        self.assertEqual(m.exposure_score, 50.0)
        self.assertEqual(m.noise_score, 50.0)
        self.assertEqual(m.sharpness_score, 50.0)


class TestComputeAllMetrics(unittest.TestCase):

    def _make_group(self, group_id=1, group_type=GroupType.SINGLE) -> BracketGroup:
        return BracketGroup(
            group_type=group_type,
            files=[
                FileExifData(filepath=Path("img1.CR2"), exposure_compensation=0.0),
            ],
            group_id=group_id,
        )

    @patch('modules.quality_scorer.compute_metrics')
    def test_computes_all_metrics(self, mock_compute):
        mock_compute.return_value = ImageMetrics(
            filepath=Path("img1.CR2"),
            exposure_score=80.0,
            sharpness_score=90.0,
            noise_score=80.0,
            detail_score=85.0,
            defect_score=95.0,
        )

        group = self._make_group()
        result = compute_all_metrics([group])

        self.assertEqual(result.total_scored, 1)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.metrics), 1)

    @patch('modules.quality_scorer.compute_metrics')
    def test_sorts_by_noise_score(self, mock_compute):
        call_count = [0]
        def side_effect(filepath):
            call_count[0] += 1
            return ImageMetrics(
                filepath=filepath,
                exposure_score=80.0,
                sharpness_score=90.0 if "good" in str(filepath) else 50.0,
                noise_score=80.0 if "good" in str(filepath) else 40.0,
                detail_score=85.0,
                defect_score=95.0,
            )

        mock_compute.side_effect = side_effect

        group = BracketGroup(
            group_type=GroupType.BURST,
            files=[
                FileExifData(filepath=Path("bad.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("good.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
        )

        result = compute_all_metrics([group])

        self.assertEqual(result.scored_groups[0].files[0].filepath, Path("good.CR2"))

    def test_empty_input(self):
        result = compute_all_metrics([])

        self.assertEqual(result.total_scored, 0)
        self.assertEqual(result.total_failed, 0)
        self.assertEqual(len(result.scored_groups), 0)

    def test_summary_output(self):
        result = MetricsResult(
            scored_groups=[],
            metrics=[ImageMetrics(
                filepath=Path("test.CR2"),
                exposure_score=80.0,
                sharpness_score=90.0,
                noise_score=80.0,
                detail_score=85.0,
                defect_score=95.0,
            )],
            total_scored=1,
            total_failed=0,
        )

        summary = result.summary()
        self.assertIn("test.CR2", summary)
        self.assertIn("exp=80", summary)


if __name__ == "__main__":
    unittest.main()
