"""
Tests for Module 06: Selector + Upload (v2 — Multi-Stage Filter Pipeline)
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.quality_scorer import ImageMetrics
from modules.selector import (
    SelectionDecision,
    SelectionResult,
    passes_hard_filters,
    compute_comparison_score,
    select_from_aeb_group,
    select_from_burst_group,
    select_from_single,
    SHARPNESS_GATE_MIN,
    NOISE_GATE_MIN,
)


class TestSelectionDecision(unittest.TestCase):

    def test_decision_keep(self):
        d = SelectionDecision(
            filepath=Path("good.CR2"),
            decision="keep",
            reason="Best quality",
        )
        self.assertEqual(d.decision, "keep")


class TestHardFilters(unittest.TestCase):

    def test_passes_good_image(self):
        m = ImageMetrics(
            filepath=Path("good.CR2"),
            exposure_score=85.0,
            noise_score=80.0,
            sharpness_score=70.0,
            detail_score=75.0,
            defect_score=90.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertTrue(passes)

    def test_fails_blurry(self):
        m = ImageMetrics(
            filepath=Path("blurry.CR2"),
            exposure_score=70.0,
            noise_score=70.0,
            sharpness_score=5.0,
            detail_score=60.0,
            defect_score=80.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertFalse(passes)
        self.assertIn("Too blurry", reason)

    def test_passes_at_boundary(self):
        m = ImageMetrics(
            filepath=Path("boundary.CR2"),
            exposure_score=70.0,
            noise_score=70.0,
            sharpness_score=10.0,
            detail_score=60.0,
            defect_score=80.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertTrue(passes)


class TestComparisonScore(unittest.TestCase):

    def test_high_noise_wins(self):
        m1 = ImageMetrics(
            filepath=Path("clean.CR2"),
            exposure_score=100.0,
            noise_score=80.0,
            sharpness_score=40.0,
            detail_score=35.0,
            defect_score=60.0,
        )
        m2 = ImageMetrics(
            filepath=Path("noisy.CR2"),
            exposure_score=100.0,
            noise_score=50.0,
            sharpness_score=50.0,
            detail_score=35.0,
            defect_score=60.0,
        )
        score1 = compute_comparison_score(m1)
        score2 = compute_comparison_score(m2)
        self.assertGreater(score1[0], score2[0])

    def test_penalty_applied_to_corrected(self):
        m = ImageMetrics(
            filepath=Path("IMG_1567_exposure_corrected.jpg"),
            exposure_score=100.0,
            noise_score=80.0,
            sharpness_score=40.0,
            detail_score=35.0,
            defect_score=60.0,
        )
        score = compute_comparison_score(m)
        # noise_curve(80) = 70 + (80-70)*0.5 = 75, minus penalty 5 = 70
        self.assertAlmostEqual(score[0], 70.0)


class TestSelectFromAebGroup(unittest.TestCase):

    def _make_aeb_group(self) -> BracketGroup:
        return BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("best.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("mid.CR2"), exposure_compensation=-1.0),
                FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0),
            ],
            group_id=1,
        )

    def test_keeps_best_noise(self):
        group = self._make_aeb_group()
        metrics = [
            ImageMetrics(filepath=Path("best.CR2"), exposure_score=100, noise_score=80, sharpness_score=40, detail_score=90, defect_score=90),
            ImageMetrics(filepath=Path("mid.CR2"), exposure_score=100, noise_score=50, sharpness_score=50, detail_score=75, defect_score=75),
            ImageMetrics(filepath=Path("dark.CR2"), exposure_score=100, noise_score=60, sharpness_score=30, detail_score=60, defect_score=60),
        ]

        kept, decisions = select_from_aeb_group(group, metrics)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], Path("best.CR2"))

    def test_rejects_all_too_blurry(self):
        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("blurry1.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("blurry2.CR2"), exposure_compensation=-1.0),
            ],
            group_id=1,
        )
        metrics = [
            ImageMetrics(filepath=Path("blurry1.CR2"), exposure_score=100, noise_score=80, sharpness_score=5, detail_score=60, defect_score=70),
            ImageMetrics(filepath=Path("blurry2.CR2"), exposure_score=100, noise_score=70, sharpness_score=3, detail_score=55, defect_score=65),
        ]

        kept, decisions = select_from_aeb_group(group, metrics)

        self.assertEqual(len(kept), 0)
        self.assertTrue(all(d.decision == "reject" for d in decisions))

    def test_empty_group(self):
        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[],
            group_id=1,
        )
        kept, decisions = select_from_aeb_group(group, [])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(decisions), 0)


class TestSelectFromBurstGroup(unittest.TestCase):

    def _make_burst_group(self) -> BracketGroup:
        return BracketGroup(
            group_type=GroupType.BURST,
            files=[
                FileExifData(filepath=Path("good1.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("good2.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("bad.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
        )

    def test_keeps_best_noise_normal_burst(self):
        group = self._make_burst_group()
        metrics = [
            ImageMetrics(filepath=Path("good1.CR2"), exposure_score=100, noise_score=80, sharpness_score=40, detail_score=80, defect_score=80),
            ImageMetrics(filepath=Path("good2.CR2"), exposure_score=100, noise_score=50, sharpness_score=50, detail_score=70, defect_score=70),
            ImageMetrics(filepath=Path("bad.CR2"), exposure_score=100, noise_score=20, sharpness_score=5, detail_score=40, defect_score=40),
        ]

        kept, decisions = select_from_burst_group(group, metrics)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], Path("good1.CR2"))

    def test_keeps_all_action_sequence(self):
        group = BracketGroup(
            group_type=GroupType.BURST,
            files=[
                FileExifData(filepath=Path("action1.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("action2.CR2"), exposure_compensation=0.0),
            ],
            group_id=1,
            is_action_sequence=True,
        )
        metrics = [
            ImageMetrics(filepath=Path("action1.CR2"), exposure_score=100, noise_score=80, sharpness_score=40, detail_score=80, defect_score=80),
            ImageMetrics(filepath=Path("action2.CR2"), exposure_score=100, noise_score=70, sharpness_score=45, detail_score=70, defect_score=70),
        ]

        kept, decisions = select_from_burst_group(group, metrics)

        self.assertEqual(len(kept), 2)


class TestSelectFromSingle(unittest.TestCase):

    def test_keeps_good_single(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        metrics = [
            ImageMetrics(filepath=Path("solo.CR2"), exposure_score=100, noise_score=60, sharpness_score=40, detail_score=75, defect_score=75),
        ]

        kept, decisions = select_from_single(group, metrics)

        self.assertEqual(len(kept), 1)
        self.assertEqual(decisions[0].decision, "keep")

    def test_rejects_too_noisy_single(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("noisy.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        metrics = [
            ImageMetrics(filepath=Path("noisy.CR2"), exposure_score=100, noise_score=25, sharpness_score=40, detail_score=75, defect_score=75),
        ]

        kept, decisions = select_from_single(group, metrics)

        self.assertEqual(len(kept), 0)
        self.assertEqual(decisions[0].decision, "reject")

    def test_empty_group(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[],
            group_id=1,
        )
        kept, decisions = select_from_single(group, [])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(decisions), 0)


class TestSelectionResult(unittest.TestCase):

    def test_summary_output(self):
        result = SelectionResult(
            decisions=[
                SelectionDecision(filepath=Path("kept.CR2"), decision="keep", reason="Good"),
                SelectionDecision(filepath=Path("rejected.CR2"), decision="reject", reason="Bad"),
            ],
            kept_files=[Path("kept.CR2")],
            rejected_files=[Path("rejected.CR2")],
            upload_success=2,
            upload_failed=0,
        )

        summary = result.summary()
        self.assertIn("1", summary)
        self.assertIn("kept.CR2", summary)
        self.assertIn("rejected.CR2", summary)

    def test_counts(self):
        result = SelectionResult(
            decisions=[],
            kept_files=[Path("a.CR2"), Path("b.CR2")],
            rejected_files=[Path("c.CR2")],
        )
        self.assertEqual(result.kept_count, 2)
        self.assertEqual(result.rejected_count, 1)


if __name__ == "__main__":
    unittest.main()
