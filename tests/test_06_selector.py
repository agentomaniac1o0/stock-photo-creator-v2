"""
Tests for Module 06: Selector + Upload (v3 — Low-Light, DRC, Tie-Aware)
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
    detect_low_light_scene,
    compute_comparison_score,
    compute_drc_bonus,
    select_best_in_group,
    select_from_aeb_group,
    select_from_burst_group,
    select_from_single,
    SHARPNESS_GATE_MIN,
    LOW_LIGHT_SHARPNESS_GATE,
    NOISE_GATE_MIN,
    LOW_LIGHT_NOISE_GATE,
    DRC_NO_NEED_BONUS,
    DRC_SUCCESS_BONUS,
    DRC_FAIL_PENALTY,
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
            exposure_score=85.0, noise_score=80.0,
            sharpness_score=70.0, detail_score=75.0, defect_score=90.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertTrue(passes)

    def test_fails_blurry_normal(self):
        m = ImageMetrics(
            filepath=Path("blurry.CR2"),
            exposure_score=70.0, noise_score=70.0,
            sharpness_score=5.0, detail_score=60.0, defect_score=80.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertFalse(passes)
        self.assertIn("normal", reason)

    def test_passes_blurry_in_low_light(self):
        m = ImageMetrics(
            filepath=Path("blurry_indoors.CR2"),
            exposure_score=30.0, noise_score=50.0,
            sharpness_score=6.0, detail_score=60.0, defect_score=70.0,
        )
        # Normal: sharp=6 < 10 → fail
        passes_normal, _ = passes_hard_filters(m, low_light=False)
        self.assertFalse(passes_normal)

        # Low-light: sharp=6 ≥ 5 → pass
        passes_ll, _ = passes_hard_filters(m, low_light=True)
        self.assertTrue(passes_ll)

    def test_passes_at_boundary_normal(self):
        m = ImageMetrics(
            filepath=Path("boundary.CR2"),
            exposure_score=70.0, noise_score=70.0,
            sharpness_score=10.0, detail_score=60.0, defect_score=80.0,
        )
        passes, reason = passes_hard_filters(m)
        self.assertTrue(passes)


class TestDetectLowLight(unittest.TestCase):

    def test_high_iso_triggers(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(
                filepath=Path("test.CR2"), exposure_compensation=0.0,
                iso="3200")],
            group_id=1,
        )
        is_ll, reason = detect_low_light_scene(group, [])
        self.assertTrue(is_ll)
        self.assertIn("ISO", reason)

    def test_low_exposure_triggers(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(
                filepath=Path("test.CR2"), exposure_compensation=0.0,
                iso="100")],
            group_id=1,
        )
        metrics = [ImageMetrics(
            filepath=Path("test.CR2"),
            exposure_score=25.0, noise_score=50.0,
            sharpness_score=40.0, detail_score=50.0, defect_score=70.0,
        )]
        is_ll, reason = detect_low_light_scene(group, metrics)
        self.assertTrue(is_ll)
        self.assertIn("exposure", reason)

    def test_normal_not_low_light(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(
                filepath=Path("test.CR2"), exposure_compensation=0.0,
                iso="200")],
            group_id=1,
        )
        metrics = [ImageMetrics(
            filepath=Path("test.CR2"),
            exposure_score=80.0, noise_score=70.0,
            sharpness_score=60.0, detail_score=50.0, defect_score=80.0,
        )]
        is_ll, reason = detect_low_light_scene(group, metrics)
        self.assertFalse(is_ll)


class TestDrcBonus(unittest.TestCase):

    def test_no_drc_needed(self):
        fd = FileExifData(filepath=Path("test.CR2"), drc_applied=False, drc_success=False)
        self.assertEqual(compute_drc_bonus(fd), DRC_NO_NEED_BONUS)

    def test_drc_success(self):
        fd = FileExifData(filepath=Path("test_drc.jpg"), drc_applied=True, drc_success=True)
        self.assertEqual(compute_drc_bonus(fd), DRC_SUCCESS_BONUS)

    def test_drc_fail(self):
        fd = FileExifData(filepath=Path("test.CR2"), drc_applied=True, drc_success=False)
        self.assertEqual(compute_drc_bonus(fd), -DRC_FAIL_PENALTY)


class TestComparisonScore(unittest.TestCase):

    def test_high_noise_wins(self):
        fd = FileExifData(filepath=Path("clean.CR2"))
        m1 = ImageMetrics(
            filepath=Path("clean.CR2"), exposure_score=100,
            noise_score=80, sharpness_score=40, detail_score=35, defect_score=60,
        )
        m2 = ImageMetrics(
            filepath=Path("noisy.CR2"), exposure_score=100,
            noise_score=50, sharpness_score=50, detail_score=35, defect_score=60,
        )
        score1 = compute_comparison_score(m1, fd)
        score2 = compute_comparison_score(m2, fd)
        self.assertGreater(score1[0], score2[0])

    def test_exposure_as_4th_tier(self):
        fd = FileExifData(filepath=Path("test.CR2"))
        m1 = ImageMetrics(
            filepath=Path("test.CR2"), exposure_score=95,
            noise_score=70, sharpness_score=30, detail_score=30, defect_score=60,
        )
        m2 = ImageMetrics(
            filepath=Path("test2.CR2"), exposure_score=60,
            noise_score=70, sharpness_score=30, detail_score=30, defect_score=60,
        )
        score1 = compute_comparison_score(m1, fd)
        score2 = compute_comparison_score(m2, fd)
        # noise+defects+sharpness identical → exposure_score decides
        self.assertEqual(score1[0], score2[0])
        self.assertEqual(score1[1], score2[1])
        self.assertEqual(score1[2], score2[2])
        self.assertGreater(score1[3], score2[3])

    def test_drc_bonus_applied(self):
        fd_no_drc = FileExifData(filepath=Path("clean.CR2"))
        fd_drc = FileExifData(
            filepath=Path("drc.jpg"), drc_applied=True, drc_success=True)

        m = ImageMetrics(
            filepath=Path("test.CR2"), exposure_score=80,
            noise_score=60, sharpness_score=30, detail_score=30, defect_score=60,
        )
        score_no = compute_comparison_score(m, fd_no_drc)
        score_drc = compute_comparison_score(m, fd_drc)
        # DRC success bonus = +5 vs no-drc bonus = +10
        # Wait: fd_no_drc: drc_applied=False → DRC_NO_NEED_BONUS=+10
        # fd_drc: drc_applied=True, drc_success=True → DRC_SUCCESS_BONUS=+5
        self.assertGreater(score_no[0], score_drc[0])

    def test_penalty_applied_to_corrected(self):
        fd = FileExifData(filepath=Path("IMG_1567_exposure_corrected.jpg"))
        m = ImageMetrics(
            filepath=Path("IMG_1567_exposure_corrected.jpg"), exposure_score=100,
            noise_score=80, sharpness_score=40, detail_score=35, defect_score=60,
        )
        score = compute_comparison_score(m, fd)
        # noise_curve(80) = 75, minus penalty 5 = 70, + DRC_NO_NEED_BONUS 10 = 80
        self.assertAlmostEqual(score[0], 80.0)

    def test_return_tuple_length(self):
        fd = FileExifData(filepath=Path("test.CR2"))
        m = ImageMetrics(
            filepath=Path("test.CR2"), exposure_score=80,
            noise_score=60, sharpness_score=30, detail_score=30, defect_score=60,
        )
        score = compute_comparison_score(m, fd)
        self.assertEqual(len(score), 4)


class TestSelectBestInGroup(unittest.TestCase):

    def test_selects_best_noise(self):
        files = [
            FileExifData(filepath=Path("best.CR2"), exposure_compensation=0.0),
            FileExifData(filepath=Path("worse.CR2"), exposure_compensation=0.0),
        ]
        metrics = [
            ImageMetrics(filepath=Path("best.CR2"), exposure_score=100,
                         noise_score=80, sharpness_score=40, detail_score=50, defect_score=80),
            ImageMetrics(filepath=Path("worse.CR2"), exposure_score=100,
                         noise_score=30, sharpness_score=50, detail_score=50, defect_score=80),
        ]
        best_fd, best_m, tie = select_best_in_group(files, metrics)
        self.assertEqual(best_fd.filename, "best.CR2")
        self.assertEqual(tie, "")

    def test_tie_detected(self):
        files = [
            FileExifData(filepath=Path("img1.CR2"), exposure_compensation=0.0),
            FileExifData(filepath=Path("img2.CR2"), exposure_compensation=0.0),
        ]
        metrics = [
            ImageMetrics(filepath=Path("img1.CR2"), exposure_score=100,
                         noise_score=80, sharpness_score=40, detail_score=50, defect_score=80),
            ImageMetrics(filepath=Path("img2.CR2"), exposure_score=100,
                         noise_score=79, sharpness_score=41, detail_score=50, defect_score=80),
        ]
        best_fd, best_m, tie = select_best_in_group(files, metrics)
        self.assertIn("TIE", tie)


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

    def test_rejects_drc_failures(self):
        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[
                FileExifData(filepath=Path("good.CR2"), exposure_compensation=0.0),
                FileExifData(filepath=Path("drc_fail.CR2"), exposure_compensation=0.0,
                             drc_applied=True, drc_success=False),
            ],
            group_id=1,
        )
        metrics = [
            ImageMetrics(filepath=Path("good.CR2"), exposure_score=100, noise_score=80, sharpness_score=40, detail_score=50, defect_score=70),
            ImageMetrics(filepath=Path("drc_fail.CR2"), exposure_score=80, noise_score=50, sharpness_score=30, detail_score=50, defect_score=60),
        ]
        kept, decisions = select_from_aeb_group(group, metrics)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], Path("good.CR2"))
        drc_rejects = [d for d in decisions if "DRC" in d.reason]
        self.assertEqual(len(drc_rejects), 1)

    def test_empty_group(self):
        group = BracketGroup(
            group_type=GroupType.AEB, files=[], group_id=1)
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
            group_id=1, is_action_sequence=True,
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
        metrics = [ImageMetrics(filepath=Path("solo.CR2"), exposure_score=100, noise_score=60, sharpness_score=40, detail_score=75, defect_score=75)]
        kept, decisions = select_from_single(group, metrics)
        self.assertEqual(len(kept), 1)
        self.assertEqual(decisions[0].decision, "keep")

    def test_rejects_too_noisy_single(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("noisy.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        metrics = [ImageMetrics(filepath=Path("noisy.CR2"), exposure_score=100, noise_score=25, sharpness_score=40, detail_score=75, defect_score=75)]
        kept, decisions = select_from_single(group, metrics)
        self.assertEqual(len(kept), 0)
        self.assertEqual(decisions[0].decision, "reject")

    def test_keeps_noisy_but_low_light(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("church.CR2"), exposure_compensation=0.0, iso="3200")],
            group_id=1,
        )
        metrics = [ImageMetrics(filepath=Path("church.CR2"), exposure_score=30, noise_score=20, sharpness_score=30, detail_score=50, defect_score=60)]
        # Normal: noise=20 < 30 → reject
        kept_normal, _ = select_from_single(group, metrics, low_light=False)
        self.assertEqual(len(kept_normal), 0)
        # Low-light: noise=20 ≥ 15 → keep
        kept_ll, decisions_ll = select_from_single(group, metrics, low_light=True)
        self.assertEqual(len(kept_ll), 1)
        self.assertIn("Passed filters", decisions_ll[0].reason)

    def test_rejects_drc_fail(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("fail.CR2"), exposure_compensation=0.0,
                                drc_applied=True, drc_success=False)],
            group_id=1,
        )
        metrics = [ImageMetrics(filepath=Path("fail.CR2"), exposure_score=60, noise_score=50, sharpness_score=30, detail_score=50, defect_score=60)]
        kept, decisions = select_from_single(group, metrics)
        self.assertEqual(len(kept), 0)
        self.assertIn("DRC", decisions[0].reason)

    def test_empty_group(self):
        group = BracketGroup(
            group_type=GroupType.SINGLE, files=[], group_id=1)
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
            upload_success=2, upload_failed=0,
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
