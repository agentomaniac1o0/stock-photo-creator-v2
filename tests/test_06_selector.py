"""
Tests for Module 06: Selector + Upload
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.quality_scorer import QualityScore, QualityScorerResult
from modules.selector import (
    SelectionDecision,
    SelectionResult,
    select_from_aeb_group,
    select_from_burst_group,
    select_from_single,
    select_and_upload,
)


class TestSelectionDecision(unittest.TestCase):

    def test_decision_keep(self):
        d = SelectionDecision(
            filepath=Path("good.CR2"),
            score=85.0,
            decision="keep",
            reason="Best quality",
        )
        self.assertEqual(d.decision, "keep")
        self.assertEqual(d.score, 85.0)


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

    def test_keeps_only_best(self):
        """AEB group should keep only the best image."""
        group = self._make_aeb_group()
        scores = [
            QualityScore(filepath=Path("best.CR2"), overall_score=90.0,
                        sharpness_score=90, noise_score=90, detail_score=90, defect_score=90),
            QualityScore(filepath=Path("mid.CR2"), overall_score=75.0,
                        sharpness_score=75, noise_score=75, detail_score=75, defect_score=75),
            QualityScore(filepath=Path("dark.CR2"), overall_score=60.0,
                        sharpness_score=60, noise_score=60, detail_score=60, defect_score=60),
        ]

        kept, decisions = select_from_aeb_group(group, scores)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], Path("best.CR2"))

        # One keep, two rejects
        keep_decisions = [d for d in decisions if d.decision == "keep"]
        reject_decisions = [d for d in decisions if d.decision == "reject"]
        self.assertEqual(len(keep_decisions), 1)
        self.assertEqual(len(reject_decisions), 2)

    def test_empty_group(self):
        """Empty group should return empty results."""
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

    def test_keeps_above_threshold(self):
        """Burst group should keep all images above threshold."""
        group = self._make_burst_group()
        scores = [
            QualityScore(filepath=Path("good1.CR2"), overall_score=80.0,
                        sharpness_score=80, noise_score=80, detail_score=80, defect_score=80),
            QualityScore(filepath=Path("good2.CR2"), overall_score=70.0,
                        sharpness_score=70, noise_score=70, detail_score=70, defect_score=70),
            QualityScore(filepath=Path("bad.CR2"), overall_score=40.0,
                        sharpness_score=40, noise_score=40, detail_score=40, defect_score=40),
        ]

        kept, decisions = select_from_burst_group(group, scores, min_score=60.0)

        self.assertEqual(len(kept), 2)
        self.assertIn(Path("good1.CR2"), kept)
        self.assertIn(Path("good2.CR2"), kept)
        self.assertNotIn(Path("bad.CR2"), kept)


class TestSelectFromSingle(unittest.TestCase):

    def test_keeps_above_threshold(self):
        """Single above threshold should be kept."""
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        scores = [
            QualityScore(filepath=Path("solo.CR2"), overall_score=75.0,
                        sharpness_score=75, noise_score=75, detail_score=75, defect_score=75),
        ]

        kept, decisions = select_from_single(group, scores, min_score=50.0)

        self.assertEqual(len(kept), 1)
        self.assertEqual(decisions[0].decision, "keep")

    def test_rejects_below_threshold(self):
        """Single below threshold should be rejected."""
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
            group_id=1,
        )
        scores = [
            QualityScore(filepath=Path("solo.CR2"), overall_score=30.0,
                        sharpness_score=30, noise_score=30, detail_score=30, defect_score=30),
        ]

        kept, decisions = select_from_single(group, scores, min_score=50.0)

        self.assertEqual(len(kept), 0)
        self.assertEqual(decisions[0].decision, "reject")

    def test_empty_group(self):
        """Empty group should return empty results."""
        group = BracketGroup(
            group_type=GroupType.SINGLE,
            files=[],
            group_id=1,
        )
        kept, decisions = select_from_single(group, [], min_score=50.0)
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(decisions), 0)


class TestSelectionResult(unittest.TestCase):

    def test_summary_output(self):
        """Summary should contain meaningful information."""
        result = SelectionResult(
            decisions=[
                SelectionDecision(filepath=Path("kept.CR2"), score=85.0, decision="keep", reason="Good"),
                SelectionDecision(filepath=Path("rejected.CR2"), score=40.0, decision="reject", reason="Bad"),
            ],
            kept_files=[Path("kept.CR2")],
            rejected_files=[Path("rejected.CR2")],
            upload_success=2,
            upload_failed=0,
        )

        summary = result.summary()
        self.assertIn("1", summary)  # kept count
        self.assertIn("1", summary)  # rejected count
        self.assertIn("kept.CR2", summary)
        self.assertIn("rejected.CR2", summary)

    def test_counts(self):
        """Counts should match file lists."""
        result = SelectionResult(
            decisions=[],
            kept_files=[Path("a.CR2"), Path("b.CR2")],
            rejected_files=[Path("c.CR2")],
        )
        self.assertEqual(result.kept_count, 2)
        self.assertEqual(result.rejected_count, 1)


class TestSelectAndUpload(unittest.TestCase):

    def _make_quality_result(self) -> QualityScorerResult:
        return QualityScorerResult(
            scored_groups=[
                BracketGroup(
                    group_type=GroupType.AEB,
                    files=[
                        FileExifData(filepath=Path("best.CR2"), exposure_compensation=0.0),
                        FileExifData(filepath=Path("dark.CR2"), exposure_compensation=-2.0),
                    ],
                    group_id=1,
                ),
                BracketGroup(
                    group_type=GroupType.SINGLE,
                    files=[FileExifData(filepath=Path("solo.CR2"), exposure_compensation=0.0)],
                    group_id=2,
                ),
            ],
            scores=[
                QualityScore(filepath=Path("best.CR2"), overall_score=90.0,
                            sharpness_score=90, noise_score=90, detail_score=90, defect_score=90),
                QualityScore(filepath=Path("dark.CR2"), overall_score=60.0,
                            sharpness_score=60, noise_score=60, detail_score=60, defect_score=60),
                QualityScore(filepath=Path("solo.CR2"), overall_score=75.0,
                            sharpness_score=75, noise_score=75, detail_score=75, defect_score=75),
            ],
            total_scored=3,
            total_failed=0,
        )

    @patch('modules.selector.upload_files')
    @patch('modules.selector.NextcloudClient')
    @patch('modules.selector.generate_selection_report')
    def test_select_and_upload_calls_upload(self, mock_report, mock_nc, mock_upload):
        """Should call upload for kept and rejected files."""
        mock_nc_instance = MagicMock()
        mock_nc.return_value = mock_nc_instance
        mock_report.return_value = Path("/tmp/report.json")
        mock_upload.return_value = (2, 0)

        quality_result = self._make_quality_result()
        temp_dir = Path("/tmp/test_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        result = select_and_upload(
            quality_result=quality_result,
            nc_client=mock_nc_instance,
            batch_name="test_batch",
            temp_dir=temp_dir,
        )

        # Should have kept 2 files (best from AEB + single)
        self.assertEqual(result.kept_count, 2)
        # Should have rejected 1 file (dark from AEB)
        self.assertEqual(result.rejected_count, 1)

        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
