"""
Tests for Module 04: Sky Recovery — DRC via pp3 Sidecar
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from modules.bracket_detector import BracketGroup, GroupType, FileExifData
from modules.sky_recovery import (
    DRCResult,
    SkyRecoveryResult,
    analyze_highlights,
    try_drc_recovery,
    recover_highlights,
    write_drc_pp3,
)


class TestAnalyzeHighlights(unittest.TestCase):

    def test_no_clipping(self):
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            img.save(f.name)
            clip_ratio, detail, grad_loss = analyze_highlights(Path(f.name))
        self.assertAlmostEqual(clip_ratio, 0.0, places=2)
        self.assertEqual(detail, 0.0)
        self.assertEqual(grad_loss, 0.0)

    def test_clipped_image(self):
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        arr[50:, :, :] = 128
        img = Image.fromarray(arr.astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            img.save(f.name)
            clip_ratio, detail, grad_loss = analyze_highlights(Path(f.name))
        self.assertGreater(clip_ratio, 0.2)

    def test_partial_clipping(self):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        arr[:20, :, :] = 255
        img = Image.fromarray(arr.astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            img.save(f.name)
            clip_ratio, detail, grad_loss = analyze_highlights(Path(f.name))
        self.assertGreater(clip_ratio, 0.0)
        self.assertLess(clip_ratio, 0.5)


class TestWriteDrcPp3(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_pp3_with_highlight_compr(self):
        out = self.tmpdir / "test_drc.pp3"
        success = write_drc_pp3(Path("test.CR2"), out, strength=15)
        self.assertTrue(success)
        self.assertTrue(out.exists())
        content = out.read_text()
        self.assertIn("HighlightCompr=75", content)
        self.assertIn("HLRecovery", content)

    def test_strength_clamped(self):
        out = self.tmpdir / "test_drc.pp3"
        success = write_drc_pp3(Path("test.CR2"), out, strength=50)
        self.assertTrue(success)
        content = out.read_text()
        self.assertIn("HighlightCompr=100", content)


class TestTryDrcRecovery(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_clipping_skips_drc(self):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr.astype(np.uint8))
        src = self.tmpdir / "test.jpg"
        img.save(str(src))
        result = try_drc_recovery(src, self.tmpdir)
        self.assertFalse(result.drc_applied)
        self.assertFalse(result.drc_success)

    def test_clipped_with_detail_tries_drc(self):
        arr = np.full((200, 200, 3), 255, dtype=np.uint8)
        for y in range(200):
            val = max(200, int(255 - y * 0.15))
            arr[y, :, :] = val
        img = Image.fromarray(arr.astype(np.uint8))
        src = self.tmpdir / "test_clip.jpg"
        img.save(str(src))

        result = try_drc_recovery(src, self.tmpdir)
        self.assertTrue(result.drc_applied)

    def test_clipped_no_detail_skips_drc(self):
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr.astype(np.uint8))
        src = self.tmpdir / "test_flat.jpg"
        img.save(str(src))
        result = try_drc_recovery(src, self.tmpdir)
        self.assertFalse(result.drc_applied)


class TestRecoverHighlights(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_updates_file_paths(self):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        fp = self.tmpdir / "test.jpg"
        Image.fromarray(arr.astype(np.uint8)).save(str(fp))

        group = BracketGroup(
            group_type=GroupType.AEB,
            files=[FileExifData(filepath=fp, exposure_compensation=0.0)],
            group_id=1,
        )

        result = recover_highlights([group], output_dir=self.tmpdir / "drc_out")
        self.assertFalse(group.files[0].drc_applied)
        self.assertEqual(group.files[0].filepath, fp)

    def test_result_summary_empty(self):
        result = SkyRecoveryResult(results=[])
        summary = result.summary()
        self.assertIn("0", summary)

    def test_drc_result_dataclass(self):
        r = DRCResult(
            filepath=Path("test.jpg"),
            pp3_path=Path("test_drc.pp3"),
            drc_applied=True,
            drc_success=True,
            clipping_before=0.1,
            highlight_detail_before=0.05,
            strength_used=15,
        )
        self.assertTrue(r.drc_applied)
        self.assertTrue(r.drc_success)
        self.assertEqual(r.filename, "test.jpg")


if __name__ == "__main__":
    unittest.main()
