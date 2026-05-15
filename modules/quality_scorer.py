"""
Module 05: Quality Scorer

AI-powered quality assessment for images.

Evaluates each image on:
- Sharpness (detail clarity)
- Noise (grain/artifacts)
- Image details (texture, structure)
- Image defects (chromatic aberration, distortion, etc.)

Returns a quality score (0-100) per image.

For AEB groups: scores all remaining images after exposure alignment
For Burst groups: scores all images (user selects later)
For Singles: scores the single image

Input:  List of BracketGroup objects
Output: Same groups with quality scores attached to each file
"""
import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import convolve as scipy_convolve

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from modules.bracket_detector import BracketGroup, GroupType, FileExifData

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment for a single image."""
    filepath: Path
    overall_score: float  # 0-100
    sharpness_score: float  # 0-100
    noise_score: float  # 0-100 (higher = less noise)
    detail_score: float  # 0-100
    defect_score: float  # 0-100 (higher = fewer defects)
    assessment: str = ""  # Human-readable assessment
    model_used: str = ""

    @property
    def filename(self) -> str:
        return self.filepath.name

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "overall_score": round(self.overall_score, 1),
            "sharpness_score": round(self.sharpness_score, 1),
            "noise_score": round(self.noise_score, 1),
            "detail_score": round(self.detail_score, 1),
            "defect_score": round(self.defect_score, 1),
            "assessment": self.assessment,
            "model_used": self.model_used,
        }


@dataclass
class QualityScorerResult:
    """Result of quality scoring for all groups."""
    scored_groups: list[BracketGroup]
    scores: list[QualityScore]
    total_scored: int = 0
    total_failed: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  QUALITY SCORING SUMMARY",
            f"{'='*60}",
            f"  Images scored:   {self.total_scored}",
            f"  Images failed:   {self.total_failed}",
        ]
        if self.scores:
            lines.append("")
            lines.append("  Scores:")
            for s in sorted(self.scores, key=lambda x: x.overall_score, reverse=True):
                lines.append(f"    {s.filename}: {s.overall_score:.0f}/100 "
                           f"(sharp={s.sharpness_score:.0f}, noise={s.noise_score:.0f}, "
                           f"detail={s.detail_score:.0f}, defects={s.defect_score:.0f})")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}


def load_image_array(filepath: Path) -> np.ndarray:
    """
    Load an image as a numpy array.

    For RAW files (CR2/CR3/etc): uses rawpy to render full-resolution RGB.
    For JPEG/PNG: uses PIL.

    Returns:
        numpy array of shape (H, W, 3) with uint8 values.
    """
    ext = filepath.suffix.lower()

    if ext in RAW_EXTENSIONS and HAS_RAWPY:
        try:
            with rawpy.imread(str(filepath)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=False,
                )
            return rgb
        except Exception as e:
            logger.warning(f"rawpy failed for {filepath.name}: {e}, falling back to PIL")

    # Fallback: PIL (works for JPEG/PNG, reads thumbnail for RAW)
    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def resize_for_vision(filepath: Path, max_size: int = 1024) -> str:
    """
    Resize image and encode as base64 for Vision API.

    Args:
        filepath: Input image path
        max_size: Maximum dimension (width or height)

    Returns:
        Base64-encoded JPEG string
    """
    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if larger than max_size
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_openai_client():
    """Initialize OpenAI client from secret-tool or environment."""
    try:
        import subprocess
        key = subprocess.check_output(
            ["secret-tool", "lookup", "service", "openai", "purpose", "stockfoto"],
            text=True
        ).strip()
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        return None


def get_gemini_client():
    """Initialize Gemini client from environment."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai
        except ImportError:
            return None
    return None


QUALITY_PROMPT = """You are a professional photo quality assessor.

Analyze this image and provide quality scores on a scale of 0-100 for:

1. **Sharpness**: How clear and well-focused are the details? (100 = tack sharp, 0 = completely blurry)
2. **Noise**: How clean is the image? (100 = perfectly clean, 0 = extremely noisy/grainy)
3. **Detail**: How much texture and structure is visible? (100 = rich detail, 0 = no detail)
4. **Defects**: How free is the image from artifacts? (100 = no defects, 0 = severe defects like chromatic aberration, distortion, banding, etc.)

Also provide an **overall_score** (weighted average) and a brief **assessment** (1-2 sentences).

Return JSON ONLY:
{
  "sharpness_score": 85,
  "noise_score": 90,
  "detail_score": 80,
  "defect_score": 95,
  "overall_score": 87,
  "assessment": "Brief quality assessment"
}"""


def score_image_openai(filepath: Path, client) -> QualityScore:
    """Score an image using OpenAI Vision API."""
    try:
        image_b64 = resize_for_vision(filepath)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": QUALITY_PROMPT},
                ]
            }],
            max_tokens=500,
            temperature=0.1,
        )

        text = response.choices[0].message.content
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])

        return QualityScore(
            filepath=filepath,
            overall_score=float(result.get("overall_score", 50)),
            sharpness_score=float(result.get("sharpness_score", 50)),
            noise_score=float(result.get("noise_score", 50)),
            detail_score=float(result.get("detail_score", 50)),
            defect_score=float(result.get("defect_score", 50)),
            assessment=result.get("assessment", ""),
            model_used="gpt-4o-mini",
        )

    except Exception as e:
        logger.error(f"OpenAI scoring error for {filepath.name}: {e}")
        return QualityScore(
            filepath=filepath,
            overall_score=50.0,
            sharpness_score=50.0,
            noise_score=50.0,
            detail_score=50.0,
            defect_score=50.0,
            assessment=f"Scoring failed: {e}",
            model_used="error",
        )


def score_image_gemini(filepath: Path, genai) -> QualityScore:
    """Score an image using Gemini Vision API."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        img = Image.open(filepath)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)

        response = model.generate_content([
            QUALITY_PROMPT,
            {"mime_type": "image/jpeg", "data": buf.getvalue()},
        ])

        text = response.text
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])

        return QualityScore(
            filepath=filepath,
            overall_score=float(result.get("overall_score", 50)),
            sharpness_score=float(result.get("sharpness_score", 50)),
            noise_score=float(result.get("noise_score", 50)),
            detail_score=float(result.get("detail_score", 50)),
            defect_score=float(result.get("defect_score", 50)),
            assessment=result.get("assessment", ""),
            model_used="gemini-2.0-flash",
        )

    except Exception as e:
        logger.error(f"Gemini scoring error for {filepath.name}: {e}")
        return QualityScore(
            filepath=filepath,
            overall_score=50.0,
            sharpness_score=50.0,
            noise_score=50.0,
            detail_score=50.0,
            defect_score=50.0,
            assessment=f"Scoring failed: {e}",
            model_used="error",
        )


def score_image_fallback(filepath: Path) -> QualityScore:
    """
    Fallback quality scoring using algorithmic methods.
    Used when no AI API is available.

    Uses:
    - Laplacian variance for sharpness
    - High-pass filter energy for noise
    - Local contrast for detail
    - Color channel correlation for defects
    """
    try:
        arr = load_image_array(filepath)
        gray = np.mean(arr.astype(float), axis=2)

        # Sharpness: Laplacian variance
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap = scipy_convolve(gray, laplacian_kernel)
        lap_var = float(np.var(lap))
        # Normalize to 0-100 (typical range: 0-500)
        sharpness_score = min(lap_var / 5.0, 100.0)

        # Noise: High-pass filter energy in flat areas
        blurred = np.zeros_like(gray)
        kernel_size = 5
        for i in range(kernel_size):
            for j in range(kernel_size):
                blurred += gray
        blurred /= (kernel_size * kernel_size)
        highpass = gray - blurred
        noise_energy = float(np.percentile(np.abs(highpass), 25))
        # Lower noise = higher score
        noise_score = max(100.0 - (noise_energy * 5), 0.0)

        # Detail: Local contrast (standard deviation in small windows)
        detail_score = min(float(np.std(gray)) / 2.0, 100.0)

        # Defects: Check for chromatic aberration (R-B channel difference in edges)
        r_channel = arr[:, :, 0].astype(float)
        b_channel = arr[:, :, 2].astype(float)
        rb_diff = np.abs(r_channel - b_channel)
        edge_mask = np.abs(lap) > np.mean(np.abs(lap)) * 2
        if np.any(edge_mask):
            ca_score = float(np.mean(rb_diff[edge_mask]))
            defect_score = max(100.0 - (ca_score * 2), 0.0)
        else:
            defect_score = 80.0

        overall_score = (
            sharpness_score * 0.35 +
            noise_score * 0.25 +
            detail_score * 0.25 +
            defect_score * 0.15
        )

        return QualityScore(
            filepath=filepath,
            overall_score=overall_score,
            sharpness_score=sharpness_score,
            noise_score=noise_score,
            detail_score=detail_score,
            defect_score=defect_score,
            assessment="Algorithmic assessment (fallback)",
            model_used="algorithmic-fallback",
        )

    except Exception as e:
        logger.error(f"Fallback scoring error for {filepath.name}: {e}")
        return QualityScore(
            filepath=filepath,
            overall_score=50.0,
            sharpness_score=50.0,
            noise_score=50.0,
            detail_score=50.0,
            defect_score=50.0,
            assessment=f"Fallback scoring failed: {e}",
            model_used="error",
        )


def score_image(filepath: Path) -> QualityScore:
    """
    Score an image using the best available method.

    Priority: OpenAI > Gemini > Algorithmic Fallback
    """
    # Try OpenAI first
    openai_client = get_openai_client()
    if openai_client is not None:
        logger.info(f"  Scoring {filepath.name} with OpenAI Vision...")
        return score_image_openai(filepath, openai_client)

    # Try Gemini
    gemini_client = get_gemini_client()
    if gemini_client is not None:
        logger.info(f"  Scoring {filepath.name} with Gemini Vision...")
        return score_image_gemini(filepath, gemini_client)

    # Fallback to algorithmic
    logger.warning(f"  No AI API available, using algorithmic fallback for {filepath.name}")
    return score_image_fallback(filepath)


def score_all_images(groups: list[BracketGroup]) -> QualityScorerResult:
    """
    Score all images in all groups.

    For each group:
    1. Score every image
    2. Attach scores to FileExifData (via QualityScore)
    3. Return groups with scores

    Args:
        groups: List of BracketGroup objects

    Returns:
        QualityScorerResult with scored groups
    """
    scored_groups = []
    scores = []
    total_scored = 0
    total_failed = 0

    for group in groups:
        logger.info(f"Scoring group #{group.group_id} ({group.group_type.value}, "
                   f"{group.file_count} files)")

        group_scores = []
        for fd in group.files:
            score = score_image(fd.filepath)
            scores.append(score)
            group_scores.append(score)

            if score.model_used == "error":
                total_failed += 1
            else:
                total_scored += 1

            logger.info(f"  {fd.filename}: {score.overall_score:.0f}/100 "
                       f"({score.model_used})")

        # Sort group files by score (best first)
        scored_files = sorted(
            zip(group.files, group_scores),
            key=lambda x: x[1].overall_score,
            reverse=True
        )

        # Create new group with files sorted by score
        new_group = BracketGroup(
            group_type=group.group_type,
            files=[fd for fd, _ in scored_files],
            group_id=group.group_id,
        )
        scored_groups.append(new_group)

    result = QualityScorerResult(
        scored_groups=scored_groups,
        scores=scores,
        total_scored=total_scored,
        total_failed=total_failed,
    )

    logger.info(result.summary())
    return result
