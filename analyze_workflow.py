"""
Analyze Workflow — Manuellen Raw Therapee-Workflow pro Gruppe analysieren.

Vergleicht Select vs. Reject innerhalb jeder Bracket-Gruppe.
Alle Metriken werden vorab einmal berechnet und gecached.

Usage:
  .venv/bin/python3 analyze_workflow.py --batch SW-England-May26-01
  .venv/bin/python3 analyze_workflow.py --batch SW-England-May26-01 --local
"""
import argparse
import configparser
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from modules.bracket_detector import BracketGroup, GroupType, detect_brackets
from modules.nextcloud_client import NextcloudClient, init_nextcloud
from modules.quality_scorer import ImageMetrics, compute_metrics

logger = logging.getLogger(__name__)

NC_PROJ_PATH = "Photos/StockFotoCreator/select-pipe-proj"
LOCAL_TEMP = Path.home() / "stock-pipeline-temp-v2" / "workflow-analysis"
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}


def parse_args():
    p = argparse.ArgumentParser(description="Analysiere Workflow pro Gruppe")
    p.add_argument("--batch", default="SW-England-May26-01")
    p.add_argument("--local", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-files", action="store_true",
                   help="Lokale RAW-Dateien nach Analyse NICHT löschen")
    return p.parse_args()


def download_folder(nc, remote, local):
    items = nc.list_dir(remote)
    if not items:
        return 0
    c = 0
    for item in items:
        name = item["name"]
        if name.startswith("."):
            continue
        if nc.download_file(f"{remote}/{name}", local / name):
            c += 1
    return c


def parse_pp3(filepath):
    if not filepath.exists() or filepath.suffix.lower() != ".pp3":
        return {}
    cfg = configparser.ConfigParser()
    try:
        cfg.read(str(filepath))
    except Exception:
        return {}
    result = {}
    for s in cfg.sections():
        result[s] = dict(cfg[s])
    return result


NOISE_KEYS = {
    "darkframe", "flatfieldfile", "exifkeys", "scale",
    "refoutput", "camerafocallength",
}

NOISE_CATEGORIES = {
    "Crop", "Perspective", "Resize", "MetaData", "RAW",
    "EPD", "Common Properties for Transformations",
    "Version", "General",
}


def pp3_diff(align, untouch):
    """Compare two .pp3 dicts, return only meaningful user edits.

    Returns: {section: {key: {"align": val, "untouch": val}}}
    """
    diff = {}

    relevant = {
        "Exposure": [
            "brightness", "contrast", "saturation", "black",
            "highlightcompr", "highlightcomprthreshold",
            "shadowcompr", "shadowcomprthreshold",
            "auto", "clip", "compensation",
            "histogrammatching", "curvefromhistogrammatching",
            "curve", "curvemode", "curvemode2",
        ],
        "Color appearance": [
            "algorithm", "detailrecovery", "artifactfiltersetting",
            "enhancement",
        ],
        "Directional Pyramid Denoising": ["methodmed"],
        "Shadows & Highlights": [
            "enabled", "highlights", "shadows",
            "highlighttoningwidth", "shadowtoningwidth",
        ],
        "PostDemosaicSharpening": [
            "enabled", "amount", "radius", "contrast",
            "deconvradius", "threshold", "method",
        ],
        "Color Management": [
            "inputprofile", "will", "temperature", "tint",
            "workingprofile",
        ],
        "Wavelet": [
            "enabled", "chromamethod", "mixmethod",
            "luminancemethod",
        ],
        "HLRecovery": ["enabled", "method"],
        "Local Contrast": ["enabled", "amount", "radius"],
        "Vibrance": ["enabled", "pastels", "saturated"],
        "Tone Curve 1": ["enabled", "curve"],
        "Tone Curve 2": ["enabled", "curve"],
        "Film Negative": ["enabled", "colorring", "colorringlimit"],
    }
    for section, keys in relevant.items():
        sa = align.get(section, {})
        su = untouch.get(section, {})
        sd = {}
        for k in keys:
            va = sa.get(k)
            vu = su.get(k)
            if va == vu:
                continue
            if va is None and vu is None:
                continue
            sd[k] = {"align": va, "untouch": vu}
        if sd:
            diff[section] = sd

    # Also catch anything else that differs and is meaningful
    all_sections = set(align.keys()) | set(untouch.keys())
    for section in all_sections - set(relevant.keys()):
        if section in NOISE_CATEGORIES:
            continue
        sa = align.get(section, {})
        su = untouch.get(section, {})
        sd = {}
        for k in set(sa.keys()) | set(su.keys()):
            if k.lower() in NOISE_KEYS:
                continue
            va = sa.get(k)
            vu = su.get(k)
            if va == vu:
                continue
            sd[k] = {"align": va, "untouch": vu}
        if sd:
            diff[section] = sd

    return diff


def describe_pp3_diff(diff):
    if not diff:
        return "-"
    parts = []
    for section, keys in diff.items():
        for key, kv in keys.items():
            va = str(kv.get("align", ""))
            vu = str(kv.get("untouch", ""))
            if key == "Enabled":
                parts.append(f"{section}:{'ON' if va=='true' else 'OFF'}")
                continue
            # Try numeric diff
            try:
                dv = float(va) - float(vu)
                d = "↑" if dv > 0 else "↓"
                parts.append(f"{section}.{key}{d}{abs(dv):.1f}")
            except (ValueError, TypeError):
                parts.append(f"{section}.{key}: {vu}→{va}")
    return ", ".join(parts)


def compute_metrics_safe(fp):
    try:
        return compute_metrics(fp)
    except Exception as e:
        logger.warning(f"Metrik-Fehler {fp.name}: {e}")
        return None


def load_metrics_cache(cache_path: Path) -> dict[str, ImageMetrics]:
    """Load cached metrics from JSON."""
    if not cache_path.exists():
        return {}
    import json
    try:
        with open(cache_path) as f:
            data = json.load(f)
        result = {}
        for stem, vals in data.items():
            result[stem] = ImageMetrics(
                filepath=Path(vals.get("filepath", "")),
                exposure_score=vals.get("exposure_score", 50),
                noise_score=vals.get("noise_score", 50),
                sharpness_score=vals.get("sharpness_score", 50),
                detail_score=vals.get("detail_score", 50),
                defect_score=vals.get("defect_score", 50),
            )
        return result
    except Exception:
        return {}


def save_metrics_cache(cache_path: Path, metrics: dict[str, ImageMetrics]):
    """Save metrics cache to JSON."""
    import json
    data = {}
    for stem, m in metrics.items():
        data[stem] = m.to_dict()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def precompute_all(
    alignment_dir: Path,
    untouched_pp3_dir: Path,
    alignment_pp3_dir: Path,
    cache_path: Path = None,
) -> tuple[dict[str, ImageMetrics], dict[str, str]]:
    """Precompute metrics and pp3 diffs for all alignment RAWs."""
    raws = sorted(p for p in alignment_dir.iterdir() if p.suffix.lower() in RAW_EXTS)
    n = len(raws)

    # Load cached metrics if available
    metrics_cache = {}
    if cache_path:
        metrics_cache = load_metrics_cache(cache_path)

    missing = [r for r in raws if r.stem not in metrics_cache]
    if missing:
        print(f"\nBerechne Metriken für {len(missing)}/{n} RAWs...")
        for i, raw in enumerate(missing, 1):
            stem = raw.stem
            sys.stdout.write(f"\r  [{i}/{len(missing)}] {raw.name[:35]:35s}")
            sys.stdout.flush()
            m = compute_metrics_safe(raw)
            if m:
                metrics_cache[stem] = m
        sys.stdout.write(f"\n")
        sys.stdout.flush()
        if cache_path:
            save_metrics_cache(cache_path, metrics_cache)
    else:
        print(f"\nMetriken aus Cache geladen ({len(metrics_cache)} Einträge)")

    # pp3 diffs (fast, always recompute)
    print(f"Berechne pp3-Diffs für {n} Bilder...")
    pp3_cache = {}
    for i, raw in enumerate(raws, 1):
        sys.stdout.write(f"\r  [{i}/{n}] {raw.name[:35]:35s}")
        sys.stdout.flush()
        align = parse_pp3(alignment_pp3_dir / f"{raw.name}.pp3")
        untouch = parse_pp3(untouched_pp3_dir / f"{raw.name}.pp3")
        pp3_cache[raw.stem] = describe_pp3_diff(pp3_diff(align, untouch))

    sys.stdout.write(f"\n  Fertig: {len(metrics_cache)} Metriken, {len(pp3_cache)} pp3-Diffs\n")
    sys.stdout.flush()
    return metrics_cache, pp3_cache


def build_decision_map(select_dir, reject_dir):
    mapping = {}
    for f in select_dir.iterdir():
        if f.suffix.lower() in RAW_EXTS:
            mapping[f.stem] = "select"
    for f in reject_dir.iterdir():
        if f.suffix.lower() in RAW_EXTS:
            mapping[f.stem] = "reject"
    return mapping


def ms(m):
    return (f"n={m.noise_score:.0f} s={m.sharpness_score:.0f} "
            f"e={m.exposure_score:.0f} d={m.detail_score:.0f} "
            f"df={m.defect_score:.0f}")


def analyze_groups(
    groups: list[BracketGroup],
    decision_map: dict[str, str],
    metrics_cache: dict[str, ImageMetrics],
    pp3_cache: dict[str, str],
) -> list[str]:
    all_lines = []

    for group in groups:
        gtype = group.group_type.value.upper()
        all_lines.append(f"\n── Gruppe #{group.group_id} [{gtype}] ({group.file_count} Dateien) ──")

        # Classify files
        sel = []
        rej = []
        unk = []
        for fd in group.files:
            v = decision_map.get(fd.filepath.stem, "unknown")
            if v == "select":
                sel.append(fd)
            elif v == "reject":
                rej.append(fd)
            else:
                unk.append(fd)

        if unk:
            all_lines.append(f"  ⚠ Unklassifiziert: {', '.join(f.filename for f in unk)}")

        if not sel and not rej:
            all_lines.append(f"  ⏭ Keine klassifizierten Dateien")
            continue

        # ─── Type A: at least 1 select ───
        if sel:
            all_lines.append(f"  ✅ SELECT: {', '.join(f.filename for f in sel)}")
            for sf in sel:
                m = metrics_cache.get(sf.filepath.stem)
                pp = pp3_cache.get(sf.filepath.stem, "N/A")
                all_lines.append(f"     📊 {sf.filepath.stem}: {ms(m) if m else 'N/A'}")
                all_lines.append(f"     📝 pp3: {pp}")

            if rej:
                all_lines.append(f"  ❌ REJECT: {', '.join(f.filename for f in rej)}")
                for rf in rej:
                    m = metrics_cache.get(rf.filepath.stem)
                    pp = pp3_cache.get(rf.filepath.stem, "N/A")
                    all_lines.append(f"     📊 {rf.filepath.stem}: {ms(m) if m else 'N/A'}")
                    all_lines.append(f"     📝 pp3: {pp}")

                # Compare
                all_lines.append(f"  🔍 Vergleich:")
                diffs = {}
                for attr, label in [("noise_score", "Noise"), ("sharpness_score", "Sharp"),
                                     ("exposure_score", "Exposure"), ("defect_score", "Defects"),
                                     ("detail_score", "Detail")]:
                    sv = [getattr(metrics_cache[sf.filepath.stem], attr) for sf in sel
                          if sf.filepath.stem in metrics_cache]
                    rv = [getattr(metrics_cache[rf.filepath.stem], attr) for rf in rej
                          if rf.filepath.stem in metrics_cache]
                    if sv and rv:
                        sa = sum(sv) / len(sv)
                        ra = sum(rv) / len(rv)
                        d = sa - ra
                        icon = "👍" if d > 0 else "👎"
                        diffs[attr] = (sa, ra, d)
                        all_lines.append(f"     {label}: S={sa:.0f} vs R={ra:.0f}  ({icon}{abs(d):.1f})")

                # DRC Detection: check if Select has shadowcompr/highlightcompr edits
                drc_signal = False
                for sf in sel:
                    pp = pp3_cache.get(sf.filepath.stem, "")
                    if "shadowcompr" in pp or "highlightcompr" in pp or "DRC" in pp:
                        drc_signal = True
                        break

                # Interpret
                reasons = []
                if drc_signal:
                    reasons.append("🔄 DRC Himmel-Rettung (shadowcompr/highlightcompr aktiv)")

                if "noise_score" in diffs:
                    nd = diffs["noise_score"][2]
                    if nd > 5:
                        reasons.append("weniger Rauschen")
                    elif nd < -5:
                        reasons.append("⚠ MEHR Rauschen (Himmel-Kompromiss?)")
                if "sharpness_score" in diffs:
                    sd = diffs["sharpness_score"][2]
                    if sd > 5:
                        reasons.append("schärfer")
                    elif sd < -5:
                        reasons.append("⚠ WENIGER scharf (Bewegungsunschärfe?)")
                if "exposure_score" in diffs:
                    ed = diffs["exposure_score"][2]
                    if ed > 5:
                        reasons.append("besser belichtet")
                    elif ed < -5:
                        reasons.append("schlechter belichtet (trotzdem gewählt)")
                if "defect_score" in diffs:
                    dd = diffs["defect_score"][2]
                    if dd > 5:
                        reasons.append("weniger CA/Fehler")

                if reasons:
                    all_lines.append(f"     💡 {', '.join(reasons)}")
                else:
                    all_lines.append(f"     💡 Metrik-Gleichstand → Tiebreaker (Komposition/Augen/Himmel/"
                                    f"keine störenden Objekte)")
            else:
                all_lines.append(f"  ℹ Keine Rejects in dieser Gruppe")

        # ─── Type B: all rejected ───
        elif rej and not sel:
            all_lines.append(f"  ❌ ALLE REJECTED:")
            noises = []
            sharps = []
            for rf in rej:
                m = metrics_cache.get(rf.filepath.stem)
                if m:
                    noises.append(m.noise_score)
                    sharps.append(m.sharpness_score)
                    all_lines.append(f"     📊 {rf.filepath.stem}: {ms(m)}")
                    all_lines.append(f"     📝 pp3: {pp3_cache.get(rf.filepath.stem, 'N/A')}")

            reasons = []
            if noises and all(n < 30 for n in noises):
                reasons.append("ALLE stark verrauscht")
            elif noises and sum(1 for n in noises if n < 30) > len(noises) / 2:
                reasons.append("mehrheitlich verrauscht")
            if sharps and all(s < 15 for s in sharps):
                reasons.append("ALLE unscharf")
            elif sharps and sum(1 for s in sharps if s < 15) > len(sharps) / 2:
                reasons.append("mehrheitlich unscharf")

            if reasons:
                all_lines.append(f"     💡 {'; '.join(reasons)}")
            elif noises:
                avg_n = sum(noises) / len(noises)
                all_lines.append(f"     💡 Tendenziell verrauscht (⌀ noise={avg_n:.0f})")
            else:
                all_lines.append(f"     💡 Keine offensichtlichen Metrik-Probleme → Komposition?")

    return all_lines


def main():
    args = parse_args()
    batch = args.batch

    untouched_dir_name = f"{batch}-untouched"
    alignment_dir_name = f"{batch}-alignment"
    select_dir_name = "user-select"
    reject_dir_name = "user-reject"

    local_base = LOCAL_TEMP / batch
    local_untouched = local_base / untouched_dir_name
    local_alignment = local_base / alignment_dir_name
    local_select = local_base / select_dir_name
    local_reject = local_base / reject_dir_name

    if args.dry_run:
        print(f"Dry Run: batch={batch}, local={local_base}")
        return

    if not args.local:
        nc = init_nextcloud()
        if not nc:
            print("Kein Nextcloud-Zugang")
            return
        if local_base.exists():
            shutil.rmtree(local_base)
        print("Download Nextcloud...")
        for d in [untouched_dir_name, alignment_dir_name, select_dir_name, reject_dir_name]:
            c = download_folder(nc, f"{NC_PROJ_PATH}/{d}", local_base / d)
            print(f"  {d}: {c}")
    else:
        print(f"Lokaler Modus: {local_base}")

    decision_map = build_decision_map(local_select, local_reject)
    n_sel = sum(1 for v in decision_map.values() if v == "select")
    n_rej = sum(1 for v in decision_map.values() if v == "reject")
    print(f"\nKlassifiziert: {n_sel} select, {n_rej} reject")

    alignment_raws = sorted(p for p in local_alignment.iterdir() if p.suffix.lower() in RAW_EXTS)
    if not alignment_raws:
        print("Keine RAWs in alignment")
        return

    print(f"\nGruppiere {len(alignment_raws)} RAWs...")
    groups = detect_brackets(alignment_raws)
    print(f"{len(groups)} Gruppen")

    # Pre-compute all metrics + pp3 diffs
    cache_path = local_base / "metrics_cache.json"
    metrics_cache, pp3_cache = precompute_all(
        alignment_dir=local_alignment,
        untouched_pp3_dir=local_untouched,
        alignment_pp3_dir=local_alignment,
        cache_path=cache_path,
    )

    # Analyze
    print(f"\n{'='*70}")
    print("  GRUPPEN-ANALYSE")
    print("  User: Himmel > Rauschen | DRC Himmel-Rettung via shadowcompr/highlightcompr")
    print("  User: Augen offen > Schärfe | Bewegungsunschärfe OK")
    print("  User: Tiebreaker bei Metrik-Gleichstand → ohne Menschen/störende Objekte")
    print(f"{'='*70}")

    all_lines = analyze_groups(groups, decision_map, metrics_cache, pp3_cache)

    for line in all_lines:
        print(line)

    # Summary
    n_select = sum(1 for l in all_lines if "✅ SELECT" in l and "ALL" not in l)
    n_reject_all = sum(1 for l in all_lines if "❌ ALLE REJECTED" in l)
    print(f"\n{'='*70}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"  Gruppen mit Select:      {n_select}")
    print(f"  Gruppen komplett Reject: {n_reject_all}")
    print(f"  Gesamt:                  {len(groups)}")
    print(f"{'='*70}\n")

    if not args.keep_files and local_base.exists():
        print(f"Räume auf: {local_base}")
        shutil.rmtree(local_base)
        print("Cleanup abgeschlossen\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    main()
