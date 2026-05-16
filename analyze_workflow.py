"""
Analyze Workflow — Manuellen Raw Therapee-Workflow analysieren.

Zweck:
  Die Auswahl-Logik des Selectors (modules/selector.py) soll anhand eines
  manuell kuratierten Workflows kalibriert werden. Dieses Script analysiert
  die Unterschiede zwischen:
    - untouched/  : RAWs mit Standard/neutral .pp3
    - alignment/  : RAWs + manuell bearbeitete .pp3
    - user-select/: Favoriten des Users
    - user-reject/: Aussortierte des Users

  Ausgabe: Klartext-Muster mit 3-5 konkreten Bildbeispielen – KEINE Zahlenstatistiken.

Design:
  - 0% Pipeline-Selection-Code (selector.py wird NICHT importiert)
  - Importiert nur Rechenmodule: bracket_detector, quality_scorer
  - .pp3-Differenz wird per configparser direkt verglichen

Usage:
  .venv/bin/python3 analyze_workflow.py --batch SW-England-May26-01
  .venv/bin/python3 analyze_workflow.py --batch SW-England-May26-01 --local --dry-run
"""
import argparse
import configparser
import logging
import shutil
from pathlib import Path
from typing import Optional

from modules.bracket_detector import BracketGroup, GroupType, detect_brackets
from modules.nextcloud_client import NextcloudClient, init_nextcloud
from modules.quality_scorer import ImageMetrics, compute_metrics

logger = logging.getLogger(__name__)

NC_PROJ_PATH = "Photos/StockFotoCreator/select-pipe-proj"
LOCAL_TEMP = Path.home() / "stock-pipeline-temp-v2" / "workflow-analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysiere einen manuellen Raw Therapee-Workflow"
    )
    parser.add_argument(
        "--batch",
        default="SW-England-May26-01",
        help="Batch-Name (entspricht Nextcloud-Ordnernamen, default: SW-England-May26-01)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Lokales Temp-Dir nutzen (kein Nextcloud-Download)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur zeigen was analysiert würde (ohne Download/Berechnung)",
    )
    return parser.parse_args()


def download_folder(nc: NextcloudClient, remote_path: str, local_dir: Path) -> int:
    """Download all RAW + pp3 files from a Nextcloud folder to local dir.
    Returns count of files downloaded."""
    items = nc.list_dir(remote_path)
    if not items:
        logger.warning(f"Leerer oder nicht gefundener Ordner: {remote_path}")
        return 0

    count = 0
    for item in items:
        name = item["name"]
        if name.startswith("."):
            continue
        remote_file = f"{remote_path}/{name}"
        local_file = local_dir / name
        if nc.download_file(remote_file, local_file):
            count += 1
            logger.debug(f"  Downloaded {name}")
        else:
            logger.warning(f"  Download fehlgeschlagen: {name}")
    return count


def gather_local_files(directory: Path) -> list[Path]:
    """Return all RAW + .pp3 files in a directory (non-recursive)."""
    if not directory.exists():
        return []
    result = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            result.append(f)
    return result


def parse_pp3(filepath: Path) -> dict[str, dict[str, str]]:
    """Parse a .pp3 file into {section: {key: value}}."""
    result = {}
    if not filepath.exists() or filepath.suffix.lower() != ".pp3":
        return result

    config = configparser.ConfigParser()
    try:
        config.read(str(filepath))
    except Exception:
        logger.warning(f"Konnte .pp3 nicht parsen: {filepath.name}")
        return result

    for section in config.sections():
        result[section] = {}
        for key in config[section]:
            result[section][key] = config[section][key]
    return result


def pp3_diff(
    alignment_pp3: dict, untouched_pp3: dict
) -> dict[str, dict[str, float]]:
    """Compare alignment vs untouched .pp3, return numeric diffs.

    Only returns sections/keys that actually differ.
    """
    diff = {}

    relevant_sections = {
        "Exposure": ["Exposure", "Black", "Contrast", "Saturation",
                      "HighlightCompression", "ShadowCompression", "Lightness"],
        "Shadows/Highlights": ["Highlights", "Shadows"],
        "Local Contrast": ["Amount"],
        "Vibrance": ["Pastels", "Saturated"],
        "Sharpening": ["Amount", "Radius"],
    }

    for section, keys in relevant_sections.items():
        s_align = alignment_pp3.get(section, {})
        s_untouch = untouched_pp3.get(section, {})

        section_diffs = {}
        for key in keys:
            val_a = s_align.get(key, "0")
            val_u = s_untouch.get(key, "0")
            try:
                diff_val = float(val_a) - float(val_u)
            except (ValueError, TypeError):
                continue
            if abs(diff_val) > 0.001:
                section_diffs[key] = round(diff_val, 3)

        if section_diffs:
            diff[section] = section_diffs

    # Also check if a feature was enabled/disabled
    for section in ["Shadows/Highlights", "Local Contrast", "Vibrance"]:
        enabled_a = alignment_pp3.get(section, {}).get("Enabled", "false")
        enabled_u = untouched_pp3.get(section, {}).get("Enabled", "false")
        if enabled_a != enabled_u:
            if section not in diff:
                diff[section] = {}
            diff[section]["Enabled"] = enabled_a

    return diff


def read_pp3_for_raw(raw_path: Path, pp3_dir: Path) -> dict:
    """Try to find and parse a .pp3 sidecar for a RAW file."""
    pp3_path = pp3_dir / f"{raw_path.stem}.pp3"
    if pp3_path.exists():
        return parse_pp3(pp3_path)
    return {}


def describe_pp3_diff(diff: dict) -> str:
    """Human-readable description of pp3 changes."""
    if not diff:
        return "keine nennenswerten Änderungen"

    parts = []
    for section, keys in diff.items():
        for key, val in keys.items():
            if key == "Enabled":
                parts.append(f"{section}: {'aktiviert' if val == 'true' else 'deaktiviert'}")
                continue
            direction = "erhöht" if val > 0 else "verringert"
            parts.append(f"{section}/{key} {direction} um {abs(val)}")
    return ", ".join(parts)


def compute_metrics_safe(filepath: Path) -> Optional[ImageMetrics]:
    """Compute metrics, returning None on failure."""
    try:
        return compute_metrics(filepath)
    except Exception as e:
        logger.warning(f"Metrik-Fehler für {filepath.name}: {e}")
        return None


def analyze_patterns(
    select_dir: Path,
    reject_dir: Path,
    alignment_pp3_dir: Path,
    untouched_pp3_dir: Path,
):
    """Core analysis: compare select vs reject patterns.

    Does NOT import any pipeline selection code.
    """
    select_raws = sorted(
        p for p in select_dir.iterdir() if p.suffix.lower() in (".cr2", ".cr3")
    )
    reject_raws = sorted(
        p for p in reject_dir.iterdir() if p.suffix.lower() in (".cr2", ".cr3")
    )

    if not select_raws and not reject_raws:
        logger.warning("Keine RAW-Dateien in select oder reject gefunden")
        return

    logger.info(f"Select: {len(select_raws)} RAWs, Reject: {len(reject_raws)} RAWs")

    # ── Phase 1: RAW-Metriken ────────────────────────────────────────────
    select_metrics: list[tuple[Path, ImageMetrics]] = []
    reject_metrics: list[tuple[Path, ImageMetrics]] = []

    for raw in select_raws:
        m = compute_metrics_safe(raw)
        if m:
            select_metrics.append((raw, m))

    for raw in reject_raws:
        m = compute_metrics_safe(raw)
        if m:
            reject_metrics.append((raw, m))

    if not select_metrics:
        logger.warning("Keine gültigen Metriken für Select-Bilder")
    if not reject_metrics:
        logger.warning("Keine gültigen Metriken für Reject-Bilder")

    # ── Phase 2: .pp3-Differenzen ────────────────────────────────────────
    select_pp3_diffs: list[tuple[Path, dict]] = []
    reject_pp3_diffs: list[tuple[Path, dict]] = []

    for raw, _ in select_metrics:
        alignment_pp3 = read_pp3_for_raw(raw, alignment_pp3_dir)
        untouched_pp3 = read_pp3_for_raw(raw, untouched_pp3_dir)
        diff = pp3_diff(alignment_pp3, untouched_pp3)
        select_pp3_diffs.append((raw, diff))

    for raw, _ in reject_metrics:
        alignment_pp3 = read_pp3_for_raw(raw, alignment_pp3_dir)
        untouched_pp3 = read_pp3_for_raw(raw, untouched_pp3_dir)
        diff = pp3_diff(alignment_pp3, untouched_pp3)
        reject_pp3_diffs.append((raw, diff))

    # ── Phase 3: Muster erkennen ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  WORKFLOW-ANALYSE: Select vs. Reject")
    print(f"  Batch: {select_dir.parent.parent.name}")
    print(f"{'='*70}")
    print(f"\nAusgewertet: {len(select_metrics)} select, {len(reject_metrics)} reject")

    # Metrik-Vergleich (Durchschnitte)
    if select_metrics and reject_metrics:
        print(f"\n── RAW-Metrik-Durchschnitte ──")
        metrics_keys = ["noise_score", "sharpness_score", "exposure_score",
                        "detail_score", "defect_score"]
        for key in metrics_keys:
            s_vals = [getattr(m, key) for _, m in select_metrics]
            r_vals = [getattr(m, key) for _, m in reject_metrics]
            s_avg = sum(s_vals) / len(s_vals) if s_vals else 0
            r_avg = sum(r_vals) / len(r_vals) if r_vals else 0
            diff_val = s_avg - r_avg
            direction = "besser" if diff_val > 0 else "schlechter"
            print(f"  {key:20s}: Select={s_avg:5.1f}  Reject={r_avg:5.1f}  "
                  f"(Select ist {abs(diff_val):.1f} Punkte {direction})")

    # pp3-Differenz-Vergleich
    if select_pp3_diffs and reject_pp3_diffs:
        print(f"\n── .pp3-Anpassungsmuster ──")
        print(f"  (alignment .pp3 minus untouched/Standard .pp3)")

        # Aggregate which adjustments are more common in select
        s_adj_count = {}
        r_adj_count = {}
        for _, diff in select_pp3_diffs:
            for section, keys in diff.items():
                for key in keys:
                    k = f"{section}/{key}"
                    s_adj_count[k] = s_adj_count.get(k, 0) + 1

        for _, diff in reject_pp3_diffs:
            for section, keys in diff.items():
                for key in keys:
                    k = f"{section}/{key}"
                    r_adj_count[k] = r_adj_count.get(k, 0) + 1

        all_keys = set(s_adj_count.keys()) | set(r_adj_count.keys())
        for k in sorted(all_keys):
            s_pct = s_adj_count.get(k, 0) / len(select_pp3_diffs) * 100
            r_pct = r_adj_count.get(k, 0) / len(reject_pp3_diffs) * 100
            delta = s_pct - r_pct
            if abs(delta) > 10:  # only report meaningful differences
                pref = "MEHR" if delta > 0 else "WENIGER"
                print(f"  {k:30s}: Select {s_pct:5.0f}%  Reject {r_pct:5.0f}%  "
                      f"(→ {abs(delta):.0f}% {pref} in Select)")

    # ── Beispiele ─────────────────────────────────────────────────────────
    print(f"\n── Beispielbilder ──")

    if select_metrics:
        s_sorted_noise = sorted(select_metrics, key=lambda x: x[1].noise_score, reverse=True)
        print(f"\n  Select — beste Rauschwerte:")
        for raw, m in s_sorted_noise[:5]:
            diff_desc = "keine .pp3-Änderung"
            for r, d in select_pp3_diffs:
                if r == raw:
                    diff_desc = describe_pp3_diff(d)
                    break
            print(f"    {raw.name:30s} noise={m.noise_score:.0f}  sharp={m.sharpness_score:.0f}  "
                  f"({diff_desc})")

    if reject_metrics:
        r_sorted_noise = sorted(reject_metrics, key=lambda x: x[1].noise_score)
        print(f"\n  Reject — schlechteste Rauschwerte:")
        for raw, m in r_sorted_noise[:5]:
            diff_desc = "keine .pp3-Änderung"
            for r, d in reject_pp3_diffs:
                if r == raw:
                    diff_desc = describe_pp3_diff(d)
                    break
            print(f"    {raw.name:30s} noise={m.noise_score:.0f}  sharp={m.sharpness_score:.0f}  "
                  f"({diff_desc})")

    # ── Muster-Zusammenfassung ───────────────────────────────────────────
    print(f"\n── Erkannte Muster ──")

    # Hier sammeln wir qualitativ, ob der User systematisch bestimmte
    # Eigenschaften bevorzugt. Keine harten Grenzwerte – das ist
    # Interpretation, keine Metrik.

    patterns = []

    if select_metrics and reject_metrics:
        avg_s_noise = sum(getattr(m, "noise_score") for _, m in select_metrics) / len(select_metrics)
        avg_r_noise = sum(getattr(m, "noise_score") for _, m in reject_metrics) / len(reject_metrics)
        delta_noise = avg_s_noise - avg_r_noise

        avg_s_sharp = sum(getattr(m, "sharpness_score") for _, m in select_metrics) / len(select_metrics)
        avg_r_sharp = sum(getattr(m, "sharpness_score") for _, m in reject_metrics) / len(reject_metrics)
        delta_sharp = avg_s_sharp - avg_r_sharp

        avg_s_exp = sum(getattr(m, "exposure_score") for _, m in select_metrics) / len(select_metrics)
        avg_r_exp = sum(getattr(m, "exposure_score") for _, m in reject_metrics) / len(reject_metrics)
        delta_exp = avg_s_exp - avg_r_exp

        avg_s_def = sum(getattr(m, "defect_score") for _, m in select_metrics) / len(select_metrics)
        avg_r_def = sum(getattr(m, "defect_score") for _, m in reject_metrics) / len(reject_metrics)
        delta_def = avg_s_def - avg_r_def

        avg_s_detail = sum(getattr(m, "detail_score") for _, m in select_metrics) / len(select_metrics)
        avg_r_detail = sum(getattr(m, "detail_score") for _, m in reject_metrics) / len(reject_metrics)
        delta_detail = avg_s_detail - avg_r_detail

        if delta_noise > 3:
            patterns.append(f"Rauschen: User bevorzugt Bilder mit weniger Rauschen "
                           f"(⌀ Select {avg_s_noise:.0f} vs ⌀ Reject {avg_r_noise:.0f})")
        elif delta_noise < -3:
            patterns.append(f"Rauschen: User toleriert höheres Rauschen in Select-Bildern "
                           f"(⌀ Select {avg_s_noise:.0f} vs ⌀ Reject {avg_r_noise:.0f})")

        if delta_sharp > 3:
            patterns.append(f"Schärfe: User bevorzugt schärfere Bilder "
                           f"(⌀ Select {avg_s_sharp:.0f} vs ⌀ Reject {avg_r_sharp:.0f})")
        elif delta_sharp < -3:
            patterns.append(f"Schärfe: User akzeptiert auch weichere Bilder "
                           f"(⌀ Select {avg_s_sharp:.0f} vs ⌀ Reject {avg_r_sharp:.0f})")

        if delta_exp > 5:
            patterns.append(f"Belichtung: Select-Bilder sind tendenziell besser belichtet "
                           f"(⌀ {avg_s_exp:.0f} vs Reject ⌀ {avg_r_exp:.0f})")
        elif delta_exp < -5:
            patterns.append(f"Belichtung: User wählt auch schlechter belichtete Bilder "
                           f"(vielleicht wegen Komposition)")

        if delta_def > 3:
            patterns.append(f"Fehler/CA: User bevorzugt Bilder mit weniger Linsenfehlern")
        elif delta_def < -3:
            patterns.append(f"Fehler/CA: User ignoriert Linsenfehler bei der Auswahl")

        if delta_detail > 3:
            patterns.append(f"Detailreichtum: User bevorzugt detailreichere Bilder")

    # pp3-basierte Muster
    if select_pp3_diffs and reject_pp3_diffs:
        s_has_contrast = sum(
            1 for _, d in select_pp3_diffs
            if "Exposure" in d and "Contrast" in d["Exposure"]
        )
        r_has_contrast = sum(
            1 for _, d in reject_pp3_diffs
            if "Exposure" in d and "Contrast" in d["Exposure"]
        )
        s_contrast_pct = s_has_contrast / max(len(select_pp3_diffs), 1) * 100
        r_contrast_pct = r_has_contrast / max(len(reject_pp3_diffs), 1) * 100
        if abs(s_contrast_pct - r_contrast_pct) > 15:
            pref = "mehr" if s_contrast_pct > r_contrast_pct else "weniger"
            patterns.append(f"Kontrast: User wendet {pref} Kontrast-Anpassungen in Select an "
                           f"(Select {s_contrast_pct:.0f}% vs Reject {r_contrast_pct:.0f}%)")

        s_has_sh = sum(
            1 for _, d in select_pp3_diffs
            if "Shadows/Highlights" in d
        )
        r_has_sh = sum(
            1 for _, d in reject_pp3_diffs
            if "Shadows/Highlights" in d
        )
        s_sh_pct = s_has_sh / max(len(select_pp3_diffs), 1) * 100
        r_sh_pct = r_has_sh / max(len(reject_pp3_diffs), 1) * 100
        if abs(s_sh_pct - r_sh_pct) > 15:
            pref = "mehr" if s_sh_pct > r_sh_pct else "weniger"
            patterns.append(f"Tonwertrettung: User nutzt {pref} Shadows/Highlights in Select "
                           f"(Select {s_sh_pct:.0f}% vs Reject {r_sh_pct:.0f}%)")

        s_has_sat = sum(
            1 for _, d in select_pp3_diffs
            if "Exposure" in d and "Saturation" in d["Exposure"]
        )
        r_has_sat = sum(
            1 for _, d in reject_pp3_diffs
            if "Exposure" in d and "Saturation" in d["Exposure"]
        )
        s_sat_pct = s_has_sat / max(len(select_pp3_diffs), 1) * 100
        r_sat_pct = r_has_sat / max(len(reject_pp3_diffs), 1) * 100
        if abs(s_sat_pct - r_sat_pct) > 15:
            pref = "mehr" if s_sat_pct > r_sat_pct else "weniger"
            patterns.append(f"Sättigung: User passt {pref} Sättigung in Select an "
                           f"(Select {s_sat_pct:.0f}% vs Reject {r_sat_pct:.0f}%)")

        s_has_exposure = sum(
            1 for _, d in select_pp3_diffs
            if "Exposure" in d and "Exposure" in d["Exposure"]
        )
        r_has_exposure = sum(
            1 for _, d in reject_pp3_diffs
            if "Exposure" in d and "Exposure" in d["Exposure"]
        )
        s_exp_pct = s_has_exposure / max(len(select_pp3_diffs), 1) * 100
        r_exp_pct = r_has_exposure / max(len(reject_pp3_diffs), 1) * 100
        if abs(s_exp_pct - r_exp_pct) > 15:
            pref = "mehr" if s_exp_pct > r_exp_pct else "weniger"
            patterns.append(f"Helligkeit: User korrigiert {pref} die Belichtung in Select "
                           f"(Select {s_exp_pct:.0f}% vs Reject {r_exp_pct:.0f}%)")

    if patterns:
        for p in patterns:
            print(f"  • {p}")
    else:
        print(f"  Keine eindeutigen Muster erkannt (zu ähnlich oder zu kleine Stichprobe)")

    print(f"\n{'='*70}\n")


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
        print(f"\n{'='*70}")
        print(f"  DRY RUN — analyze_workflow.py")
        print(f"{'='*70}")
        print(f"  Batch: {batch}")
        print(f"  Nextcloud-Pfad: {NC_PROJ_PATH}/")
        print(f"    {untouched_dir_name}/")
        print(f"    {alignment_dir_name}/")
        print(f"    {select_dir_name}/")
        print(f"    {reject_dir_name}/")
        print(f"  Lokales Temp: {local_base}")
        print(f"\n  Würde herunterladen, Metriken berechnen und Muster erkennen.")
        print(f"  Keine Änderungen an Pipeline oder Selector.\n")
        return

    if args.local:
        logger.info(f"Lokaler Modus: suche Dateien in {local_base}")
    else:
        nc = init_nextcloud()
        if not nc:
            logger.error("Kein Nextcloud-Zugang. --local verwenden oder ~/.env prüfen.")
            return

        if local_base.exists():
            shutil.rmtree(local_base)

        print(f"\nLade Workflow-Ordner aus Nextcloud...")
        for dir_name in [untouched_dir_name, alignment_dir_name]:
            remote = f"{NC_PROJ_PATH}/{dir_name}"
            local = local_base / dir_name
            count = download_folder(nc, remote, local)
            print(f"  {dir_name}: {count} Dateien")

        # select/reject are directly in alignment/ subdir
        for dir_name in [select_dir_name, reject_dir_name]:
            remote = f"{NC_PROJ_PATH}/{dir_name}"
            local = local_base / dir_name
            count = download_folder(nc, remote, local)
            print(f"  {dir_name}: {count} Dateien")

    # Verify folders exist
    for name, path in [
        (untouched_dir_name, local_untouched),
        (alignment_dir_name, local_alignment),
        (select_dir_name, local_select),
        (reject_dir_name, local_reject),
    ]:
        if not path.exists() or not any(path.iterdir()):
            logger.warning(f"Ordner ist leer oder fehlt: {path}")

    # Run analysis
    analyze_patterns(
        select_dir=local_select,
        reject_dir=local_reject,
        alignment_pp3_dir=local_alignment,
        untouched_pp3_dir=local_untouched,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    main()
