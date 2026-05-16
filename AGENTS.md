# Stock Photo Creator v2 — AGENTS.md

## Project: stock-photo-creator-v2

Modulare, testgetriebene Neufassung der Stock Photo Pipeline.

**Repo:** https://github.com/agentomaniac1o0/stock-photo-creator-v2

### Architektur

Jeder Pipeline-Schritt ist ein eigenes Modul mit:
- Klarer Input/Output-Schnittstelle
- Eigenen Unit Tests
- Eigener Fehlerbehandlung + Logging
- Unabhängig vom Rest testbar

### Modul-Übersicht

| Modul | Datei | Aufgabe | Status |
|-------|-------|---------|--------|
| 01 | `modules/importer.py` | RAWs aus Nextcloud laden | ✅ |
| 02 | `modules/bracket_detector.py` | AEB vs. Burst vs. Single erkennen | ✅ |
| 03 | `modules/overexposure_checker.py` | Clipping prüfen, unrettbarbare löschen | ✅ |
| 04 | `modules/exposure_aligner.py` | Belichtung angleichen (Highlights/Helligkeit) | ✅ |
| 05 | `modules/quality_scorer.py` | KI-Qualitätsbewertung (rawpy für volles RAW) | ✅ |
| 06 | `modules/selector.py` | Beste Bilder auswählen → `RAW/cleaned/` | ✅ |
| 07 | `modules/raw_developer.py` | RAW → JPEG mit pp3-Profilen | ⏳ |
| 08 | `modules/post_processor.py` | EXIF-Rotation, sRGB, Upscale, Straighten, Crop | ⏳ |
| 09 | `modules/metadata_generator.py` | GPT Vision: Szene, Titel, Keywords | ⏳ |
| 10 | `modules/metadata_writer.py` | exiftool + JSON Sidecar | ⏳ |
| 11 | `modules/uploader.py` | Nextcloud WebDAV Upload | ⏳ |

### Pipeline Flow

**Durchlauf 1 — Bereinigung (v3 — Multi-Stage Filter):**
```
Nextcloud RAW/{batch}/
    │
    ├─→ 01_importer: Download → lokales Temp-Dir
    ├─→ 02_bracket_detector: AEB / Burst / Single gruppieren + Action-Flag
    ├─→ 03_overexposure_checker: Unrettbar überbelichtete löschen
    ├─→ 04_exposure_aligner: ALLE AEB-Bilder an Referenz angleichen
    │   └─→ 3-Bild-AEB: mittleres Bild als Referenz
    │   └─→ <3 Bilder: bestes Histogramm als Referenz
    ├─→ 05_quality_scorer: Metriken berechnen (kein Gesamtscore!)
    │   └─→ noise, sharpness, defects, exposure, detail einzeln
    ├─→ 06_selector: Multi-Stage Filter-Pipeline
    │   └─→ Schritt 1: Harte Filter (sharp < 10 → reject)
    │   └─→ Schritt 2: Vergleich (wenigstes Rauschen + wenigste Fehler)
    │   └─→ Schritt 3: Auswahl
    │       └─→ AEB: 1 Bild gewinnt
    │       └─→ Burst normal: 1 Bild gewinnt
    │       └─→ Burst Action (ExpTime < 1/250s): alle die Filter bestehen
    │       └─→ Singles: behalten wenn nicht zu unscharf + moderates Rauschen
    │
    └─→ Upload → Nextcloud RAW/{batch}/selected-phase_1/
        └─→ Upload → Nextcloud RAW/{batch}/rejected-phase_1/
```

**Durchlauf 2 — Entwicklung (nach Freigabe):**
```
Nextcloud RAW/{batch}/cleaned/
    │
    ├─→ 07_raw_developer: RAW → JPEG (minimal/medium/full)
    ├─→ 08_post_processor: Straighten, Crop, Upscale
    ├─→ 09_metadata_generator: Szene, Titel, Keywords
    ├─→ 10_metadata_writer: exiftool + Sidecar
    ├─→ 11_uploader: Nextcloud output/{batch}/
    │
    └─→ Cleanup
```

### Key Decisions

| Entscheidung | Wahl |
|-------------|------|
| Bracket-Erkennung | Zwei Modi: AEB (unterschiedliche ExposureTime/EV) vs. Burst (gleiche EV) |
| Erkennungskriterium | EXIF ExposureCompensation (Fraction-Parsing) + ExposureTime + 0-3s Zeitfenster |
| Belichtungsangleich | ALLE AEB-Bilder an Referenz (nicht nur <-0.5 EV) |
| Referenz-Bild | 3-Bild-AEB: mittleres Bild, <3 Bilder: bestes Histogramm |
| Qualitätsvergleich | Metrik-basiert (kein Gesamtscore): Noise primär, Defects sekundär, Sharpness Tie-Breaker |
| Bereinigungsergebnis | Nextcloud `RAW/{batch}/selected-phase_1/` + `RAW/{batch}/rejected-phase_1/` |
| Modul-Struktur | Ein Modul pro Schritt |
| EXIF-Keys | exiftool -json gibt Keys OHNE "EXIF:" Präfix zurück |
| CR2-Rendering | rawpy für volles RAW-Rendering (nicht PIL Thumbnail) |
| Hard Filter | sharpness < 10 → reject (zu unscharf für Nachschärfung) |
| Noise-Vergleich | noise_curve() mit diminishing returns ab 70, Penalty ab 40 |
| Sharpness-Vergleich | sharpness_curve() mit Bonus ab 35, Penalty unter 15 |
| Exposure-Penalty | -5 Punkte für exposure-corrected Bilder (aufgehellt) |
| Action-Erkennung | ExposureTime < 1/250s → alle Burst-Bilder behalten die Filter bestehen |
| Burst-Auswahl | Normal: 1 bestes Bild; Action: alle die Filter bestehen |

### Testing

```bash
# Alle Tests
.venv/bin/python -m unittest discover tests -v

# Einzelnes Modul
.venv/bin/python -m unittest tests.test_02_bracket_detector -v
```

### CLI Usage

```bash
# Nextcloud mode (default)
.venv/bin/python3 main.py Barcelona_Trip

# Mit Limit (für Testing)
.venv/bin/python3 main.py Barcelona_Trip --max-images 10

# Dry run
.venv/bin/python3 main.py Barcelona_Trip --dry-run

# Test-Run (30 Bilder)
.venv/bin/python3 test_run.py
```

### Nextcloud Verzeichnisstruktur

```
Nextcloud: /Photos/StockFotoCreator/
  RAW/
    {batch}/
      IMG_001.CR2          ← Input
      IMG_002.CR2
      selected-phase_1/    ← Output Phase 1 (ausgewählte RAWs)
        IMG_002.CR2
      rejected-phase_1/    ← Abgelehnte RAWs
      phase_1_report.json   ← Report mit Scores
      test_run_01/          ← Test-Run Ergebnisse
  output/                   ← Output Phase 2
    {batch}_minimal/
    {batch}_medium/
    {batch}_full/
  profiles/                 ← pp3 Profile
```

### Real-Life Test Results (2026-05-15)

**Batch:** SW-England-May26-01 (180 CR2-Dateien, Canon EOS M6)

| Metrik | v1 | v2 | v3 (Pipeline-Redesign) |
|--------|----|----|------------------------|
| Overexposure Check | 2 rejected | 2 rejected | 2 rejected |
| Selektion | 69 selected | 66 selected | TBD |
| Upload | 178/178 | 178/178 | TBD |
| Unscharfe Gewinner | 2 (1659, 1698) | 0 | 0 |

**Pipeline-Redesign v3 (2026-05-15):**
- Kompletter Umbau von Score-basiert zu Multi-Stage Filter
- Schritt 1: Harte Filter (sharp < 10 → reject)
- Schritt 2: Belichtungsangleich (alle AEB-Bilder an Referenz)
- Schritt 3: Vergleich (wenigstes Rauschen + wenigste Fehler gewinnt)
- Burst normal: 1 bestes Bild; Burst Action (ExpTime < 1/250s): alle behalten
- Kein Gesamtscore mehr — Metriken werden einzeln genutzt
- Referenz-Bild: mittleres bei 3-Bild-AEB, bestes Histogramm bei <3
- Exposure-Correction-Penalty: -5 Punkte für aufgehellte Bilder

**Bug-Fix-Session (2026-05-16):**

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | CRITICAL | `SINGLE_NOISE_MAX` Logik invertiert — rejectet saubere Singles | `noise_score > 80` → `noise_score < 30` (NOISE_GATE_MIN) |
| 2 | MEDIUM | Sortierung in `compute_all_metrics` nach raw scores statt Kurven | Nutzt jetzt `noise_curve()`, `defect_score`, `sharpness_curve()` |
| 3 | LOW | `SINGLE_NOISE_MAX` definiert aber nie genutzt | Entfernt, `NOISE_GATE_MIN` wird jetzt für Singles genutzt |
| 4 | MEDIUM | Duplizierte Selektions-Logik in `run_phase_1.py` | Ersetzt durch `select_and_upload()` Aufruf |
| 5 | LOW | Fragile Fehlererkennung (`== 50.0`) | Prüft jetzt alle 5 Metriken auf Default-Werte |
| 6 | LOW | Default `1/100` Exposure-Time konnte Referenz verfälschen | Fallback auf Helligkeits-Schätzung wenn EXIF fehlt |

### Workflow-Analyse (2026-05-16)

**Ziel:** Kalibrierung der Selection-Logik durch Analyse eines manuell kuratierten Raw Therapee-Workflows.

**Methode:** `analyze_workflow.py` (NEU, getrennt vom Pipeline-Code) analysiert 4 Nextcloud-Ordner:
- `select-pipe-proj/{batch}-untouched/` — RAWs mit Standard/neutral .pp3
- `select-pipe-proj/{batch}-alignment/` — RAWs + manuell bearbeitete .pp3
- `select-pipe-proj/alignment/user-select/` — User-Favoriten (RAWs + .pp3)
- `select-pipe-proj/alignment/user-reject/` — Aussortierte (RAWs + .pp3)

**Analyse-Instrumente:**
- RAW-Metriken: `quality_scorer.compute_metrics()` (noise, sharpness, exposure, defects, detail)
- .pp3-Differenz: `alignment.pp3 - untouched.pp3` → Helligkeit, Highlights, Schatten, Kontrast, Sättigung
- `bracket_detector.detect_brackets()` für Gruppenbildung

**.pp3 Struktur (Raw Therapee, INI-ähnlich):**

| Section | Relevante Keys | Bedeutung |
|---------|---------------|-----------|
| `[Exposure]` | `Exposure`, `Black`, `Contrast`, `Saturation`, `HighlightCompression`, `ShadowCompression`, `Lightness` | Grundbelichtung |
| `[Shadows/Highlights]` | `Highlights`, `Shadows` (nur wenn `Enabled=true`) | Tonwertrettung |
| `[Local Contrast]` | `Amount` (nur wenn `Enabled=true`) | Lokaler Kontrast |
| `[Vibrance]` | `Pastels`, `Saturated` (nur wenn `Enabled=true`) | Farbsättigung |
| `[Sharpening]` | `Amount`, `Radius` | Nachschärfung |

**Report-Format:** Klartext-Muster (z.B. "User wählt Bilder mit mehr Kontrast und weniger Rauschen") + 3-5 konkrete Dateinamen als Beispiele.

**Next Steps:**
1. User erstellt Ordner in Nextcloud: `Photos/StockFotoCreator/select-pipe-proj/SW-England-May26-01-untouched/`, `-alignment/`, `alignment/user-select/`, `alignment/user-reject/`
2. `analyze_workflow.py` ausführen
3. Erkenntnisse validieren → neue Thresholds in `selector.py`

### Altes Script

Das alte `photo_pipeline.py` bleibt im `stock-photo-creator/` Ordner erhalten.
Teile davon werden wiederverwendet (NextcloudClient, EXIF-Leselogik, etc.).
