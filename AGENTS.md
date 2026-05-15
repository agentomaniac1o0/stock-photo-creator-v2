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

**Durchlauf 1 — Bereinigung:**
```
Nextcloud RAW/{batch}/
    │
    ├─→ 01_importer: Download → lokales Temp-Dir
    ├─→ 02_bracket_detector: AEB / Burst / Single gruppieren
    ├─→ 03_overexposure_checker: Unrettbar überbelichtete löschen
    ├─→ 04_exposure_aligner: Belichtung angleichen
    ├─→ 05_quality_scorer: KI-Qualitätsbewertung
    ├─→ 06_selector: Beste auswählen
    │
    └─→ Upload → Nextcloud RAW/{batch}/cleaned/ (zur Qualitätskontrolle)
        └─→ Upload → Nextcloud RAW/{batch}/rejected/
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
| Belichtungskorrektur | Automatisch im ersten Durchlauf (rawpy für RAW, Pillow für JPEG) |
| Qualitätsvergleich | KI-basiert (OpenAI/Gemini) mit rawpy Fallback (volles RAW-Rendering) |
| Bereinigungsergebnis | Nextcloud `RAW/{batch}/cleaned/` + `RAW/{batch}/rejected/` |
| Modul-Struktur | Ein Modul pro Schritt |
| EXIF-Keys | exiftool -json gibt Keys OHNE "EXIF:" Präfix zurück |
| CR2-Rendering | rawpy für volles RAW-Rendering (nicht PIL Thumbnail) |

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
      cleaned/             ← Output Durchlauf 1 (zur Qualitätskontrolle)
        IMG_002.CR2        ← Ausgewählte/korrigierte RAWs
      rejected/            ← Abgelehnte RAWs
      selection_report.json ← Report mit Scores
      test_run_01/         ← Test-Run Ergebnisse
  output/                  ← Output Durchlauf 2
    {batch}_minimal/
    {batch}_medium/
    {batch}_full/
  profiles/                ← pp3 Profile
```

### Real-Life Test Results (2026-05-15)

**Batch:** SW-England-May26-01 (30 CR2-Dateien, Canon EOS M6)

| Metrik | Ergebnis |
|--------|----------|
| Bracket Detection | 9 AEB-Gruppen + 1 Burst-Gruppe (vorher: 30 Singles) |
| Overexposure Check | 0 rejected (keine unrettbaren Überbelichtungen) |
| Quality Scores (rawpy) | 34-58/100 (vorher mit PIL: 31-44) |
| Selektion | 12 kept, 18 rejected |
| Upload | 30/30 erfolgreich |

**Bug-Fixes während Test:**
1. EXIF-Key Bug: exiftool -json gibt Keys ohne "EXIF:" Präfix zurück
2. Fraction-Parsing: ExposureCompensation "-1/3" → -0.33
3. rawpy für volles CR2-Rendering statt PIL Thumbnail (160x120)
4. ExposureTime-basierte Bracket-Erkennung wenn EV-Werte identisch

### Altes Script

Das alte `photo_pipeline.py` bleibt im `stock-photo-creator/` Ordner erhalten.
Teile davon werden wiederverwendet (NextcloudClient, EXIF-Leselogik, etc.).
