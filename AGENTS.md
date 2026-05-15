# Stock Photo Creator v2 — AGENTS.md

## Project: stock-photo-creator-v2

Modulare, testgetriebene Neufassung der Stock Photo Pipeline.

### Architektur

Jeder Pipeline-Schritt ist ein eigenes Modul mit:
- Klarer Input/Output-Schnittstelle
- Eigenen Unit Tests
- Eigener Fehlerbehandlung + Logging
- Unabhängig vom Rest testbar

### Modul-Übersicht

| Modul | Datei | Aufgabe |
|-------|-------|---------|
| 01 | `modules/importer.py` | RAWs aus Nextcloud laden |
| 02 | `modules/bracket_detector.py` | AEB vs. Burst vs. Single erkennen |
| 03 | `modules/overexposure_checker.py` | Clipping prüfen, unrettbarbare löschen | ✅ |
| 04 | `modules/exposure_aligner.py` | Belichtung angleichen (Highlights/Helligkeit) | ✅ |
| 05 | `modules/quality_scorer.py` | KI-Qualitätsbewertung | ✅ |
| 06 | `modules/selector.py` | Beste Bilder auswählen → `RAW/cleaned/` | ✅ |
| 07 | `modules/raw_developer.py` | RAW → JPEG mit pp3-Profilen |
| 08 | `modules/post_processor.py` | EXIF-Rotation, sRGB, Upscale, Straighten, Crop |
| 09 | `modules/metadata_generator.py` | GPT Vision: Szene, Titel, Keywords |
| 10 | `modules/metadata_writer.py` | exiftool + JSON Sidecar |
| 11 | `modules/uploader.py` | Nextcloud WebDAV Upload |

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
| Bracket-Erkennung | Zwei Modi: AEB (EV unterschiedlich) vs. Burst (EV gleich) |
| Erkennungskriterium | EXIF ExposureCompensation + 0-3s Zeitfenster |
| Belichtungskorrektur | Automatisch im ersten Durchlauf |
| Qualitätsvergleich | KI-basierte Bewertung (GPT/Gemini Vision) |
| Bereinigungsergebnis | Nextcloud `RAW/{batch}/cleaned/` |
| Modul-Struktur | Ein Modul pro Schritt |

### Testing

```bash
# Alle Tests
python3 -m unittest discover tests -v

# Einzelnes Modul
python3 -m unittest tests.test_02_bracket_detector -v
```

### CLI Usage

```bash
# Nextcloud mode (default)
python3 main.py Barcelona_Trip

# Mit Limit (für Testing)
python3 main.py Barcelona_Trip --max-images 10

# Dry run
python3 main.py Barcelona_Trip --dry-run
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
  output/                  ← Output Durchlauf 2
    {batch}_minimal/
    {batch}_medium/
    {batch}_full/
  rejected/                ← Abgelehnte Bilder
  profiles/                ← pp3 Profile
```

### Altes Script

Das alte `photo_pipeline.py` bleibt im `stock-photo-creator/` Ordner erhalten.
Teile davon werden wiederverwendet (NextcloudClient, EXIF-Leselogik, etc.).
