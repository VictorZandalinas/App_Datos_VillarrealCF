# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sports analytics dashboard for Villarreal CF first team. Aggregates data from three sources (Opta, MediaCoach, Sportian), generates PDF performance reports, and provides a web dashboard for match analysis.

**Tech Stack**: Python 3.11+, Dash, Pandas, Matplotlib, mplsoccer, PyPDF2. Node.js used for MediaCoach extraction (xlsx, parquetjs).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server (http://localhost:8050)
python app.py

# Production server
gunicorn app:server --bind 0.0.0.0:8050

# MediaCoach Node.js extraction (optional)
cd extraccion_mediacoach && npm install && cd ..
```

## Architecture

### Data Flow
```
External APIs (Opta, MediaCoach, Sportian)
    → extraccion_*/  (ETL pipelines)
    → Parquet files (columnar storage)
    → app.py (Dash dashboard)
    → PDF reports (via report generator scripts)
```

### Report Generation System

Three report types, each with a master orchestrator script:
- **Set Pieces (ABP)**: `abp_informe_todo.py` orchestrates `abp1-26*.py`
- **Physical Analysis**: `fisico_completo_pdf.py` orchestrates `fisico1-14*.py`
- **Tactical Analysis**: `tactic_informe_todo.py` orchestrates `tactic1-8*.py`

Each analysis script generates individual PDFs that get merged by the master script.

### Key Files
- `app.py` - Main Dash application with login, dashboard, and report generation UI
- `actualizar_datos.py` - Data update orchestration for all three sources
- `extraccion_opta/` - Opta API integration (OAuth)
- `extraccion_mediacoach/` - MediaCoach pipeline (Python + Node.js)
- `extraccion_sportian/` - Sportian CSV to Parquet conversion

### Data Storage
All data stored as Parquet files:
- `extraccion_opta/datos_opta_parquet/abp_events.parquet` - Set pieces
- `extraccion_mediacoach/data/estadisticas_equipo.parquet` - MediaCoach stats
- `extraccion_sportian/corners_tracking.parquet` - Corner tracking

Parquet files must contain a column named `week`, `jornada`, or `Jornada` for gameweek filtering.

## Environment Variables

```bash
APP_USERNAME  # Dashboard login username (default: 'admin')
APP_PASSWORD  # Dashboard login password
```

## Deployment

GitHub Actions deploys to production server (154.56.153.161) on push to main:
1. Auto-saves server-generated data
2. Pulls latest code
3. Installs dependencies
4. Restarts `villarreal.service` (systemd)

## Code Patterns

- Progress tracking via global `progress_data` dict for UI updates
- `@lru_cache` on `obtener_resumen_datos_cached()` for Parquet caching
- Threading for long-running tasks (data updates, report generation)
- Emoji-prefixed logging comments throughout codebase
