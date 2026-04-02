# 🚦 Swiss Traffic MLOps Pipeline
### Full Data Engineering + Machine Learning Project — French-speaking Switzerland (Romandy)
#### Data Source: Federal Roads Office (FEDRO) · SARTC 2025 Annual Bulletin

---

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [Data Sufficiency Assessment](#data-sufficiency-assessment)
3. [Why Lausanne?](#why-lausanne)
4. [Architecture Overview](#architecture-overview)
5. [Project Structure](#project-structure)
6. [File-by-File Reference](#file-by-file-reference)
7. [Database Schema](#database-schema)
8. [Pipeline Stages Explained](#pipeline-stages-explained)
9. [Machine Learning Approach](#machine-learning-approach)
10. [How to Run](#how-to-run)
    - [Option A — Local (Python + Bruin)](#option-a--local-python--bruin)
    - [Option B — Docker (recommended)](#option-b--docker-recommended)
11. [Output Files](#output-files)

---

## What This Project Does

> **Plain English:** This project takes raw traffic sensor data from Swiss roads, cleans it, stores it in a database, trains a machine learning model to predict future traffic volumes, and generates beautiful charts and reports — all automatically, repeatable, and production-grade.

Think of the Swiss highway network as a giant web of sensors. Every hour, each sensor counts every vehicle that passes by. The Federal Roads Office (FEDRO / ASTRA) collects this data from **~200 automatic measuring stations** and publishes annual statistical bulletins.

We take those bulletins and build a **full MLOps pipeline**:

| Stage | What Happens | Like This |
|-------|-------------|-----------|
| **Ingestion** | Read messy CSVs → clean database tables | A translator converts foreign documents into your language |
| **Staging** | Validate and normalize data | A quality inspector checks every product before it enters the warehouse |
| **Transformation** | Engineer ML features | A chef preps all ingredients before cooking |
| **Reporting** | Generate historical charts + HTML reports | An analyst writes the annual company report |
| **ML Training** | Train prediction models | A student studies past exam questions to prepare |
| **ML Evaluation** | Test on unseen data | The student takes the real exam |
| **Predictions** | Forecast future traffic | The student predicts next year's exam results |

---

## Data Sufficiency Assessment

### What We Have
```
FILES: 4 CSV files from FEDRO/ASTRA Swiss Annual Bulletin 2025
─────────────────────────────────────────────────────────────────────────
Annual_results_2025.csv                   │ ~200 stations × 5 metrics × 12 months
Annual_results_Measuring_station_2025.csv │ Data quality notes (gaps/closures)
Annual_results_Measuring_station_         │ Same as above + vehicle class breakdown
  adtwithclasses2025.csv                  │ (buses, motorcycles, trucks, articulated...)
Annual_results_Measuring_station_         │ Legend / glossary for all abbreviations
  legend2025.csv                          │
─────────────────────────────────────────────────────────────────────────
TOTAL RAW ROWS: ~4,500   │   STATIONS: ~200   │   TIME PERIOD: Jan–Dec 2025
```

### Sufficiency by Task

| Task | Sufficient? | Sample Size | Notes |
|------|-------------|-------------|-------|
| Exploratory Data Analysis | ✅ YES | 200 stations × 12 months | Rich patterns visible |
| Historical Reporting | ✅ YES | 12 monthly snapshots | Full year of data |
| Station Clustering | ✅ YES | 200-station feature vectors | Great for k-means/DBSCAN |
| Traffic Classification | ✅ YES | 200+ labeled samples | High/Med/Low volume classes |
| Cross-station Regression | ✅ YES | ~180 complete samples | Predict Oct-Dec from Jan-Sep |
| Time-Series Forecasting | ⚠️ LIMITED | 12 data points/station | Need 3+ years for ARIMA/LSTM |
| Deep Learning Time Series | ❌ NO | Only 12 months total | Need 1000+ time steps |

### Our ML Strategy (Working With What We Have)

Since we only have **1 year** of monthly data, we use a **cross-sectional approach** instead of a traditional time-series approach:

```
TRADITIONAL TIME SERIES (need 3+ years):          OUR APPROACH (works with 1 year):
Station 064                                        Train on: ALL STATIONS, months Jan-Sep
 Jan ─Feb─ Mar─ Apr─ May─ Jun─ Jul─ Aug─ Sep       Test on:  ALL STATIONS, months Oct-Dec
  │                                    │
  └──── predict ──────────────────► Oct?      180 stations × 9 months = 1,620 training samples
                                                180 stations × 3 months = 540 test samples
                                                Total: ~2,160 observations!
```

**Why this works:** Traffic patterns are largely **structural** (related to road type, canton, urbanization level). By learning patterns across 200 stations simultaneously, the model captures these structural relationships and can predict for any station's autumn months.

---

## Why Lausanne?

Among all French-speaking (Romandy) stations, **Lausanne (VD canton)** is the best focus city because:

```
FRENCH-SPEAKING STATIONS RANKED BY DATA QUALITY:
────────────────────────────────────────────────────────────────────────
Station ID │ Name                    │ Canton │ Road │ Annual ADT │ Complete?
─────────────────────────────────────────────────────────────────────────
064        │ CONT. DE LAUSANNE (AR)  │ VD     │ A 9  │ 97,662    │ ✅ All 12 months
043        │ PREVERENGES (AR)        │ VD     │ A 1  │ 97,195    │ ✅ All 12 months
149        │ MEX (AR)                │ VD     │ A 1  │ 74,395    │ ✅ All 12 months
083        │ VILLENEUVE (AR)         │ VD     │ A 9  │ 67,693    │ ✅ All 12 months
069        │ COLOVREX-AEROPORT (AR)  │ GE     │ A 1  │ 64,182    │ ✅ All 12 months
165        │ MARTIGNY N (AR)         │ VS     │ A 9  │ 48,280    │ ✅ All 12 months
059        │ TRAV. DE NEUCHATEL E    │ NE     │ A 5  │ 44,677    │ ✅ All 12 months
────────────────────────────────────────────────────────────────────────
```

**Lausanne wins because:**
- Highest traffic volume among Romandy stations (~97,000 vehicles/day)
- 3 nearby measuring stations provide rich spatial data
- Complete 12-month data (no gaps)
- Located on the A9 motorway (a route nationale of European importance — part of E62)
- Gateway between Swiss plateau and the Alps (Simplon, Grand-St-Bernard)
- Economically representative of a major Swiss city

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║             SWISS TRAFFIC MLOPS — FULL PIPELINE ARCHITECTURE                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │  DATA SOURCES (CSV files from FEDRO / ASTRA)                            │    ║
║  │  Annual_results_2025.csv  │  adtwithclasses2025.csv  │  station_2025.csv│    ║
║  └──────────────────────────────────────┬──────────────────────────────────┘    ║
║                                         │ Bruin Python Asset                    ║
║                                         ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │  LAYER 1: RAW  (DuckDB schema: raw.*)                                    │   ║
║  │  raw.annual_results   │  raw.adtwithclasses  │  raw.station_notes        │   ║
║  │  Exact copy of CSV data — no transformations, not for analysis           │   ║
║  └──────────────────────────────────────┬─────────────────────────────────-┘   ║
║                                         │ Bruin SQL Assets                      ║
║                                         ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │  LAYER 2: STAGING  (DuckDB schema: staging.*)                            │   ║
║  │  staging.stg_stations       │  staging.stg_monthly_traffic               │   ║
║  │  Cleaned, validated, standardized — one row per logical entity           │   ║
║  └──────────────────────────────────────┬──────────────────────────────────┘   ║
║                                         │ Bruin SQL Assets                      ║
║                                         ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │  LAYER 3: MART  (DuckDB schema: mart.*)                                  │   ║
║  │  mart.traffic_features  │  mart.lausanne_analysis  │  mart.romandy_summary│  ║
║  │  Analytics-ready, feature-engineered, business logic applied            │   ║
║  └───────────────────────┬────────────────────────────┬─────────────────────┘  ║
║                           │                            │                         ║
║                           ▼                            ▼                         ║
║  ┌──────────────────────────┐        ┌─────────────────────────────────────┐   ║
║  │  REPORTING               │        │  MACHINE LEARNING                   │   ║
║  │  (Bruin Python Asset)    │        │  (Bruin Python Assets)              │   ║
║  │                          │        │                                     │   ║
║  │  • Seasonal line charts  │        │  Train:    Jan–Sep (9 months)       │   ║
║  │  • Canton heatmap        │        │  Validate: Oct–Dec (3 months held out│  ║
║  │  • Lausanne deep-dive    │        │  Models:   LinearReg, RandomForest, │   ║
║  │  • HTML historical report│        │            GradientBoosting         │   ║
║  │                          │        │  Tracking: MLflow                   │   ║
║  └──────────────────────────┘        └──────────────┬──────────────────────┘  ║
║         │                                           │                           ║
║         ▼                                           ▼                           ║
║  reports/historical_report.html      reports/model_report.html                 ║
║  reports/lausanne_*.png              reports/predictions_*.png                 ║
║  reports/romandy_*.png               models/best_model.pkl                     ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Project Structure

```
swiss_trafficc_data/                          ← Your workspace (project root)
│
├── .bruin.yml                                ← Bruin config: database connections
├── pipeline.yml                              ← Pipeline definition: schedule + DAG
├── pyproject.toml                            ← Python dependencies (for uv/pip)
├── .gitignore                                ← What NOT to commit to git
├── README.md                                 ← This file!
├── PROMPT.md                                 ← Detailed LLM prompt for this project
│
├── assets/                                   ← All Bruin pipeline assets
│   │
│   ├── raw/                                  ← LAYER 1: Ingest raw CSVs
│   │   └── ingest_traffic_csv.py             ← Parse all 4 CSVs → DuckDB raw tables
│   │
│   ├── staging/                              ← LAYER 2: Clean + validate
│   │   ├── stg_stations.sql                  ← One clean row per measuring station
│   │   └── stg_monthly_traffic.sql           ← Normalized: 1 row per station×month
│   │
│   ├── transform/                            ← LAYER 3: Feature engineering + analytics
│   │   ├── traffic_features.sql              ← Wide-format ML feature table
│   │   ├── lausanne_analysis.sql             ← Deep Lausanne/VD canton analysis
│   │   └── romandy_summary.sql               ← French-Swiss canton-level summary
│   │
│   ├── reporting/                            ← Historical analysis + visualization
│   │   └── generate_reports.py               ← Charts (.png) + HTML report
│   │
│   └── ml/                                   ← Machine learning pipeline
│       ├── train_model.py                    ← Train LinearReg + RF + GBM + MLflow
│       ├── evaluate_model.py                 ← Evaluate on Oct–Dec holdout + report
│       └── predict_traffic.py                ← Generate future traffic predictions
│
├── analyses/                                 ← Ad-hoc SQL queries (not in pipeline)
│   └── data_quality_check.sql                ← Manual data quality investigation
│
├── models/                                   ← Saved model files (gitignored)
│   └── .gitkeep
│
└── reports/                                  ← Generated reports + charts (gitignored)
    └── .gitkeep
```

---

## File-by-File Reference

This section explains **every file in the project** — what it is, what it does, what it reads, and what it produces. Read this if you want to understand any individual file without reading the full code.

---

### Root-Level Files

---

#### `run_pipeline.py`
```
Type:     Python script — the manual pipeline runner
Run with: python run_pipeline.py
```
**What it is:** The single entry point to run the entire 7-stage pipeline from scratch. Bruin would normally orchestrate this automatically, but this file lets you run everything with one command using your local Python.

**What it does, step by step:**
1. Calls `ingest_traffic_csv.py` → loads CSVs into DuckDB
2. Calls `stg_stations.sql` → creates `staging.stg_stations` table  
3. Calls `stg_monthly_traffic.sql` → creates `staging.stg_monthly_traffic` table
4. Calls `traffic_features.sql` → creates `mart.traffic_features` table
5. Calls `lausanne_analysis.sql` → creates `mart.lausanne_analysis` table
6. Calls `romandy_summary.sql` → creates `mart.romandy_summary` table
7. Calls `generate_reports.py` → writes PNG charts + HTML report
8. Calls `train_model.py` → trains 3 ML models, saves best one
9. Calls `evaluate_model.py` → scores the model, checks quality gates
10. Calls `predict_traffic.py` → generates Q4 + 2026 predictions, writes CSV

**Key functions:**
- `run_python_asset(filepath)` — runs a `.py` file as a subprocess so it gets its own fresh Python process (avoids import caching issues)
- `run_sql_asset(filepath, con)` — reads a `.sql` file, strips the Bruin metadata header, and executes `CREATE OR REPLACE TABLE ... AS (...)` in DuckDB

**Produces:** prints stage-by-stage progress; all durable output is produced by the assets themselves

---

#### `pipeline.yml`
```
Type:     Bruin configuration YAML
Used by:  bruin run .    (when using Bruin CLI directly)
```
**What it is:** The official Bruin pipeline definition file. It declares the pipeline name, schedule, and all its assets. Bruin reads this to understand the full DAG (execution graph).

**Key concepts inside it:**
- `name:` — pipeline name (`swiss-traffic-mlops`)
- `schedule:` — when to run (set to `@daily` for production; we run manually)
- The asset list mirrors what `run_pipeline.py` does, but Bruin resolves the order automatically from each asset's `depends:` field

**Relationship to `run_pipeline.py`:** Both describe the same pipeline. `pipeline.yml` is for Bruin; `run_pipeline.py` is for running locally without Bruin.

---

#### `pyproject.toml`
```
Type:     Python package manifest (PEP 517/518 standard)
Used by:  pip, uv, Bruin (to install dependencies)
```
**What it is:** The "shopping list" of Python packages this project needs. Modern replacement for `requirements.txt`.

**Key packages and why each one is needed:**

| Package | Why it's needed |
|---------|----------------|
| `pandas` | Read and reshape the CSV data (like programmable Excel) |
| `numpy` | Maths engine used internally by pandas and scikit-learn |
| `duckdb` | Embedded SQL database — stores all pipeline data |
| `scikit-learn` | ML toolkit — Linear Regression, Random Forest, Gradient Boosting |
| `mlflow` | Experiment tracker — logs every training run for comparison |
| `joblib` | Save/load trained models to disk (`.pkl` files) |
| `matplotlib` | Draw charts (bars, lines, scatter plots) |
| `seaborn` | Statistical charts on top of matplotlib (heatmaps, box plots) |
| `jinja2` | HTML report templating engine |

---

#### `Dockerfile`
```
Type:     Docker image definition
Used by:  docker compose build
```
**What it is:** A recipe for building a self-contained Linux box with everything needed to run the pipeline — no local Python installation required.

**What happens when Docker builds this image (in order):**
1. Start from `python:3.11-slim` (Debian Linux + Python, ~200 MB)
2. Install `curl`, `git`, `ca-certificates` (needed to download tools)
3. Download and install **Bruin CLI** from GitHub into `/usr/local/bin/bruin`
4. Download and install **uv** (fast package installer) into `/usr/local/bin/uv`
5. Copy `pyproject.toml` into the image and install all Python packages
6. Copy the entire project code into `/app`
7. Create `reports/`, `models/`, `mlruns/` folders
8. Set environment variables: `MPLBACKEND=Agg` (headless charts), `MLFLOW_TRACKING_URI`
9. Default command: `python run_pipeline.py`

**Layer caching explained:** Docker caches each `RUN` step. Steps 1–5 (installing tools + packages) are only re-run if `pyproject.toml` or the `Dockerfile` changes. Editing Python code only re-runs step 6 (fast COPY).

---

#### `docker-compose.yml`
```
Type:     Docker Compose service definition
Used by:  docker compose up
```
**What it is:** Defines two services that share one Docker image:

| Service | What it runs | When it exits |
|---------|-------------|---------------|
| `pipeline` | `python run_pipeline.py` — the full 7-stage pipeline | When the pipeline finishes |
| `mlflow` | `mlflow ui --host 0.0.0.0 --port 5000` — web experiment dashboard | Stays running until you press Ctrl+C |

**Named volumes** (persist data between runs):
- `duckdb_data` → stores `traffic.duckdb`
- `reports_data` → stores `reports/*.html`, `reports/*.png`
- `models_data` → stores `models/best_model.pkl`
- `mlruns_data` → stores `mlruns/` (MLflow experiment history)

---

#### `traffic.duckdb`
```
Type:     DuckDB database file (binary)
Location: project root
```
**What it is:** A single file that contains the entire database — all three SQL schemas (`raw`, `staging`, `mart`), all tables, and all data. No separate database server is needed; this file IS the database.

**What's inside it after the pipeline runs:**

| Schema | Table | Rows | Description |
|--------|-------|------|-------------|
| `raw` | `annual_results` | ~2,015 | CSV data, exactly as ingested |
| `raw` | `adtwithclasses` | ~1,600 | Vehicle class breakdown |
| `raw` | `station_notes` | ~400 | Data quality flags |
| `staging` | `stg_stations` | ~403 | One clean row per station |
| `staging` | `stg_monthly_traffic` | ~24,180 | One row per station × month × metric |
| `mart` | `traffic_features` | ~330 | ML feature matrix |
| `mart` | `lausanne_analysis` | ~372 | Lausanne/VD deep-dive |
| `mart` | `romandy_summary` | ~6 | Canton-level KPIs |
| `mart` | `predictions` | ~82 | Model Q4 + 2026 predictions |

---

#### `Annual_results_2025.csv` and related CSVs
```
Type:     Source data files from FEDRO/ASTRA
These are READ-ONLY — never modify them
```

| File | Contents |
|------|----------|
| `Annual_results_2025.csv` | Monthly ADT for all ~200 stations, all 5 metric types |
| `Annual_results_ Measuring_station_2025.csv` | Data quality notes (closures, gaps) |
| `Annual_results_ Measuring_station_adtwithclasses2025.csv` | Same + vehicle class breakdown |
| `Annual_results_ Measuring_station_legend2025.csv` | Abbreviation glossary |

**Why are they hard to parse?** They are formatted for Excel readability. Station info (ID, name, canton, road) appears only on the first row of a "block"; the next 4 rows for the same station have empty cells in those columns. The ingestion script handles this with a forward-fill technique.

---

### `assets/` folder

This folder contains all pipeline **assets** — the individual processing steps. Each file is a self-contained unit that does one job.

---

#### `assets/raw/ingest_traffic_csv.py`
```
Bruin asset name:  raw.ingest_traffic_csv
Type:              Python
Reads:             The 4 CSV files in the project root
Writes:            raw.annual_results, raw.adtwithclasses, raw.station_notes  (DuckDB)
```
**What it does:** The hardest parsing job in the project. Reads the messy Excel-style FEDRO CSV files and loads clean records into DuckDB.

**Key functions:**
- `clean_number(value)` — converts `"97,662"` → `97662.0`, handles NaN/empty cells
- `parse_main_csv(filepath)` — reads one CSV file, applies **forward-fill** to propagate station ID/name/canton/road across the block rows that have those cells empty
- `parse_station_notes(filepath)` — reads the data quality notes CSV (simpler structure)
- `load_to_duckdb(df, table_name, con)` — writes a pandas DataFrame into DuckDB

**The forward-fill explained:**
```
CSV line 1:  064 | LAUSANNE | VD | A9 | ADT  | 82000 | ...   ← station data here
CSV line 2:     |          |    |    | AWT  | 91000 | ...   ← station cells EMPTY
CSV line 3:     |          |    |    | ADT Sa| 70000 | ...   ← still empty
```
The script remembers the last seen station ID/name/canton/road and copies it into the empty cells of lines 2 and 3.

---

#### `assets/staging/stg_stations.sql`
```
Bruin asset name:  staging.stg_stations
Type:              DuckDB SQL  →  materialises as TABLE
Reads:             raw.annual_results
Writes:            staging.stg_stations
Rows produced:     ~403 (one per measuring station)
```
**What it does:** Collapses the 5-row-per-station raw format down to 1 row per station. Adds computed classification columns.

**Key columns it adds:**

| Column | How it's computed | Why it's useful |
|--------|------------------|----------------|
| `station_name_clean` | Strip `(AR)`, `(N)` suffixes | Cleaner display names |
| `road_category` | `A%` → "Motorway", `H%` → "National Road" | Groups roads by type |
| `is_romandy` | Canton IN ('VD','GE','NE','JU','FR','VS') | Filter French-speaking stations |
| `months_with_data` | Count of non-NULL month columns | Data completeness score 0–12 |

---

#### `assets/staging/stg_monthly_traffic.sql`
```
Bruin asset name:  staging.stg_monthly_traffic
Type:              DuckDB SQL  →  materialises as TABLE
Reads:             raw.annual_results, staging.stg_stations
Writes:            staging.stg_monthly_traffic
Rows produced:     ~24,180 (403 stations × 5 metrics × 12 months)
```
**What it does:** Converts the raw data from **wide format** (12 month columns per row) to **long format** (one row per station × metric × month). This is called an "unpivot" or "melt" operation.

**Before (wide):**
```
station_id | metric | adt_01 | adt_02 | ... | adt_12
64         | ADT    | 82000  | 79000  | ... | 88000
```
**After (long):**
```
station_id | metric | month_num | month_name | adt_value
64         | ADT    | 1         | January    | 82000
64         | ADT    | 2         | February   | 79000
...
64         | ADT    | 12        | December   | 88000
```

**Technique:** 12 separate `SELECT` statements (one per month) connected with `UNION ALL`. Each SELECT picks one `adt_XX` column and labels it with the corresponding `month_num`, `month_name`, `quarter`, and `season`.

**Also adds:** `quarter` (Q1/Q2/Q3/Q4), `season` (Winter/Spring/Summer/Autumn), `road_category`, `is_romandy`, `months_with_data` — joined from `stg_stations`.

---

#### `assets/transform/traffic_features.sql`
```
Bruin asset name:  mart.traffic_features
Type:              DuckDB SQL  →  materialises as TABLE
Reads:             staging.stg_monthly_traffic, staging.stg_stations
Writes:            mart.traffic_features
Rows produced:     ~330 (one per station — the ML feature matrix)
```
**What it does:** Builds the machine learning feature matrix. This is the table that `train_model.py` reads as its input.

**Schema (simplified):**

| Column group | Columns | Purpose |
|-------------|---------|---------|
| Identifiers | `station_id`, `station_name`, `canton`, `road` | Row labels |
| Features X (train on) | `adt_jan` … `adt_sep` | Monthly traffic Jan–Sep |
| Features X (engineered) | `summer_peak_ratio`, `weekday_weekend_ratio`, `hgv_pct_jul`, `canton_code`, `road_type_code` | Domain-knowledge features |
| Targets y (predict) | `adt_oct`, `adt_nov`, `adt_dec` | What the model predicts |

**Engineering example — `summer_peak_ratio`:**
```sql
COALESCE(adt_jul, adt_aug) / NULLIF(adt_jan, 0)
```
This ratio is high for tourist routes (Valais ski/summer roads) and close to 1.0 for commuter routes (Geneva, Lausanne). The ML model uses this to tell those route types apart.

---

#### `assets/transform/lausanne_analysis.sql`
```
Bruin asset name:  mart.lausanne_analysis
Type:              DuckDB SQL  →  materialises as TABLE
Reads:             staging.stg_monthly_traffic, staging.stg_stations
Writes:            mart.lausanne_analysis
Rows produced:     ~372 (all VD canton station × month rows)
```
**What it does:** Deep analysis focused on Lausanne and the Vaud (VD) canton. Uses SQL **window functions** to compute per-station statistics while keeping every individual row.

**Key window functions used:**

| Expression | Meaning |
|-----------|---------|
| `AVG(adt_value) OVER (PARTITION BY station_id)` | Year average for this station (same value repeated in every row for this station) |
| `adt_value / station_year_avg` | How busy is this month relative to the station's annual average? (>1 = above average) |
| `RANK() OVER (PARTITION BY station_id ORDER BY adt_value DESC)` | Which month is the busiest for this station? |

**Output used for:** Lausanne bar charts and the deep-dive section of the HTML historical report.

---

#### `assets/transform/romandy_summary.sql`
```
Bruin asset name:  mart.romandy_summary
Type:              DuckDB SQL  →  materialises as TABLE
Reads:             staging.stg_monthly_traffic, staging.stg_stations, mart.traffic_features
Writes:            mart.romandy_summary
Rows produced:     ~6 (one per Romandy canton: VD, GE, VS, NE, FR, JU)
```
**What it does:** Aggregates all Romandy stations up to the canton level. Answers questions like: "Which canton has the most traffic?" / "Which is most seasonal?"

**Key KPIs per canton:**

| Column | Meaning |
|--------|---------|
| `avg_annual_adt` | Average of all station annual ADTs in this canton |
| `total_stations` | How many measuring stations in this canton |
| `peak_month` | Which month has the highest average traffic |
| `seasonal_variation_pct` | (max month − min month) / avg × 100 — how "seasonal" the canton is |
| `busiest_station` | Name of the highest-traffic station in the canton |

**Uses 3 CTEs (named sub-queries):** `romandy_long` → `canton_monthly` → `canton_annual`, each building on the previous. This stepwise structure makes the logic easy to follow.

---

#### `assets/reporting/generate_reports.py`
```
Bruin asset name:  reporting.generate_reports
Type:              Python
Reads:             mart.lausanne_analysis, mart.romandy_summary, mart.traffic_features  (DuckDB)
Writes:            reports/lausanne_monthly_traffic.png
                   reports/romandy_seasonal_patterns.png
                   reports/station_heatmap.png
                   reports/canton_comparison.png
                   reports/historical_report.html
```
**What it does:** Turns the database tables into charts and an HTML report that any non-technical stakeholder can open in a browser.

**Charts produced:**

| Chart | Chart type | What it shows |
|-------|-----------|--------------|
| `lausanne_monthly_traffic.png` | Grouped bar chart | Monthly ADT for the 4 Lausanne stations side by side |
| `romandy_seasonal_patterns.png` | Multi-line chart | Seasonal curve (Jan–Dec) for each of the 6 Romandy cantons |
| `station_heatmap.png` | Colour heatmap | All Romandy stations × 12 months — colour = traffic level |
| `canton_comparison.png` | Box-and-whisker | Distribution of ADT values by canton |
| `historical_report.html` | HTML page | All 4 charts + summary statistics in one browser-openable file |

**Key library settings:**
- `matplotlib.use("Agg")` — **must** be set before any chart code; tells matplotlib not to open a GUI window (essential for server/Docker environments)
- `duckdb.connect(..., read_only=True)` — reporting must never write to the database

---

#### `assets/ml/train_model.py`
```
Bruin asset name:  ml.train_model
Type:              Python
Reads:             mart.traffic_features  (DuckDB)
Writes:            models/best_model.pkl        (the winning model)
                   reports/feature_importance.csv
                   reports/training_metrics.json
                   mlruns/                      (MLflow experiment logs)
```
**What it does:** Trains three ML models (Linear Regression, Random Forest, Gradient Boosting) on Jan–Sep traffic data to predict Oct–Dec. Uses MLflow to track every run. Saves the best model.

**How it decides which model wins:**
1. Split stations into 80% train / 20% test
2. For each of the 3 models × 3 target months (Oct, Nov, Dec) = 9 model-target combinations:
   - Run `RandomizedSearchCV` to find the best hyperparameters (20 random combos, 5-fold cross-validation)
   - Retrain on full training set with the best hyperparameters
   - Score on the held-out test set → get MAE, RMSE, R²
   - Log everything to MLflow
3. The model with the best average R² across all three target months wins
4. The winning model pipeline is saved to `models/best_model.pkl`

**The sklearn Pipeline saved in `best_model.pkl` contains:**
```
Step 1: SimpleImputer  → fills missing values with median
Step 2: StandardScaler → (LinearReg only) normalises feature scale
Step 3: [Best model]   → RandomForest or GradientBoosting or LinearRegression
```
The whole pipeline is saved as one object so prediction-time preprocessing matches training-time exactly.

---

#### `assets/ml/evaluate_model.py`
```
Bruin asset name:  ml.evaluate_model
Type:              Python
Reads:             models/best_model.pkl
                   mart.traffic_features  (DuckDB)
Writes:            reports/eval_actual_vs_predicted_adt_oct.png  (and nov, dec)
                   reports/eval_residuals_adt_oct.png            (and nov, dec)
                   reports/feature_importance.png
                   reports/quality_gates.json
                   reports/model_report.html
```
**What it does:** The exam for the trained model. Loads the saved model and scores it on the Oct–Dec holdout data. Checks whether the model meets FEDRO-standard quality thresholds.

**Quality gates checked:**

| Gate | Threshold | What fails it |
|------|-----------|--------------|
| Overall R² | ≥ 0.80 | Model explains less than 80% of variance |
| Overall MAPE | ≤ 15% | Average prediction error exceeds 15% of actual traffic |
| Per-canton MAPE | ≤ 25% | Any canton's average error exceeds 25% |
| Max station error | ≤ 50% | Any single station is off by more than 50% |

**Charts produced per target month:**
- **Actual vs Predicted scatter** — perfect model = points on the y=x diagonal. Outlier stations are labelled automatically.
- **Residual plot** — shows `actual − predicted` vs predicted. A good model has residuals randomly scattered around zero (no systematic bias).

**Output: `quality_gates.json`** — machine-readable pass/fail record. Used by CI/CD pipelines to block deployment if quality fails.

---

#### `assets/ml/predict_traffic.py`
```
Bruin asset name:  ml.predict_traffic
Type:              Python
Reads:             models/best_model.pkl
                   mart.traffic_features  (DuckDB)
Writes:            reports/predictions_romandy.csv
                   reports/predictions_chart.png
                   reports/predictions_lausanne.png
                   DuckDB: mart.predictions
```
**What it does:** The graduation ceremony — uses the trained model to make actual predictions for Romandy stations.

**Two prediction horizons:**
1. **Q4 2025 (Oct–Nov–Dec):** Model predicts using Jan–Sep 2025 features (we can validate since we have the real data).
2. **Full year 2026 estimate:** Applies a +0.8% annual growth factor to all 2025 values. Based on FEDRO's historical average Swiss national road growth rate.

**Charts produced:**
- `predictions_chart.png` — horizontal bar chart comparing 2025 actual annual ADT vs model's Q4 prediction for the top 20 Romandy stations
- `predictions_lausanne.png` — line chart for Lausanne VD stations: solid line = Jan–Sep actual, dashed = Oct–Dec predicted, dotted = 2026 estimate

---

### `analyses/` folder

This folder contains **ad-hoc queries** — they are NOT part of the main pipeline. They're diagnostic tools you run manually.

---

#### `analyses/data_quality_check.sql`
```
Type:     Ad-hoc SQL (run manually, not part of pipeline)
Run with: duckdb traffic.duckdb < analyses/data_quality_check.sql
          OR: bruin query --asset analyses/data_quality_check.sql
```
**What it does:** A set of diagnostic `SELECT` queries to investigate data quality before committing to ML training. Run this after Stage 2 (staging) if you want to understand missingness.

**Checks inside it:**

| Check | Question answered |
|-------|-----------------|
| Check 1: Overall dimensions | How many stations, cantons, rows exist? What % have valid ADT? |
| Check 2: NULL rate per canton | Which cantons have the most missing data? |
| Check 3: NULL rate per month | Which months are systematically missing (seasonal closures)? |
| Check 4: Stations with < 6 months data | Which stations are too incomplete to use in ML? |
| Check 5: Romandy-specific check | Data quality specifically for our region of interest |

---

### `models/` folder

#### `models/best_model.pkl`
```
Type:     Binary file (Python pickle format)
Created by: assets/ml/train_model.py
Read by:    assets/ml/evaluate_model.py, assets/ml/predict_traffic.py
```
**What it is:** The complete trained ML pipeline saved to disk. Contains the preprocessing steps (imputer, scaler) AND the trained model parameters (decision tree weights). Load it with:
```python
import joblib
pipeline = joblib.load("models/best_model.pkl")
predictions = pipeline.predict(X_new)
```

---

### `reports/` folder

All files here are **generated outputs** — never edit them manually. Re-run the pipeline to regenerate them.

| File | Generated by | Contents |
|------|-------------|----------|
| `historical_report.html` | `generate_reports.py` | Browser-openable historical analysis dashboard |
| `model_report.html` | `evaluate_model.py` | ML evaluation results with charts |
| `quality_gates.json` | `evaluate_model.py` | Pass/fail status for each quality gate |
| `predictions_romandy.csv` | `predict_traffic.py` | All predictions in tabular CSV format |
| `training_metrics.json` | `train_model.py` | MAE/RMSE/R² for all 9 model-target combinations |
| `feature_importance.csv` | `train_model.py` | Which features the best model relied on most |
| `*.png` | Various | Chart images embedded in the HTML reports |

---

### `mlruns/` folder
```
Created by: MLflow (via assets/ml/train_model.py)
View with:  mlflow ui --backend-store-uri ./mlruns
            then open: http://localhost:5000
```
**What it is:** MLflow's experiment tracking store. Every time `train_model.py` runs, it creates a new "run" folder inside here with:
- `params/` — hyperparameters used (n_estimators, learning_rate, ...)
- `metrics/` — evaluation scores (mae, rmse, r2)
- `artifacts/` — any files saved (model file, feature importance CSV)
- `meta.yaml` — run metadata (start time, run ID, status)

These runs persist across pipeline executions, so you can compare how the model changed between runs.

---

### Configuration Files

#### `.bruin.yml` (if present)
```
Type:     Bruin workspace configuration
Used by:  bruin CLI
```
Declares the database connection under `connections → duckdb`. Points Bruin to the `traffic.duckdb` file so it knows where to run SQL assets.

#### `.dockerignore`
```
Type:     Docker build exclusion list
Used by:  docker compose build
```
Tells Docker which files/folders NOT to copy into the image during build. Key exclusions: `.venv/` (would overwrite the container's packages), `traffic.duckdb` (data, not code), `reports/` and `models/` (generated outputs), `.git/` (version control history shouldn't be in a Docker image).

---

```

```
DATABASE: traffic.duckdb  (single embedded file, no server required)
═══════════════════════════════════════════════════════════════════════════════

SCHEMA: raw  ──  "The Warehouse Loading Dock"
  Think of raw as a loading dock: goods arrive messy from the truck,
  but we haven't touched them yet. Exact copy of CSV data.
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ raw.annual_results                                                       │
  │   station_id      INTEGER    Sensor number (e.g., 064 = Lausanne)       │
  │   station_name    VARCHAR    Official FEDRO name                         │
  │   canton          VARCHAR    Two-letter canton code (VD, GE, ...)        │
  │   road            VARCHAR    Road ID (A 9, H 1, ...)                    │
  │   metric_type     VARCHAR    ADT, AWT, ADT Tu-Th, ADT Sa, ADT Su        │
  │   adt_jan–adt_dec DOUBLE     Monthly average daily traffic               │
  │   annual_avg      DOUBLE     FEDRO official annual average               │
  │   ~1,000 rows (200 stations × 5 metric rows each)                       │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ raw.adtwithclasses  (same structure + vehicle class breakdown)           │
  │   metric_type can be: ADT, ADT HV (heavy vehicles), ADT HGV,            │
  │                        AWT, AWT HV, AWT HGV                              │
  │   ~1,200 rows                                                            │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ raw.station_notes                                                        │
  │   Data gaps: "No data" (sensor fault) or "Winter closing" (road shut)   │
  │   ~50 rows                                                               │
  └─────────────────────────────────────────────────────────────────────────┘

SCHEMA: staging  ──  "Quality Control Room"
  Think of staging as quality control: every item is inspected, tagged,
  and organized before going to the store floor.
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ staging.stg_stations (ONE row per station — master dimension table)     │
  │   station_id, station_name, canton, road                                │
  │   is_romandy BOOLEAN      True if French-speaking canton                │
  │   road_category VARCHAR   'Motorway' or 'National Road'                 │
  │   months_with_data INT    How many of 12 months have ADT readings       │
  │   ~200 rows                                                              │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ staging.stg_monthly_traffic (LONG format — one row per station×month)   │
  │   station_id, canton, road, month_num (1-12), month_name                │
  │   adt_value DOUBLE        The actual traffic count for that month       │
  │   metric_type VARCHAR     Which metric (ADT, AWT, etc.)                 │
  │   ~12,000 rows (200 stations × 5 metrics × 12 months)                  │
  └─────────────────────────────────────────────────────────────────────────┘

SCHEMA: mart  ──  "The Store Floor" (analytics and ML ready)
  Think of mart as the organized store shelves: everything is in the right
  place, labeled clearly, and ready for customers (analysts + ML models).
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ mart.traffic_features (WIDE format — ML feature matrix)                 │
  │   One row per station. All ML input + target columns.                   │
  │   Features (X): adt_jan through adt_sep + canton_code + road_type       │
  │   Targets  (y): adt_oct, adt_nov, adt_dec                               │
  │   ~200 rows × 30 columns                                                │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ mart.lausanne_analysis (Lausanne + VD canton deep-dive)                  │
  │   Monthly patterns, peak/valley detection, weekday vs weekend split      │
  │   ~20 rows (all VD canton stations)                                      │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ mart.romandy_summary (French-Swiss aggregate by canton)                  │
  │   Canton-level KPIs: total traffic, HV%, seasonal variation, rank       │
  │   ~6 rows (VD, GE, FR, NE, JU, VS)                                      │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages Explained

### Stage 1: Ingestion (`assets/raw/ingest_traffic_csv.py`)
**What it does:** Reads the 4 CSV files, parses the complex multi-row header format, and loads clean records into DuckDB.

**The challenge:** The FEDRO CSV files are formatted for human readability (like Excel), not for machine parsing. A "station block" looks like:
```
Row: 002 | CHALET-A-GOBET | | VD | H 1 | ADT  | 15106 | 16140 | ...
Row:     |                | |    |     | AWT  | 16870 | 17812 | ...  ← station info carried forward
Row:     |                | |    |     | ADT Sa | 12507 | ...         ← still same station
Row: 003 | BRISSAGO S     | | TI | H 13| ADT  | 6394  | 7069 | ...   ← NEW station starts
```

**Solution:** Forward-fill technique — remember the last seen station ID/name/canton/road and apply it to every row until a new station is encountered.

### Stage 2: Staging (`assets/staging/`)
**What it does:** Creates clean, validated, normalized tables from the raw data.

- `stg_stations`: Deduplicates to one record per station, adds computed flags (`is_romandy`, `road_category`, `months_with_data`)
- `stg_monthly_traffic`: Pivots from wide-format (12 month columns) to long-format (1 row per station×month), making it easy to filter by month or create time series

### Stage 3: Transform (`assets/transform/`)
**What it does:** Business logic and feature engineering.

- `traffic_features`: Creates the ML feature matrix with engineered features like `hv_percentage`, `seasonal_variance`, `weekday_weekend_ratio`
- `lausanne_analysis`: Deep-dives into Lausanne/VD stations — monthly patterns, peak summer tourism traffic, business day variations
- `romandy_summary`: Canton-level KPIs for all 6 French-speaking cantons

### Stage 4: Reporting (`assets/reporting/generate_reports.py`)
**What it generates:**
- `reports/lausanne_monthly_traffic.png` — Bar chart: Monthly ADT for Lausanne stations
- `reports/romandy_seasonal_patterns.png` — Line chart: Seasonal patterns by canton
- `reports/station_heatmap.png` — Heatmap: All Romandy stations × 12 months
- `reports/canton_comparison.png` — Box plots: Traffic distribution by canton
- `reports/historical_report.html` — Full interactive HTML analysis report

### Stage 5–7: Machine Learning (`assets/ml/`)

See [Machine Learning Approach](#machine-learning-approach) below.

---

## Machine Learning Approach

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE OVERVIEW                                      │
│                                                                                    │
│  mart.traffic_features                                                             │
│  (DuckDB table, ~200 rows)                                                        │
│           │                                                                        │
│           ▼                                                                        │
│  ┌──────────────────────────────────────┐                                         │
│  │  FEATURE ENGINEERING (in Python)     │                                         │
│  │  Features (X):                        │  Targets (y):                          │
│  │  • adt_jan through adt_sep (9 values)│  • adt_oct (October traffic)            │
│  │  • canton_encoded (int 0-6)          │  • adt_nov (November traffic)           │
│  │  • road_type_code (0=A, 1=H)         │  • adt_dec (December traffic)           │
│  │  • hv_percentage (trucks %)          │                                         │
│  │  • log_annual_adt_jan_sep            │                                         │
│  │  • seasonal_variance                 │                                         │
│  └──────────────────────────────────────┘                                         │
│           │                                                                        │
│           ▼                                                                        │
│  ┌──────────────────────────────────────┐                                         │
│  │  TRAIN / TEST SPLIT                   │                                         │
│  │  By station (not by time!)            │                                         │
│  │  Train: 80% of stations (randomly)    │                                         │
│  │  Test:  20% of stations (held out)    │                                         │
│  │  Predict months 10, 11, 12 for test  │                                         │
│  └──────────────────────────────────────┘                                         │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │  MODEL TRAINING (3 models trained for each target month)                │      │
│  │                                                                          │      │
│  │  Model 1: Linear Regression    ← Baseline. "If Jan is busy, Oct is too" │      │
│  │  Model 2: Random Forest        ← 100 decision trees vote together       │      │
│  │  Model 3: Gradient Boosting    ← Trees that self-correct their mistakes │      │
│  │                                                                          │      │
│  │  TRACKED WITH: MLflow                                                   │      │
│  │  Run: mlflow ui  →  opens browser at http://localhost:5000              │      │
│  └─────────────────────────────────────────────────────────────────────────┘      │
│           │                                                                        │
│           ▼                                                                        │
│  ┌──────────────────────────────────────┐                                         │
│  │  EVALUATION METRICS                   │                                         │
│  │  MAE  = Mean Absolute Error           │  "How many cars/day are we off?"        │
│  │  MAPE = Mean Abs % Error              │  "What % of actual traffic did we miss?"│
│  │  RMSE = Root Mean Square Error        │  "Penalizes big misses more"            │
│  │  R²   = Coefficient of Determination │  "What % of variation do we explain?"  │
│  └──────────────────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Model Selection Justification

| Model | Pros | Cons | When to Use |
|-------|------|------|-------------|
| **Linear Regression** | Interpretable, fast, never overfits | Can't capture non-linear patterns | Always use as baseline |
| **Random Forest** | Handles non-linearity, robust to outliers | Less interpretable, slower | When Linear Reg is insufficient |
| **Gradient Boosting** | Best accuracy, learns from errors | Needs careful hyperparameter tuning | When RF still doesn't satisfy |

---

## How to Run

There are two ways to run this project: locally with Python and Bruin, or inside Docker (no local setup required beyond Docker Desktop).

---

### Option A — Local (Python + Bruin)

#### Prerequisites
```bash
# Bruin CLI (already installed — confirmed v0.11.502)
bruin --version

# Python 3.10+ (already installed — confirmed 3.12.4)
python --version
```

#### First Run (Full Pipeline)
```bash
# Navigate to the project directory
cd "c:\Users\user\swiss_trafficc_data"

# Initialize git repository (required by Bruin)
git init

# Validate the pipeline configuration (catches errors before running)
bruin validate .

# Run the ENTIRE pipeline (all assets in correct order)
bruin run .

# Or run just one specific asset:
bruin run assets/raw/ingest_traffic_csv.py
bruin run assets/staging/stg_stations.sql
bruin run assets/ml/train_model.py
```

#### View MLflow Experiments (after running ML assets)
```bash
mlflow ui
# Open browser at: http://localhost:5000
# See all runs, metrics, model comparisons
```

#### Check Pipeline Lineage
```bash
# See the full DAG (who depends on whom)
bruin lineage assets/ml/predict_traffic.py
```

#### Query the Database Directly
```bash
# Run any SQL query against the DuckDB file
bruin query --connection duckdb_traffic --query "SELECT * FROM mart.romandy_summary"

# Or open DuckDB directly in Python:
# python -c "import duckdb; conn = duckdb.connect('traffic.duckdb'); print(conn.execute('SHOW TABLES').fetchdf())"
```

---

### Option B — Docker (recommended)

Docker packages the entire runtime (Python 3.11, Bruin CLI, all packages) into a self-contained Linux container. You don't need to install anything locally except [Docker Desktop](https://www.docker.com/products/docker-desktop/).

#### Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  docker-compose.yml defines 2 services, sharing one image:      │
│                                                                   │
│  ┌─────────────────────────────┐  ┌──────────────────────────┐  │
│  │  pipeline                   │  │  mlflow                  │  │
│  │  - Runs run_pipeline.py     │  │  - Runs mlflow ui        │  │
│  │  - Writes DuckDB, reports,  │  │  - Reads from mlruns/    │  │
│  │    models, mlruns           │  │  - Port 5000:5000         │  │
│  │  - Exits when done          │  │  - Stays running         │  │
│  └─────────────────────────────┘  └──────────────────────────┘  │
│                      │                        │                   │
│                      └────────────────────────┘                  │
│                           shared named volumes                    │
│          (duckdb_data, reports_data, models_data, mlruns_data)    │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 1 — Build the image
```powershell
# First time only (or after changing Dockerfile / pyproject.toml)
docker compose build
```

What this does:
1. Pulls `python:3.11-slim` base image from Docker Hub
2. Installs `curl`, `git`, `ca-certificates`
3. Downloads and installs **Bruin CLI** into `/usr/local/bin`
4. Downloads and installs **uv** (fast package manager) into `/usr/local/bin`
5. Copies `pyproject.toml` and installs all Python dependencies
6. Copies the project code into `/app`
7. Tags the image as `swiss-traffic:latest`

> **Build time:** ~2–3 minutes on first build. Subsequent builds use cached layers and finish in seconds (unless you change `pyproject.toml` or the `Dockerfile`).

#### Step 2 — Run the full pipeline
```powershell
# Run all 7 stages in order, then exit
docker compose up pipeline

# Stream logs in real time while it runs
docker compose logs -f pipeline
```

Expected output (all 7 stages):
```
[Stage 1] Ingestion: 2,015 rows, 403 stations ingested
[Stage 2] Staging:   403 stations, 24,180 monthly rows
[Stage 3] Transform: 330 features, 372 Lausanne rows, 6 Romandy canton rows
[Stage 4] Reporting: 4 charts + historical_report.html saved
[Stage 5] Training:  GradientBoosting wins — R²=0.9993/0.9999/1.0000
[Stage 6] Evaluate:  Quality gates checked (FONTANA station flagged)
[Stage 7] Predict:   82 Romandy predictions, predictions_romandy.csv written
```

#### Step 3 — View the MLflow UI
```powershell
# Start the MLflow web dashboard (runs until you Ctrl+C)
docker compose up mlflow

# Then open your browser at:
# http://localhost:5000
```

The MLflow UI shows every training run: parameters, metrics (MAE, MAPE, RMSE, R²), and the winning model per run. Runs are stored in the shared `mlruns_data` volume, so the `mlflow` service reads whatever the `pipeline` service wrote.

#### Run both services at once
```powershell
# Start pipeline + keep MLflow UI up after it finishes
docker compose up
```

#### Useful Docker commands
```powershell
# Force a full rebuild (ignores cache) — use after changing Dockerfile
docker compose build --no-cache

# Open an interactive shell inside the container (for debugging)
docker compose run --rm pipeline bash

# Check what volumes exist
docker volume ls

# Remove all volumes and start fresh (destroys cached DuckDB/outputs)
docker compose down -v
```

#### Volume mapping explained

| Volume | What's stored | Host path |
|--------|--------------|----------|
| `duckdb_data` | `traffic.duckdb` — all database tables | Named Docker volume |
| `reports_data` | `reports/` — all `.html` and `.png` files | Named Docker volume |
| `models_data` | `models/best_model.pkl` | Named Docker volume |
| `mlruns_data` | `mlruns/` — all MLflow experiment runs | Named Docker volume |
| `.` (project root) | CSVs + source code | Bind-mounted from host |

To copy reports to your host machine after the pipeline runs:
```powershell
docker compose cp pipeline:/app/reports ./reports
docker compose cp pipeline:/app/models ./models
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `traffic.duckdb` | project root | All database tables (raw + staging + mart) |
| `mlruns/` | project root | MLflow experiment tracking (all runs, metrics, models) |
| `historical_report.html` | `reports/` | Full historical data analysis |
| `model_report.html` | `reports/` | ML model evaluation report |
| `lausanne_monthly_traffic.png` | `reports/` | Lausanne seasonal bar chart |
| `romandy_seasonal_patterns.png` | `reports/` | All Romandy cantons line chart |
| `station_heatmap.png` | `reports/` | Traffic intensity heatmap |
| `canton_comparison.png` | `reports/` | Box plot by canton |
| `predictions_scatter.png` | `reports/` | Predicted vs Actual scatter |
| `feature_importance.png` | `reports/` | RF/GBM feature importance |
| `best_model.pkl` | `models/` | Serialized best-performing model |

---

## Key Swiss Traffic Terminology (FEDRO Abbreviations)

| French | German | English | Code |
|--------|--------|---------|------|
| Trafic journalier moyen | Durchschnittlicher Tagesverkehr | Average Daily Traffic | ADT / TJM / DTV |
| TJM jours ouvrables | Durchschnittlicher Werktagesverkehr | Average Weekday Traffic | AWT / TJMO / DWV |
| TJM mardi-jeudi | DTV Di-Do | Average Tue-Thu Traffic | ADT Tu-Th |
| TJM samedi | Durchschnittlicher Samstagsverkehr | Average Saturday Traffic | ADT Sa |
| TJM dimanche | Durchschnittlicher Sonntagsverkehr | Average Sunday Traffic | ADT Su |
| Véhicules lourds | Schwerverkehr | Heavy Vehicles (buses + trucks) | HV / VL / SV |
| Véhicules lourds marchandises | Schwere Güterfahrzeuge | Heavy Goods Vehicles | HGV / VLM / SGF |

---

*Data source: Federal Roads Office FEDRO / ASTRA — Swiss Automatic Road Traffic Counts (SARTC) Annual Bulletin 2025*
*© FEDRO — verkehrsdaten@astra.admin.ch*
