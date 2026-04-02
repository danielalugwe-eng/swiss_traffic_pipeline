# Looker Studio — Swiss Traffic Dashboard Guide

**Looker Studio:** https://lookerstudio.google.com (free, Google account required)

This guide covers:
1. [How the data gets into Looker Studio](#1-how-the-data-gets-into-looker-studio)
2. [The six CSV data sources — every column explained](#2-the-six-csv-data-sources)
3. [Data cleaning applied before export](#3-data-cleaning-applied-before-export)
4. [Dashboard pages and charts to build](#4-dashboard-pages-and-charts-to-build)
5. [Step-by-step: setting up in Looker Studio](#5-step-by-step-setup)
6. [Recommended calculated fields](#6-recommended-calculated-fields)
7. [Filter controls to add to every page](#7-filter-controls)

---

## 1. How the Data Gets into Looker Studio

Looker Studio has a **native BigQuery connector** — no CSV files, no manual uploads. The pipeline writes directly to BigQuery tables after every run, and Looker Studio queries BigQuery in real time.

```
DuckDB (local pipeline) ──► BigQuery dataset ──► Looker Studio dashboards
       traffic.duckdb           swiss_traffic            (auto-refresh)
```

**Six BigQuery tables are written on every pipeline run:**

```
BigQuery dataset: swiss_traffic   (configurable via BQ_DATASET env var)
├── monthly_traffic      ← MAIN fact table — all stations × months × metrics (~24K rows)
├── stations             ← Station master list (dimensions/metadata, ~403 rows)
├── romandy_summary      ← Canton KPIs (6 rows)
├── traffic_features     ← ML engineered features per station (~330 rows)
├── predictions          ← Model Q4 + 2026 forecasts (~82 rows)
└── quality_gates        ← Model evaluation pass/fail results (~15 rows)
```

**Required setup (one time):**
1. Create a GCP project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the BigQuery API
3. Create a dataset named `swiss_traffic` in BigQuery
4. Set environment variables: `BQ_PROJECT_ID=your-gcp-project` and optionally `BQ_DATASET=swiss_traffic`
5. Authenticate: run `gcloud auth application-default login` (local) or attach a service account (Docker/CI)

**Running the pipeline:**
```bash
python run_pipeline.py
```
`BQ_PROJECT_ID` defaults to `swiss-traffic-mlops`. Override via environment variable only if you use a different GCP project. Tables are overwritten on each run. No re-upload step needed.

---

## 2. The Six BigQuery Tables

---

### `monthly_traffic` — The Main Fact Table

**One row = one station × one metric × one month**
~24,000 rows. This is the backbone of most charts.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `station_id` | Number | `64` | FEDRO sensor number. Join key to `ls_stations.csv` |
| `station_name` | Text | `CONT. DE LAUSANNE` | Official FEDRO station name (raw, may include suffixes like `(AR)`) |
| `canton_code` | Text | `VD` | Two-letter Swiss canton abbreviation |
| `canton_display` | Text | `Vaud (VD)` | Human-readable canton name — **use this in charts** |
| `road` | Text | `A 9` | Road identifier (`A` = motorway, `H` = national road) |
| `road_category` | Text | `Motorway` | `Motorway` or `National Road` |
| `is_romandy` | Boolean | `true` | `true` if French-speaking canton (VD, GE, VS, NE, FR, JU) |
| `traffic_date` | Date | `2025-07-01` | First day of the measurement month — Looker auto-detects as a date |
| `month_num` | Number | `7` | Month number 1–12. Use for sorting time-series charts correctly |
| `month_name` | Text | `July` | Full English month name |
| `quarter` | Text | `Q3` | Quarter: Q1 (Jan–Mar), Q2 (Apr–Jun), Q3 (Jul–Sep), Q4 (Oct–Dec) |
| `season` | Text | `Summer` | Meteorological season: Winter, Spring, Summer, Autumn |
| `metric_type` | Text | `ADT` | FEDRO metric code (see table below) |
| `metric_display` | Text | `Average Daily Traffic (all days)` | Human-readable metric name — **use this in charts** |
| `adt_value` | Number | `97662` | The actual traffic count (average vehicles per day for this month) |
| `data_status` | Text | `Present` | `Present` if `adt_value` is not null, `Missing` if the sensor had no data |
| `months_with_data` | Number | `12` | How many of the 12 months this station has valid ADT readings (quality indicator) |

**Metric type reference:**

| `metric_type` | `metric_display` | What it counts |
|--------------|-----------------|---------------|
| `ADT` | Average Daily Traffic (all days) | All vehicles, all days of the week, averaged |
| `AWT` | Average Weekday Traffic (Mon–Fri) | All vehicles, weekdays only |
| `ADT Tu-Th` | Average Traffic (Tue–Thu) | All vehicles, Tuesday–Thursday only (mid-week baseline) |
| `ADT Sa` | Average Saturday Traffic | All vehicles, Saturdays only |
| `ADT Su` | Average Sunday Traffic | All vehicles, Sundays only |
| `ADT HV` | Heavy Vehicles (all days) | Buses + all truck classes, all days |
| `ADT HGV` | Heavy Goods Vehicles (trucks) | Freight trucks only (excludes buses) |
| `AWT HV` | Heavy Vehicles (weekdays) | Heavy vehicles on weekdays |
| `AWT HGV` | Heavy Goods Vehicles (weekdays) | Freight trucks on weekdays |

> **Tip:** Filter `metric_type = 'ADT'` for most charts — it's the primary, standardised metric used in all Swiss transport planning.

---

### `stations` — Station Dimension Table

**One row = one measuring station**
~403 rows. Use this as a lookup / dimension table.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `station_id` | Number | `64` | Unique FEDRO sensor number — join key |
| `station_name` | Text | `CONT. DE LAUSANNE` | Raw name from FEDRO CSV |
| `station_name_clean` | Text | `CONT. DE LAUSANNE` | Name with suffix codes like `(AR)`, `(N)` removed |
| `canton_code` | Text | `VD` | Two-letter canton code |
| `canton_display` | Text | `Vaud (VD)` | Full canton name for display |
| `road` | Text | `A 9` | Road number |
| `road_category` | Text | `Motorway` | `Motorway` or `National Road` |
| `is_romandy` | Boolean | `true` | French-speaking Switzerland filter |
| `months_with_data` | Number | `12` | Count of months with valid ADT data (0 = total sensor failure, 12 = complete year) |
| `annual_adt_2025` | Number | `97662` | FEDRO official annual average ADT for 2025 |

---

### `romandy_summary` — Canton KPI Table

**One row = one Romandy canton (6 rows total)**
Aggregated statistics for French-speaking Switzerland.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `canton` | Text | `VD` | Canton code (join key) |
| `canton_display` | Text | `Vaud (VD)` | Full canton name |
| `avg_annual_adt` | Number | `58420` | Average of all station annual ADTs in this canton |
| `total_stations` | Number | `18` | Number of measuring stations in this canton |
| `peak_month` | Text | `July` | Month with the highest average traffic across all canton stations |
| `trough_month` | Text | `January` | Month with lowest average traffic |
| `seasonal_variation_pct` | Number | `23.4` | `(peak − trough) / avg × 100` — how seasonal the canton is |
| `busiest_station` | Text | `CONT. DE LAUSANNE` | Highest-traffic station name |
| `busiest_station_adt` | Number | `97662` | ADT of the busiest station |
| `avg_hgv_pct` | Number | `8.2` | Average heavy goods vehicle percentage across canton stations |

**How to interpret `seasonal_variation_pct`:**
- **VD: ~20%** — mix of commuter (Lausanne) and ski roads (Alps)
- **VS: ~45%** — highly seasonal (Valais alpine passes, Zermatt, Verbier)
- **GE: ~10%** — mostly commuter/international traffic, very stable year-round

---

### `traffic_features` — ML Feature Matrix

**One row = one station**
~330 rows. Contains the engineered features the ML model was trained on.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `station_id` | Number | `64` | Join key |
| `station_name` | Text | `CONT. DE LAUSANNE` | Station name |
| `canton_code` / `canton_display` | Text | `VD` / `Vaud (VD)` | Canton |
| `road` / `road_category` | Text | `A 9` / `Motorway` | Road info |
| `is_romandy` | Boolean | `true` | Romandy filter |
| `months_with_data` | Number | `12` | Data completeness |
| `adt_jan` … `adt_sep` | Number | `82000` | Monthly ADT Jan–Sep (the ML training features) |
| `adt_oct`, `adt_nov`, `adt_dec` | Number | `88000` | Q4 actuals (ground truth targets) |
| `summer_peak_ratio` | Number | `1.42` | `adt_jul / adt_jan` — >2 = tourist road, ~1 = commuter road |
| `weekday_weekend_ratio` | Number | `1.18` | `awt_jul / adt_jul` — >1.3 = business route, ~1 = leisure/tourism |
| `hgv_pct_jul` | Number | `6.8` | % of traffic that is heavy goods vehicles in July. >15% = freight corridor |
| `winter_depression_ratio` | Number | `0.91` | `adt_jan / mean_adt_jan_sep` — <0.7 = significant winter drop (alpine pass) |
| `mean_adt_jan_sep` | Number | `84500` | Average ADT Jan–Sep — overall volume indicator |

**How to use in Looker:**
- **Scatter chart:** x=`summer_peak_ratio`, y=`mean_adt_jan_sep`, colour by `canton_display` → shows tourist vs commuter roads
- **Bar chart:** `hgv_pct_jul` sorted descending → reveals freight corridors
- **Colour-coded table:** `winter_depression_ratio` with conditional formatting

---

### `predictions` — Model Forecasts

**One row = one Romandy station with predictions**
~82 rows. Contains the ML model's Q4 2025 predictions and 2026 estimates.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `station_id` | Number | `64` | Join key |
| `station_name` | Text | `CONT. DE LAUSANNE` | Station name |
| `canton_code` / `canton_display` | Text | `VD` / `Vaud (VD)` | Canton |
| `road` / `road_category` | Text | `A 9` / `Motorway` | Road info |
| `annual_adt_2025` | Number | `97662` | FEDRO official 2025 annual average (known, not predicted) |
| `pred_oct_2025` | Number | `92100` | Model prediction for October 2025 ADT |
| `pred_nov_2025` | Number | `88300` | Model prediction for November 2025 ADT |
| `pred_dec_2025` | Number | `85800` | Model prediction for December 2025 ADT |
| `pred_q4_avg` | Number | `88733` | Average of the three Q4 month predictions |
| `est_annual_2026` | Number | `98444` | Full-year 2026 estimate (`annual_adt_2025 × 1.008` — using FEDRO's +0.8% growth rate) |
| `abs_error_vs_annual` | Number | `4929` | `|pred_q4_avg − annual_adt_2025|` — how far Q4 prediction is from annual (in vehicles/day) |
| `pct_error_vs_annual` | Number | `5.0` | Percentage version of the above |

> **Note on `pct_error_vs_annual`:** This compares Q4 vs annual average (not Q4 vs actual Q4). It is an indicative accuracy measure, not the model's formal MAPE (which is in `ls_quality_gates.csv`).

---

### `quality_gates` — Model Evaluation Results

**One row = one quality check**
Contains the pass/fail result for each model evaluation standard.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `gate_name` | Text | `r2_adt_oct` | Name of the quality check |
| `status` | Text | `PASS` | `PASS` or `FAIL` |
| `threshold` | Number | `0.8` | The minimum/maximum required value |
| `actual_value` | Number | `0.9993` | The model's actual score on this check |
| `target_month` | Text | `adt_oct` | Which prediction target this check applies to (`overall` for cross-month checks) |
| `notes` | Text | `R² above threshold` | Human-readable explanation |

**Quality gate thresholds used:**

| Gate | Threshold | Meaning |
|------|-----------|---------|
| R² per target month | ≥ 0.80 | Model explains ≥80% of traffic variance |
| Overall MAPE | ≤ 15% | Average prediction within 15% of actual |
| Per-canton MAPE | ≤ 25% | No region catastrophically mis-modelled |
| Max station error | ≤ 50% | No completely rogue single-station prediction |

---

## 3. Data Cleaning Applied Before Export

Understanding the cleaning steps helps you interpret nulls and anomalies you may see in Looker.

---

### Step 1 — Skip metadata rows (in `ingest_traffic_csv.py`)

**Problem:** The FEDRO CSV has 5 rows of title text at the top before the real data begins.

**Fix:** `pandas.read_csv(skiprows=5)` skips those rows. The 6th row becomes the column header.

**What you see in the data:** No effect — metadata rows are discarded entirely.

---

### Step 2 — Forward-fill station identity (in `ingest_traffic_csv.py`)

**Problem:** The FEDRO CSV uses Excel "merged cells" logic. Station ID, name, canton, and road appear only on the first row of a 5-row block. The next 4 rows have empty cells in those columns.

**Raw CSV looks like:**
```
064 | CONT. DE LAUSANNE | VD | A 9 | ADT    | 82000 | 79000 | ...
    |                   |    |     | AWT    | 91000 | 89000 | ...   ← blank station cells
    |                   |    |     | ADT Sa | 70000 | 68000 | ...   ← blank
    |                   |    |     | ADT Su | 56000 | 55000 | ...   ← blank
    |                   |    |     | ADT Tu-Th | 95000 | ...        ← blank
065 | PREVERENGES ...   | VD | A 1 | ADT    | 88000 | ...           ← NEW station
```

**Fix:** `pandas.DataFrame.ffill()` (forward-fill) copies each non-empty value downward until the next non-empty value. Think of it as filling in the blanks with "same as above".

**What you see in the data:** Every row in `ls_monthly_traffic.csv` has a complete `station_id`, `station_name`, `canton`, and `road`. There are no nulls in those columns.

---

### Step 3 — Strip thousands-separator commas from numbers (in `ingest_traffic_csv.py`)

**Problem:** FEDRO formats numbers with comma separators: `"97,662"`. Python's `float()` raises a `ValueError` on this because it looks like a string, not a number.

**Fix:** `re.sub(r",", "", s)` removes all commas before parsing. For example: `"97,662"` → `"97662"` → `97662.0`.

**What you see in the data:** All `adt_value` fields in `ls_monthly_traffic.csv` are plain numbers without commas.

---

### Step 4 — Handle missing values (in `ingest_traffic_csv.py`)

**Problem:** Some cells are empty (sensor fault), contain `"-"` (FEDRO uses this for "no data"), or contain text like `"*"` (estimated value) that cannot be parsed as a number.

**Fix:**
- Empty string `""` → Python `None` → CSV null → blank cell in Looker
- Dash `"-"` → `None`
- Anything unparseable → `None`

**What you see in the data:** Null fields in `ls_monthly_traffic.csv`. The `data_status` column tells you `"Missing"` for these rows. In Looker, set chart filters to `data_status = 'Present'` to exclude them from averages.

**Root causes for missing data:**
- **Winter road closures:** Alpine passes (some VS/GR stations) close Dec–Mar; these months are null by design
- **Sensor faults:** Any month can be null if the sensor malfunctioned
- **Administrative estimates:** FEDRO marks some values with `*`; we currently treat these as clean data (the asterisk is stripped)

---

### Step 5 — Normalise month column names (in `ingest_traffic_csv.py`)

**Problem:** Different FEDRO bulletin editions use different month column headers. The 2025 bulletin uses English names (`January`, `February`, `Mai` [German for May]). Older bulletins use two-digit numbers (`01`, `02`).

**Fix:** A lookup table maps all variants to a single standard name (`adt_01` through `adt_12`). `"mai"` maps to `"05"` specifically for the German month name.

**What you see in the data:** All monthly columns are consistently named `adt_jan`–`adt_dec` in the export CSVs.

---

### Step 6 — Deduplicate to one row per station (in `stg_stations.sql`)

**Problem:** The raw table has 5–9 rows per station (one per metric type). Analytics tools need one canonical row per station for dimension tables and join operations.

**Fix:** `WHERE metric_type = 'ADT'` filters to just the Primary ADT metric row — the most complete and universally comparable metric.

**What you see in the data:** `ls_stations.csv` has exactly one row per station with no duplicates.

---

### Step 7 — Standardise road category (in `stg_stations.sql`)

**Problem:** Road identifiers like `A 9`, `A1`, `H 1`, `H13` are inconsistent (spaces vary). We need a clean categorical field.

**Fix:** `CASE WHEN road LIKE 'A%' THEN 'Motorway' WHEN road LIKE 'H%' THEN 'National Road' END`

**What you see in the data:** `road_category` is always either `"Motorway"` or `"National Road"` (or `"Other"` for rare cases). Use this for chart grouping.

---

### Step 8 — Remove station name suffix codes (in `stg_stations.sql`)

**Problem:** FEDRO appends codes like `(AR)` (Autoroutière = on motorway), `(N)` (Nationalstrasse), `(NW)` (Nordwest) to station names. These clutter chart labels.

**Fix:** `REGEXP_REPLACE(station_name, '\s*\(.*\)\s*$', '')` strips anything in parentheses at the end of the name.

**Example:** `"CONT. DE LAUSANNE (AR)"` → `"CONT. DE LAUSANNE"`

**What you see in the data:** `station_name_clean` in `ls_stations.csv` has clean names. The original `station_name` column is also kept if you need the suffix.

---

### Step 9 — Handle division-by-zero in engineered features (in `traffic_features.sql`)

**Problem:** Computed ratios like `summer_peak_ratio = adt_jul / adt_jan` would produce a division-by-zero error if `adt_jan` is null or zero (winter-closed station).

**Fix:** `NULLIF(adt_jan, 0)` returns `NULL` instead of 0, causing the division to produce `NULL` rather than crash. For example: `COALESCE(adt_jul, 0) / NULLIF(adt_jan, 0)`.

**What you see in the data:** Some engineered feature cells in `ls_traffic_features.csv` are null for stations with seasonal closures. This is correct — those ratios are meaningless for stations with no winter data.

---

### Step 10 — Impute missing values for ML (in `train_model.py`)

**Problem:** scikit-learn models crash if the feature matrix contains any `NaN` values.

**Fix:** A `SimpleImputer(strategy="median")` fills missing values with the median of that column across all stations. This is included in the saved `best_model.pkl` pipeline, so predictions work even on stations with some null months.

**What you see in the data:** The model's predictions in `ls_predictions.csv` exist even for stations that had some null months in training. The imputer handled those gaps internally.

---

## 4. Dashboard Pages and Charts to Build

We recommend a 4-page Looker Studio dashboard.

---

### Page 1: Romandy Overview

**Data source:** `ls_monthly_traffic.csv` (filtered: `is_romandy = true`, `metric_type = 'ADT'`)

| Chart | Type | Dimension | Metric | Notes |
|-------|------|-----------|--------|-------|
| Total Romandy stations | Scorecard | — | `COUNT(DISTINCT station_id)` | Filter: `is_romandy = true` |
| Average annual ADT | Scorecard | — | `AVG(adt_value)` | Filter: `is_romandy = true`, `month_num = all` |
| Traffic by canton (bar) | Bar chart | `canton_display` | `AVG(adt_value)` | Sort descending. Shows VD as highest |
| Seasonal patterns (line) | Line chart | `month_name` | `AVG(adt_value)` | Breakdown: `canton_display`. Set x-axis sort to `month_num` |
| Station heatmap | Table + conditional formatting | `station_name` rows, `month_name` cols | `adt_value` | Colour scale: light=low, dark=high |

**Filter controls to add:**
- Dropdown: `canton_display`
- Dropdown: `road_category`

---

### Page 2: Station Deep-Dive

**Data source:** `ls_monthly_traffic.csv` (primary) + `ls_stations.csv` (for metadata)

| Chart | Type | Dimension | Metric | Notes |
|-------|------|-----------|--------|-------|
| Monthly traffic (bar) | Grouped bar | `month_name` | `adt_value` | Breakdown: individual selected stations. Set x sort to `month_num` |
| Metric comparison | Line chart | `month_name` | `adt_value` | Breakdown: `metric_display`. Select one station. Shows ADT vs AWT vs ADT Sa etc. |
| Station metadata table | Table | All columns from `ls_stations.csv` | — | Searchable station list |
| Weekday vs weekend gap (bar) | Bar | `canton_display` | `weekday_weekend_ratio` | **From `ls_traffic_features.csv`** |

**Filter controls to add:**
- Search box: `station_name` (allows typing a station name)
- Dropdown: `canton_display`
- Dropdown: `season`

---

### Page 3: ML Predictions

**Data source:** `ls_predictions.csv` + `ls_quality_gates.csv`

| Chart | Type | Dimension | Metric | Notes |
|-------|------|-----------|--------|-------|
| Model overall status | Scorecard | — | Count of `status = 'PASS'` | **From `ls_quality_gates.csv`** — shows how many gates passed |
| Actual vs Predicted (scatter) | Scatter chart | x=`annual_adt_2025`, y=`pred_q4_avg` | — | Add a reference line `y = x` to show perfect prediction |
| Prediction error by canton | Bar | `canton_display` | `AVG(pct_error_vs_annual)` | Shows which cantons the model is least accurate for |
| 2026 forecast (bar) | Horizontal bar | `station_name` | `est_annual_2026` | Top 20 Romandy stations. Compare to `annual_adt_2025` |
| Quality gates table | Table | `gate_name`, `threshold`, `actual_value`, `status` | — | Add conditional formatting: `status = PASS` → green, `FAIL` → red |

---

### Page 4: Freight & Heavy Vehicles

**Data source:** `ls_monthly_traffic.csv` (filtered: `metric_type IN ('ADT HGV', 'ADT HV')`) + `ls_traffic_features.csv`

| Chart | Type | Dimension | Metric | Notes |
|-------|------|-----------|--------|-------|
| HGV % by station (bar) | Horizontal bar | `station_name` | `hgv_pct_jul` | **From `ls_traffic_features.csv`** — sorted descending reveals freight corridors |
| Seasonal HGV pattern | Line chart | `month_name` | `AVG(adt_value)` | Filter: `metric_type = 'ADT HGV'`. Compare summer vs winter freight |
| HGV vs total ADT scatter | Scatter | x=`mean_adt_jan_sep`, y=`hgv_pct_jul` | — | From `ls_traffic_features.csv` — shows which busy roads are also freight-heavy |
| Canton HGV table | Table | `canton_display` | `AVG(hgv_pct_jul)`, `AVG(mean_adt_jan_sep)` | Summary freight profile per canton |

---

## 5. Step-by-Step Setup

### Step 1 — GCP and BigQuery setup (one time only)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and open (or create) the project **`swiss-traffic-mlops`**
2. Note the **Project ID** shown in the top bar — it should be `swiss-traffic-mlops`
3. In the left menu: **BigQuery** → **Done** (this enables the API)
4. In BigQuery Studio, click **+ Create** → **Dataset**:
   - Dataset ID: `swiss_traffic`
   - Location: choose your nearest region (e.g. `EU` or `US`)
   - Click **Create dataset**

### Step 2 — Authenticate locally

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth application-default login
```

A browser window opens — log in with your Google account. This creates credentials that `pandas-gbq` picks up automatically.

### Step 3 — Run the pipeline

The project ID is pre-configured. Just run:

```bash
python run_pipeline.py
```

To use a different GCP project, override the environment variable:

**Windows (PowerShell):**
```powershell
$env:BQ_PROJECT_ID = "other-gcp-project-id"
python run_pipeline.py
```

**Linux / macOS / Docker:**
```bash
BQ_PROJECT_ID=other-gcp-project-id python run_pipeline.py
```

The pipeline prints each table as it uploads:
```
  [BQ] swiss_traffic.monthly_traffic    (24,183 rows × 18 columns)
  [BQ] swiss_traffic.stations           (403 rows × 10 columns)
  [BQ] swiss_traffic.romandy_summary    (6 rows × 10 columns)
  [BQ] swiss_traffic.traffic_features   (330 rows × 21 columns)
  [BQ] swiss_traffic.predictions        (82 rows × 14 columns)
  [BQ] swiss_traffic.quality_gates      (14 rows × 6 columns)
```

### Step 4 — Connect Looker Studio to BigQuery

1. Go to [lookerstudio.google.com](https://lookerstudio.google.com)
2. Click **Create** → **Report**
3. In the "Add data to report" panel, select **BigQuery**
4. Sign in with the same Google account used in Step 2
5. Navigate: **My Projects** → `swiss-traffic-mlops` → `swiss_traffic` dataset → `monthly_traffic` table
6. Click **Add** → **Add to report**

### Step 5 — Add the remaining tables as data sources

1. In your report, click **Resource** (top menu) → **Manage added data sources**
2. Click **Add a data source** → **BigQuery**
3. Navigate to `swiss_traffic` dataset and select each remaining table
4. Repeat for `stations`, `romandy_summary`, `traffic_features`, `predictions`, `quality_gates`

### Step 6 — Set correct data types

Verify these in each data source (click the data source → Edit):

| Table | Column | Should be |
|-------|--------|-----------|
| `monthly_traffic` | `traffic_date` | Date (YYYY-MM-DD) — BigQuery usually auto-detects this |
| `monthly_traffic` | `adt_value` | Number |
| `monthly_traffic` | `month_num` | Number (used for sorting month axis correctly) |
| `monthly_traffic` | `is_romandy` | Boolean |
| `stations` | `annual_adt_2025` | Number |
| `predictions` | `pred_q4_avg`, `est_annual_2026` | Number |

### Step 7 — Add blends (optional, for joined analyses)

To combine `monthly_traffic` with `stations` metadata in one chart:
1. In a chart, click **Data** → **Blend data**
2. Select `monthly_traffic` as the left table
3. Select `stations` as the right table
4. Join on: `station_id` = `station_id`
5. Pull in any extra columns from `stations` you need

---

## 6. Recommended Calculated Fields

Add these in Looker Studio (Data Source → Add a Field):

```
Field name:  Formatted ADT
Formula:     CONCAT(FORMAT_NUMBER(adt_value), " veh/day")
Use for:     Tooltip labels on charts

Field name:  Is High Volume
Formula:     IF(adt_value > 50000, "High (>50k)", IF(adt_value > 20000, "Medium (20–50k)", "Low (<20k)"))
Use for:     Colour dimension in scatter charts

Field name:  Romandy Label
Formula:     IF(is_romandy = TRUE, "Romandy", "Rest of Switzerland")
Use for:     Filter/grouping to compare Romandy vs national

Field name:  Error Category (in ls_quality_gates.csv)
Formula:     IF(status = "PASS", "✅ Pass", "❌ Fail")
Use for:     Scorecard and table colour coding
```

---

## 7. Filter Controls

Add these controls to every page so users can explore the data:

| Control type | Field | Page |
|-------------|-------|------|
| Drop-down | `canton_display` | All pages |
| Drop-down | `road_category` | Page 1, 2 |
| Drop-down | `season` | Page 1, 2 |
| Drop-down | `metric_display` | Page 2 (station deep-dive) |
| Drop-down | `data_status` | Set default to `Present` on all pages to hide missing data |
| Date range | `traffic_date` | Page 1, 2 (useful when multiple years are added) |

> **Important:** Always add a `data_status = 'Present'` filter to prevent nulls from skewing averages. A station with 3 missing months would otherwise drag down its average if Looker counts nulls as 0.

---

## Data Refresh

Each time FEDRO releases a new annual bulletin (typically February/March for the prior year):

1. Drop the new CSV files into the project root (replacing the 2025 files)
2. Run `python run_pipeline.py`
3. The export stage overwrites all BigQuery tables automatically (`if_exists='replace'`)
4. Looker Studio charts refresh automatically on next open — no manual steps needed

No changes to the Looker Studio dashboard structure are needed across annual bulletins as the column names stay the same.
