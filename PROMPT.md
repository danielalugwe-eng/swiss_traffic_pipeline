# Swiss Traffic MLOps — LLM Replication & Extension Prompt

## PROJECT CONTEXT

You are a senior data/ML engineer building a complete MLOps pipeline for Swiss road traffic data published by FEDRO (Federal Roads Office / ASTRA — Astra Strassenverkehrszählung).

The dataset is the **Annual Results 2025** bulletin: ~200 automatic measuring stations across Switzerland, measuring **Average Daily Traffic (ADT)** and vehicle class breakdowns every month of 2025. Files ship as Latin-1-encoded CSVs with Excel-style multi-row headers and merged station-block rows that must be forward-filled.

The strategic focus is **French-speaking Switzerland (Romandy)** — specifically the cantons **VD, GE, VS, NE, FR, JU** — with Lausanne (Station 064, CONT. DE LAUSANNE, A9 motorway, ~97,662 vehicles/day) as the flagship "hero city" for all deep-dive analysis and charts.

---

## DATASET DESCRIPTION

| File | Description |
|------|-------------|
| `Annual_results_2025.csv` | Main file: one row per station-month-metric. Metrics: ADT (average daily traffic), DTV (daily total volume), DTVW (daily total weekday), GVM (heavy goods vehicles), PW (passenger cars). 12 monthly columns (`01` through `12`). |
| `Annual_results_ Measuring_station_2025.csv` | Station metadata: GPS coordinates, altitude, lane info, opening year |
| `Annual_results_ Measuring_station_adtwithclasses2025.csv` | ADT broken down by vehicle class (motorcycles, cars, light goods, HGV, buses, etc.) |
| `Annual_results_ Measuring_station_legend2025.csv` | Legend/notes: gaps, closures, estimation flags per station |

**Encoding**: `latin-1`  
**Header rows to skip**: 5 (rows 0–4 are title/blank/metadata; data starts at row 5 with column headers, actual data at row 6)  
**Station block format**: Station ID, name, canton, road appear only on the first row of their block; subsequent rows for other metrics have blank values → must be forward-filled with `pandas.DataFrame.ffill()`

---

## TECH STACK

| Tool | Version | Role |
|------|---------|------|
| Python | ≥3.10 | All executable assets |
| **Bruin** | ≥0.9.0 | Pipeline orchestration and asset DAG |
| DuckDB | ≥0.10.0 | Embedded analytical database (`traffic.duckdb`) |
| pandas | ≥2.1.0 | CSV parsing, DataFrame operations |
| numpy | ≥1.26.0 | Numerical operations, array math |
| scikit-learn | ≥1.4.0 | ML models, pipelines, metrics |
| MLflow | ≥2.11.0 | Experiment tracking and model registry |
| joblib | ≥1.3.0 | Model serialisation to `.pkl` |
| matplotlib | ≥3.8.0 | All charts (use `matplotlib.use("Agg")` for headless) |
| seaborn | ≥0.13.0 | Statistical charts (heatmaps, boxplots) |
| Jinja2 | ≥3.1.0 | HTML report templating |
| uv | ≥0.4.0 | Fast Python package manager |

---

## PIPELINE ARCHITECTURE

```
                                  ┌──────────────────────┐
  4 FEDRO CSVs  ──────────────▶  │  Stage 1: INGEST     │
  (project root)                  │  (Python asset)       │
                                  │  raw.annual_results   │
                                  │  raw.adtwithclasses   │
                                  │  raw.station_notes    │
                                  └────────────┬─────────┘
                                               │
                              ┌────────────────┴──────────────────┐
                              │            Stage 2: STAGING        │
                              │  stg_stations    stg_monthly_traffic│
                              │  (1 row/station) (long format fact) │
                              └────────────────┬──────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
          ┌─────────▼────────┐     ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │ traffic_features │     │  lausanne_analysis   │   │  romandy_summary    │
          │ (ML feature mat.)│     │  (VD deep-dive)      │   │  (Romandy cantons)  │
          └─────────┬────────┘     └──────────┬──────────┘   └──────────┬──────────┘
                    │                          │                          │
          ┌─────────▼────────┐                └────────────┬─────────────┘
          │  Stage 5: ML     │                             │
          │  train_model.py  │               ┌─────────────▼──────────────┐
          │  evaluate.py     │               │  Stage 4: REPORTING        │
          │  predict.py      │               │  generate_reports.py       │
          └──────────────────┘               │  historical_report.html    │
                                             └────────────────────────────┘
```

**Bruin DAG** (from `pipeline.yml`):
```
ingest_traffic_csv
  → stg_stations
  → stg_monthly_traffic
    → traffic_features
    → lausanne_analysis
    → romandy_summary
      → generate_reports
      → train_model
        → evaluate_model
          → predict_traffic
```

---

## BRUIN ASSET FORMAT

Every asset starts with a YAML header block:

```python
"""
@bruin
name: raw.ingest_traffic_csv
type: python
depends:
  - []
tags:
  - raw
  - ingestion
@end
...rest of docstring...
"""
```

For SQL assets:
```sql
/* @bruin
name: staging.stg_stations
type: duckdb.sql
materialization:
  type: table
  schema: staging
depends:
  - raw.ingest_traffic_csv
tags:
  - staging
@end */
```

---

## ML STRATEGY

**Why cross-sectional, not time-series?**  
We have only 1 year (2025) of data. ARIMA, LSTM, and Prophet all require multiple years of historical observations to learn seasonal patterns. With ~200 stations × 1 year, we instead treat each station-month as an independent observation and train **across stations**.

**Train/test split**:
- Training: All stations, months Jan–Sep (columns `adt_jan` through `adt_sep`) → 9 feature months × ~180 usable stations = ~1,620 training samples
- Test (holdout): Month Oct, Nov, Dec actual values — withheld from training

**Models trained** (via scikit-learn `Pipeline`):
1. `LinearRegression` with `StandardScaler` — baseline
2. `RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3)` — ensemble
3. `GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8)` — boosting

**Feature matrix** (17 features per station):
```
RAW MONTHS:     adt_jan, adt_feb, adt_mar, adt_apr, adt_may,
                adt_jun, adt_jul, adt_aug, adt_sep
ENGINEERED:     summer_peak_ratio      = adt_jul / mean(jan–sep)
                weekday_weekend_ratio  = dtvw_jul / adt_jul
                hgv_pct_jul            = gvm_jul / adt_jul
                winter_depression_ratio = mean(jan,feb,mar) / mean(jun,jul,aug)
                mean_adt_jan_sep       = AVG(adt_jan..adt_sep)
                canton_code            = integer-encoded canton (1–18)
                road_type_code         = 0 (motorway), 1 (national road)
                is_romandy_int         = 1 if VD/GE/VS/NE/FR/JU else 0
```

**Best model selection**: The model with the highest mean R² across all 3 target months is saved as `models/best_model.pkl`.

**MLflow**: All experiments logged to `./mlruns` under experiment `"swiss-traffic-adt-prediction"`. View with `mlflow ui --backend-store-uri ./mlruns`.

---

## OUTPUTS PRODUCED

| File | Stage | Description |
|------|-------|-------------|
| `traffic.duckdb` | Ingest | DuckDB database (all schemas) |
| `reports/historical_report.html` | Reporting | HTML dashboard: KPIs, 4 charts, canton tables |
| `reports/model_report.html` | ML Evaluate | HTML evaluation: R², MAPE, residuals, canton errors |
| `reports/predictions_romandy.csv` | ML Predict | Per-station predictions for Oct/Nov/Dec 2025 |
| `reports/lausanne_monthly_traffic.png` | Reporting | Grouped bar: Lausanne monthly ADT vs other metrics |
| `reports/romandy_seasonal_patterns.png` | Reporting | Multi-line: seasonal index per Romandy canton |
| `reports/station_heatmap.png` | Reporting | Seaborn heatmap: month × station normalised ADT |
| `reports/canton_comparison.png` | Reporting | Boxplot: ADT distribution by Romandy canton |
| `reports/predictions_chart.png` | ML Predict | Horizontal bar: actual 2025 vs model Q4 prediction |
| `reports/predictions_lausanne.png` | ML Predict | Line: actual Jan–Sep + predicted Oct–Dec for VD stations |
| `reports/predictions_2026_projections.png` | ML Predict | Bar: 2025 actual vs 2026 +0.8% growth estimate |
| `models/best_model.pkl` | ML Train | joblib bundle: `{target: Pipeline, feature_cols, metrics}` |
| `mlruns/` | ML Train | MLflow experiment tracker |

---

## FILE STRUCTURE

```
swiss_trafficc_data/
├── .bruin.yml                    # DuckDB connection config
├── pipeline.yml                  # Bruin DAG definition
├── pyproject.toml                # Python deps (uv)
├── .gitignore
├── README.md                     # Full project documentation
├── PROMPT.md                     # This file
├── traffic.duckdb                # Generated: DuckDB database
│
├── Annual_results_2025.csv
├── Annual_results_ Measuring_station_2025.csv
├── Annual_results_ Measuring_station_adtwithclasses2025.csv
├── Annual_results_ Measuring_station_legend2025.csv
│
├── analyses/
│   └── data_quality_check.sql   # Ad-hoc DQ queries (9 checks)
│
├── assets/
│   ├── raw/
│   │   └── ingest_traffic_csv.py   # Stage 1: CSV → DuckDB raw tables
│   ├── staging/
│   │   ├── stg_stations.sql         # Stage 2: dimension table
│   │   └── stg_monthly_traffic.sql  # Stage 2: long-format fact table
│   ├── transform/
│   │   ├── traffic_features.sql     # Stage 3: ML feature matrix
│   │   ├── lausanne_analysis.sql    # Stage 3: VD deep-dive
│   │   └── romandy_summary.sql      # Stage 3: canton aggregates
│   ├── reporting/
│   │   └── generate_reports.py      # Stage 4: charts + HTML report
│   └── ml/
│       ├── train_model.py           # Stage 5: train 3 models
│       ├── evaluate_model.py        # Stage 6: evaluate + charts
│       └── predict_traffic.py       # Stage 7: predict + 2026 est.
│
├── models/
│   └── best_model.pkl               # Generated by train_model.py
└── reports/
    └── *.html, *.png, *.csv         # Generated outputs
```

---

## HOW TO RUN

### Prerequisites
```bash
# Install uv (fast Python package manager)
pip install uv

# Install Bruin
brew install bruin  # macOS
# or: pip install bruin-io

# Install Python dependencies
uv sync
```

### Full pipeline run
```bash
# Run entire pipeline end-to-end
bruin run

# Or run individual stages
bruin run --asset raw.ingest_traffic_csv          # Step 1
bruin run --asset staging.stg_stations            # Step 2a
bruin run --asset staging.stg_monthly_traffic     # Step 2b
bruin run --asset mart.traffic_features           # Step 3a
bruin run --asset mart.lausanne_analysis          # Step 3b
bruin run --asset mart.romandy_summary            # Step 3c
bruin run --asset reporting.generate_reports      # Step 4
bruin run --asset ml.train_model                  # Step 5
bruin run --asset ml.evaluate_model               # Step 6
bruin run --asset ml.predict_traffic              # Step 7

# Run data quality checks
bruin query --asset analyses/data_quality_check.sql

# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

### Query predictions directly
```python
import duckdb
con = duckdb.connect("traffic.duckdb")

# See all Romandy predictions
con.execute("""
    SELECT station_name, canton, road,
           pred_oct, pred_nov, pred_dec,
           annual_adt_2025, adt_2026_est
    FROM mart.predictions
    WHERE canton IN ('VD','GE','VS','NE','FR','JU')
    ORDER BY annual_adt_2025 DESC
""").df()
```

---

## EXTENSION IDEAS (for LLMs extending this project)

1. **Add 2024 data**: If you obtain the FEDRO 2024 bulletin, you can add a `year` column and build a proper time-series model. Import the file as `raw.annual_results_2024` and JOIN on `station_id`.

2. **Weather features**: The MeteoSwiss open data API provides monthly temperature and precipitation per station. Cold temperatures strongly suppress traffic in alpine corridors. Add a `weather_features.sql` asset that joins Met data to `traffic_features`.

3. **Streamlit dashboard**: Wrap `mart.predictions` in a Streamlit app for interactive exploration. Chart library: Plotly Express (replaces matplotlib for interactivity).

4. **Station-type growth rates**: Instead of flat 0.8% growth for 2026, use regression on OpenStreetMap road category + canton GDP growth to estimate per-station growth rates.

5. **Multi-year bulletin**: FEDRO releases annual bulletins going back to the 1990s. With 10+ years, train a proper SARIMA or LightGBM + time features model.

6. **Anomaly detection**: Use IsolationForest on the monthly feature vectors to flag stations with unusual 2025 traffic patterns (construction zones, new road openings).

---

## KEY DESIGN DECISIONS AND RATIONALE

| Decision | Rationale |
|----------|-----------|
| DuckDB over PostgreSQL | Zero-infrastructure setup; analytical queries on CSVs run in seconds; easily committed to git as a single file |
| Bruin for orchestration | DAG-native framework built for data pipelines; supports Python + SQL assets natively; YAML headers are minimal boilerplate |
| Cross-sectional ML | Only 1 year of data; time-series models require multi-year seasonality; ~200 stations as independent observations gives enough training data |
| scikit-learn Pipeline | Encapsulates imputer + scaler + model; prevents data leakage; single `.predict()` call at inference time |
| joblib over pickle | joblib handles numpy arrays more efficiently; standard in scikit-learn ecosystem |
| matplotlib Agg backend | Headless rendering in pipeline scripts; avoids Tkinter/GUI errors in CI/CD or SSH sessions |
| Jinja2 HTML templates | Portable reports that open in any browser; no external JS dependencies; easy to embed base64 chart images |
| 7/9 month threshold | Stations with >2 missing training months provide unreliable features; this filter keeps ~90% of stations while removing the worst-quality sensors |
| 0.8% growth rate | Conservative midpoint of FEDRO's historical 0.5–1.2% annual growth; documented for transparency; easily changed via `GROWTH_RATE_2026` constant |

---

## COMMON PITFALLS

1. **CSV encoding**: Always open FEDRO files with `encoding="latin-1"`. UTF-8 will fail on accented French/German characters (e.g., "Bâle", "Zürich").

2. **Forward-fill scope**: In `parse_main_csv()`, forward-fill only the station-identifying columns (`station_id`, `station_name`, `canton`, `road`, `direction`). Never forward-fill numeric values — you would silently propagate the wrong metric's numbers.

3. **DuckDB schema creation**: Always `CREATE SCHEMA IF NOT EXISTS` before `CREATE TABLE`. DuckDB raises an error if the schema does not exist.

4. **MLflow run nesting**: Each call to `mlflow.start_run()` must be properly closed. Use `with mlflow.start_run() as run:` context manager to guarantee cleanup.

5. **matplotlib state leakage**: Always call `plt.close(fig)` after saving a figure. In a script that creates many charts, unclosed figures accumulate in memory and may corrupt later plots.

6. **DuckDB read/write connections**: Pass `read_only=True` to `duckdb.connect()` for query-only operations. Use `read_only=False` (default) only when writing tables. Mixing both modes in the same connection is fine, but explicit intent improves clarity.

---

*Generated by Swiss Traffic MLOps v1.0 — FEDRO/ASTRA Annual Results 2025*  
*Pipeline framework: Bruin · Database: DuckDB · ML: scikit-learn + MLflow*
