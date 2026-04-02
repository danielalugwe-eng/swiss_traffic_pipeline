"""
@bruin
name: reporting.export_looker_studio
type: python

description: >
  Export all mart and prediction tables from DuckDB to BigQuery.
  Looker Studio connects directly to BigQuery — no CSV uploads needed.

  BIGQUERY TABLES WRITTEN (dataset = BQ_DATASET env var, default: swiss_traffic):
    monthly_traffic      ← All stations × months × metrics (main fact table)
    stations             ← Station master list (dimension table)
    romandy_summary      ← Canton KPIs (6 rows)
    traffic_features     ← ML feature matrix and engineered features
    predictions          ← Model Q4 2025 + 2026 forecasts
    quality_gates        ← Model evaluation pass/fail results

  REQUIRED ENVIRONMENT VARIABLES:
    BQ_PROJECT_ID        GCP project that owns the BigQuery dataset
    BQ_DATASET           BigQuery dataset name (default: swiss_traffic)

  AUTHENTICATION:
    Uses Google Application Default Credentials. Run one of:
      gcloud auth application-default login   (local development)
      Attach a service account with roles/bigquery.dataEditor  (CI / Docker)

depends:
  - mart.romandy_summary
  - mart.traffic_features
  - mart.predictions
  - ml.evaluate_model

tags:
  - reporting
  - looker-studio
  - bigquery
  - export
@end

=============================================================================
WHY BIGQUERY FOR LOOKER STUDIO
=============================================================================

Looker Studio has a native BigQuery connector — no CSV files, no manual
uploads. Once the pipeline writes to BigQuery, every Looker Studio chart
refreshes automatically by querying BigQuery directly.

DuckDB is an embedded file-based database with no network server, so
Looker Studio cannot connect to it directly. We read from DuckDB in the
pipeline, transform the data, and write the final tables to BigQuery.

WORKFLOW:
  1. Set environment variables: BQ_PROJECT_ID, BQ_DATASET
  2. Run python run_pipeline.py
  3. This script writes all tables to BigQuery (if_exists='replace')
  4. In Looker Studio, "Add data" → "BigQuery" → select project/dataset/table
  5. No re-upload needed on future runs — charts auto-refresh on next open

=============================================================================
CSV DESIGN PRINCIPLES FOR LOOKER STUDIO
=============================================================================

Column naming:
  • Use lowercase with underscores (canton_code, not "Canton Code")
  • No spaces in column names — Looker handles them but they create awkward field names
  • Date/time columns as YYYY-MM-DD strings (Looker auto-detects them)
  • Use human-readable display values where Looker will show them in charts
    (e.g. "Vaud (VD)" not just "VD")

Row structure:
  • One fact table in LONG format (one row per station × month × metric)
    — this powers all time-series and filter-by-month charts
  • Separate DIMENSION tables (stations, cantons) joined via station_id / canton

=============================================================================
"""

import os
import sys
import json
import glob

import pandas as pd
import duckdb
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account

# =============================================================================
# CONFIGURATION
# =============================================================================

THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.join(THIS_DIR, "..", "..")
DB_PATH       = os.path.join(PROJECT_ROOT, "traffic.duckdb")
QUALITY_JSON  = os.path.join(PROJECT_ROOT, "reports", "quality_gates.json")

# BigQuery destination — override via environment variables if needed.
# BQ_PROJECT_ID: GCP project ID (default: "swiss-traffic-mlops")
# BQ_DATASET:    BigQuery dataset name (default: "swiss_traffic")
BQ_PROJECT_ID = os.environ.get("BQ_PROJECT_ID", "swiss-traffic-mlops")
BQ_DATASET    = os.environ.get("BQ_DATASET", "swiss_traffic")

# Service account authentication
# Looks for a service account JSON in the project root.
# Override with GOOGLE_APPLICATION_CREDENTIALS env var if needed.
def _load_credentials():
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        # Auto-discover any service account JSON in the project root
        matches = glob.glob(os.path.join(PROJECT_ROOT, "*.json"))
        key_path = matches[0] if matches else None
    if key_path and os.path.exists(key_path):
        return service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
    return None  # Falls back to Application Default Credentials

GCP_CREDENTIALS = _load_credentials()


def ensure_dataset_exists() -> None:
    """Create the BigQuery dataset if it does not already exist."""
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=GCP_CREDENTIALS)
    dataset_ref = bigquery.Dataset(f"{BQ_PROJECT_ID}.{BQ_DATASET}")
    dataset_ref.location = "EU"   # change to "US" if your GCP project is US-based
    try:
        client.get_dataset(dataset_ref)
        print(f"  [BQ] Dataset '{BQ_DATASET}' already exists.")
    except Exception:
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"  [BQ] Dataset '{BQ_DATASET}' created.")


def connect_db() -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the DuckDB database."""
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        print("  Run python run_pipeline.py first.")
        sys.exit(1)
    return duckdb.connect(database=DB_PATH, read_only=True)


def upload_to_bq(df: pd.DataFrame, table_name: str) -> None:
    """Upload a DataFrame to a BigQuery table, replacing it on each run."""
    destination = f"{BQ_DATASET}.{table_name}"
    pandas_gbq.to_gbq(
        df,
        destination_table=destination,
        project_id=BQ_PROJECT_ID,
        credentials=GCP_CREDENTIALS,
        if_exists="replace",   # Overwrite table on every pipeline run
        progress_bar=False,
    )
    print(f"  [BQ] {destination}  ({len(df):,} rows × {len(df.columns)} columns)")


# =============================================================================
# EXPORT 1: monthly_traffic
# THE MAIN FACT TABLE — powers most Looker charts
# =============================================================================

def export_monthly_traffic(con: duckdb.DuckDBPyConnection) -> None:
    """
    Export the long-format monthly traffic table to BigQuery.

    This is the CORE data source for Looker Studio. Every chart that shows
    traffic over time, filtered by canton/metric/road type, uses this table.

    One row = one station × one metric × one month

    LOOKER STUDIO USAGE:
      • Time series chart: dimension=month_num, metric=adt_value
      • Bar chart by canton: dimension=canton_display, metric=AVG(adt_value)
      • Filter control: canton_display, metric_type, season, road_category
    """
    df = con.execute("""
        SELECT
            -- ── IDENTIFIERS ──────────────────────────────────────────────────
            t.station_id,
            t.station_name,

            -- Add canton full name for display in Looker charts
            t.canton                                              AS canton_code,
            CASE t.canton
                WHEN 'VD' THEN 'Vaud (VD)'
                WHEN 'GE' THEN 'Geneva (GE)'
                WHEN 'VS' THEN 'Valais (VS)'
                WHEN 'NE' THEN 'Neuchâtel (NE)'
                WHEN 'FR' THEN 'Fribourg (FR)'
                WHEN 'JU' THEN 'Jura (JU)'
                WHEN 'BE' THEN 'Berne (BE)'
                WHEN 'ZH' THEN 'Zurich (ZH)'
                WHEN 'AG' THEN 'Aargau (AG)'
                WHEN 'LU' THEN 'Lucerne (LU)'
                WHEN 'TI' THEN 'Ticino (TI)'
                WHEN 'SG' THEN 'St. Gallen (SG)'
                WHEN 'GR' THEN 'Graubünden (GR)'
                WHEN 'SO' THEN 'Solothurn (SO)'
                WHEN 'BS' THEN 'Basel-Stadt (BS)'
                WHEN 'BL' THEN 'Basel-Landschaft (BL)'
                WHEN 'TG' THEN 'Thurgau (TG)'
                WHEN 'SH' THEN 'Schaffhausen (SH)'
                WHEN 'ZG' THEN 'Zug (ZG)'
                ELSE t.canton
            END                                                   AS canton_display,

            t.road,
            t.road_category,
            t.is_romandy,

            -- ── TIME DIMENSIONS ───────────────────────────────────────────────
            -- Looker Studio needs a proper date column for time-series charts.
            -- We construct a YYYY-MM-01 string. Looker auto-detects it as a date.
            '2025-' || LPAD(CAST(t.month_num AS VARCHAR), 2, '0') || '-01'
                                                                  AS traffic_date,
            t.month_num,
            t.month_name,
            t.quarter,
            t.season,

            -- ── METRIC ───────────────────────────────────────────────────────
            -- metric_type uses FEDRO abbreviations. We add a human-readable version.
            t.metric_type,
            CASE t.metric_type
                WHEN 'ADT'       THEN 'Average Daily Traffic (all days)'
                WHEN 'AWT'       THEN 'Average Weekday Traffic (Mon–Fri)'
                WHEN 'ADT Tu-Th' THEN 'Average Traffic (Tue–Thu)'
                WHEN 'ADT Sa'    THEN 'Average Saturday Traffic'
                WHEN 'ADT Su'    THEN 'Average Sunday Traffic'
                WHEN 'ADT HV'    THEN 'Heavy Vehicles (all days)'
                WHEN 'ADT HGV'   THEN 'Heavy Goods Vehicles (trucks)'
                WHEN 'AWT HV'    THEN 'Heavy Vehicles (weekdays)'
                WHEN 'AWT HGV'   THEN 'Heavy Goods Vehicles (weekdays)'
                ELSE t.metric_type
            END                                                   AS metric_display,

            -- ── VALUE ─────────────────────────────────────────────────────────
            ROUND(t.adt_value, 0)                                 AS adt_value,
            -- Null flag: useful in Looker to show "No data" labels
            CASE WHEN t.adt_value IS NULL THEN 'Missing' ELSE 'Present' END
                                                                  AS data_status,

            -- ── DATA QUALITY ─────────────────────────────────────────────────
            t.months_with_data
        FROM staging.stg_monthly_traffic t
        ORDER BY t.station_id, t.metric_type, t.month_num
    """).df()

    upload_to_bq(df, "monthly_traffic")


# =============================================================================
# EXPORT 2: ls_stations.csv
# DIMENSION TABLE — join key for all other tables
# =============================================================================

def export_stations(con: duckdb.DuckDBPyConnection) -> None:
    """
    Export the master station dimension table.

    One row = one measuring station.
    Use this in Looker as a lookup table joined to ls_monthly_traffic on station_id.

    LOOKER STUDIO USAGE:
      • Scorecard: total station count
      • Table: searchable list of all stations with metadata
      • Filter: is_romandy = TRUE to show only French-speaking stations
    """
    df = con.execute("""
        SELECT
            station_id,
            station_name,
            station_name_clean,
            canton                  AS canton_code,
            CASE canton
                WHEN 'VD' THEN 'Vaud (VD)'
                WHEN 'GE' THEN 'Geneva (GE)'
                WHEN 'VS' THEN 'Valais (VS)'
                WHEN 'NE' THEN 'Neuchâtel (NE)'
                WHEN 'FR' THEN 'Fribourg (FR)'
                WHEN 'JU' THEN 'Jura (JU)'
                WHEN 'BE' THEN 'Berne (BE)'
                WHEN 'ZH' THEN 'Zurich (ZH)'
                WHEN 'AG' THEN 'Aargau (AG)'
                WHEN 'LU' THEN 'Lucerne (LU)'
                WHEN 'TI' THEN 'Ticino (TI)'
                WHEN 'SG' THEN 'St. Gallen (SG)'
                WHEN 'GR' THEN 'Graubünden (GR)'
                ELSE canton
            END                     AS canton_display,
            road,
            road_category,
            is_romandy,
            months_with_data,
            annual_adt              AS annual_adt_2025
        FROM staging.stg_stations
        ORDER BY station_id
    """).df()

    upload_to_bq(df, "stations")


# =============================================================================
# EXPORT 3: ls_romandy_summary.csv
# CANTON KPI TABLE — powers the Romandy overview dashboard tab
# =============================================================================

def export_romandy_summary(con: duckdb.DuckDBPyConnection) -> None:
    """
    Export the canton-level Romandy KPI table.

    One row = one Romandy canton (6 rows total: VD, GE, VS, NE, FR, JU).

    LOOKER STUDIO USAGE:
      • Scorecards: avg_annual_adt per canton
      • Bar chart: avg_annual_adt ranked by canton
      • Table: all KPIs side by side
    """
    df = con.execute("""
        SELECT *,
            CASE canton
                WHEN 'VD' THEN 'Vaud (VD)'
                WHEN 'GE' THEN 'Geneva (GE)'
                WHEN 'VS' THEN 'Valais (VS)'
                WHEN 'NE' THEN 'Neuchâtel (NE)'
                WHEN 'FR' THEN 'Fribourg (FR)'
                WHEN 'JU' THEN 'Jura (JU)'
            END AS canton_display
        FROM mart.romandy_summary
        ORDER BY canton_avg_adt DESC NULLS LAST
    """).df()

    upload_to_bq(df, "romandy_summary")


# =============================================================================
# EXPORT 4: ls_traffic_features.csv
# ML FEATURE TABLE — powers model input analysis charts
# =============================================================================

def export_traffic_features(con: duckdb.DuckDBPyConnection) -> None:
    """
    Export the ML feature matrix.

    One row = one station with all engineered features and Q4 actuals.

    LOOKER STUDIO USAGE:
      • Scatter chart: summer_peak_ratio vs annual ADT (shows tourist vs commuter roads)
      • Bar chart: hgv_pct_jul by station (freight corridor identification)
      • Filter: is_romandy to focus on French-speaking Switzerland
    """
    df = con.execute("""
        SELECT
            station_id,
            station_name,
            canton,
            CASE canton
                WHEN 'VD' THEN 'Vaud (VD)'
                WHEN 'GE' THEN 'Geneva (GE)'
                WHEN 'VS' THEN 'Valais (VS)'
                WHEN 'NE' THEN 'Neuchâtel (NE)'
                WHEN 'FR' THEN 'Fribourg (FR)'
                WHEN 'JU' THEN 'Jura (JU)'
                WHEN 'BE' THEN 'Berne (BE)'
                WHEN 'ZH' THEN 'Zurich (ZH)'
                WHEN 'AG' THEN 'Aargau (AG)'
                WHEN 'LU' THEN 'Lucerne (LU)'
                WHEN 'TI' THEN 'Ticino (TI)'
                ELSE canton
            END  AS canton_display,
            road,
            road_category,
            is_romandy,
            months_with_data,
            -- Monthly ADT Jan-Sep (training features)
            adt_jan, adt_feb, adt_mar, adt_apr, adt_may,
            adt_jun, adt_jul, adt_aug, adt_sep,
            -- Q4 actuals (ground truth targets)
            adt_oct, adt_nov, adt_dec,
            -- Engineered features
            summer_peak_ratio,
            weekday_weekend_ratio,
            hgv_pct_jul,
            winter_depression_ratio,
            mean_adt_jan_sep
        FROM mart.traffic_features
        ORDER BY mean_adt_jan_sep DESC NULLS LAST
    """).df()

    upload_to_bq(df, "traffic_features")


# =============================================================================
# EXPORT 5: ls_predictions.csv
# MODEL PREDICTIONS TABLE — powers the forecast dashboard tab
# =============================================================================

def export_predictions(con: duckdb.DuckDBPyConnection) -> None:
    """
    Export the model predictions for all Romandy stations.

    One row = one station with actual Q4 2025 and predicted Q4 2025 + 2026 estimates.

    LOOKER STUDIO USAGE:
      • Scatter chart: actual vs predicted (model accuracy visual)
      • Bar chart: predicted 2026 ADT by canton
      • Table: predictions with error columns highlighted
    """
    try:
        df = con.execute("""
            SELECT
                station_id,
                station_name,
                canton,
                CASE canton
                    WHEN 'VD' THEN 'Vaud (VD)'
                    WHEN 'GE' THEN 'Geneva (GE)'
                    WHEN 'VS' THEN 'Valais (VS)'
                    WHEN 'NE' THEN 'Neuchâtel (NE)'
                    WHEN 'FR' THEN 'Fribourg (FR)'
                    WHEN 'JU' THEN 'Jura (JU)'
                    ELSE canton
                END  AS canton_display,
                road,
                road_category,
                annual_adt_2025,
                pred_oct            AS pred_oct_2025,
                pred_nov            AS pred_nov_2025,
                pred_dec            AS pred_dec_2025,
                pred_q4_avg,
                adt_2026_est        AS est_annual_2026,
                -- Absolute error vs annual average
                ROUND(ABS(pred_q4_avg - annual_adt_2025), 0) AS abs_error_vs_annual,
                ROUND(
                    ABS(pred_q4_avg - annual_adt_2025) / NULLIF(annual_adt_2025, 0) * 100,
                    1
                ) AS pct_error_vs_annual
            FROM mart.predictions
            ORDER BY annual_adt_2025 DESC NULLS LAST
        """).df()
        upload_to_bq(df, "predictions")
    except Exception as e:
        print(f"  [SKIP] predictions — mart.predictions not found ({e})")
        print("         Run Stage 7 (predict_traffic.py) first.")


# =============================================================================
# EXPORT 6: ls_quality_gates.csv
# MODEL QUALITY TABLE — powers the model health dashboard tab
# =============================================================================

def export_quality_gates() -> None:
    """
    Convert quality_gates.json to a flat CSV for Looker Studio.

    LOOKER STUDIO USAGE:
      • Scorecard: overall PASS / FAIL status
      • Table: all gate results with threshold vs actual values
      • Colour coding: red=FAIL, green=PASS
    """
    if not os.path.exists(QUALITY_JSON):
        print("  [SKIP] ls_quality_gates.csv — quality_gates.json not found.")
        print("         Run Stage 6 (evaluate_model.py) first.")
        return

    with open(QUALITY_JSON, encoding="utf-8") as f:
        gates = json.load(f)

    rows = []
    for gate_name, details in gates.items():
        if isinstance(details, dict):
            rows.append({
                "gate_name":    gate_name,
                "status":       details.get("status", "UNKNOWN"),
                "threshold":    details.get("threshold"),
                "actual_value": details.get("actual"),
                "target_month": details.get("target", "overall"),
                "notes":        details.get("notes", ""),
            })
        else:
            rows.append({
                "gate_name":    gate_name,
                "status":       str(details),
                "threshold":    None,
                "actual_value": None,
                "target_month": "overall",
                "notes":        "",
            })

    df = pd.DataFrame(rows)
    upload_to_bq(df, "quality_gates")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\nExporting tables to BigQuery...")
    print(f"  Project : {BQ_PROJECT_ID}")
    print(f"  Dataset : {BQ_DATASET}")
    print()

    ensure_dataset_exists()
    print()

    con = connect_db()

    export_monthly_traffic(con)
    export_stations(con)
    export_romandy_summary(con)
    export_traffic_features(con)
    export_predictions(con)

    con.close()

    export_quality_gates()

    print()
    print(f"Done. All tables are in BigQuery dataset '{BQ_DATASET}'.")
    print("In Looker Studio: Add data -> BigQuery -> select your project and dataset.")
    print("See LOOKER_STUDIO.md for step-by-step setup instructions.")


if __name__ == "__main__":
    main()
