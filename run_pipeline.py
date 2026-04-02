"""
run_pipeline.py — Execute all pipeline stages sequentially.

=============================================================================
WHY DOES THIS FILE EXIST?
=============================================================================
Bruin normally orchestrates everything: it reads the `depends:` field in each
asset's header, builds a DAG, and runs assets in the correct order using its
own isolated Python environment.

However, when running locally (not through `bruin run .`), we need a single
entry point that:
  1. Knows the correct execution order
  2. Reuses the SAME Python interpreter that has all our packages installed
  3. Handles SQL assets by reading and executing the SQL directly (Bruin would
     normally do this itself via its internal SQL runner)

Think of this file as the CONDUCTOR of the orchestra: it doesn't play
any instruments itself — it just tells each musician when to play.

=============================================================================
EXECUTION ORDER (mirrors the DAG in pipeline.yml):
=============================================================================
  1. ingest_traffic_csv.py    ← Read CSVs, load into DuckDB raw tables
  2. stg_stations.sql         ← Clean station metadata (1 row per station)
  3. stg_monthly_traffic.sql  ← Normalize to long format (1 row per station×month)
  4. traffic_features.sql     ← Build ML feature matrix (wide format)
  5. lausanne_analysis.sql    ← Deep VD canton analysis
  6. romandy_summary.sql      ← Canton-level KPIs for all 6 Romandy cantons
  7. generate_reports.py      ← Draw charts + produce HTML report
  8. train_model.py           ← Train 3 ML models, pick the best
  9. evaluate_model.py        ← Score on holdout months, check quality gates
 10. predict_traffic.py       ← Generate 2025 Q4 + 2026 predictions
 11. export_looker_studio.py  ← Export CSVs for Looker Studio dashboards

=============================================================================
Usage:
    python run_pipeline.py
=============================================================================
"""

import subprocess  # Run external processes (the Python asset scripts)
import sys         # Access the current Python interpreter path
import os          # File path construction
import duckdb      # Direct DuckDB connection for SQL asset execution

# __file__ resolves to the absolute path of this script.
# dirname() gives us the folder it lives in = the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The single DuckDB file that stores ALL our data (raw, staging, mart schemas).
DB_PATH = os.path.join(PROJECT_ROOT, "traffic.duckdb")

# sys.executable = path to the Python binary that is currently running THIS script.
# We pass it to subprocess.run() so that asset scripts use the SAME interpreter
# (and therefore the SAME installed packages) rather than picking up a system Python.
PYTHON = sys.executable


def header(title):
    """Print a visual separator between pipeline stages for easy log reading."""
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_sql_asset(filepath, con):
    """
    Execute a Bruin SQL asset file against the DuckDB connection.

    HOW BRUIN SQL ASSETS WORK:
    Each .sql file has a /* @bruin ... @end */ comment block at the top
    that contains metadata: the asset name (e.g. "staging.stg_stations"),
    materialization type ("table" or "view"), and dependencies.

    Bruin would normally:
      1. Read this header to know the destination table name
      2. Strip the header (it's a comment, not valid SQL)
      3. Wrap the SELECT in  CREATE OR REPLACE TABLE name AS (...)
      4. Execute it

    Since we're bypassing Bruin's runner, this function does the same thing:

    STEP 1 — Parse the header block (/* ... */) to extract:
      - asset_name: e.g. "staging.stg_stations"  → becomes the table name
      - mat_type:   "table" or "view"             → determines CREATE keyword

    STEP 2 — Create the schema (DuckDB won't auto-create schemas).
      e.g. asset_name = "staging.stg_stations"
      → schema = "staging"
      → execute: CREATE SCHEMA IF NOT EXISTS staging

    STEP 3 — Wrap the SQL body and run it:
      CREATE OR REPLACE TABLE staging.stg_stations AS (
          <original SQL from the file>
      )

    PARAMETERS:
        filepath : absolute path to the .sql file
        con      : open duckdb.DuckDBPyConnection
    """
    import re as _re
    raw = open(filepath, encoding="utf-8").read()

    # Look for a Bruin header block: /* @bruin ... @end ... */
    asset_name = None
    mat_type = "table"
    if "/* " in raw and "@bruin" in raw:
        # Grab everything between /* and */ (the Bruin metadata block)
        header_block = raw[raw.index("/*"):raw.index("*/") + 2]

        # Extract  name: staging.stg_stations  using a regex
        # ^\s*name:\s*(\S+)  means: at the start of a line, optional whitespace,
        # the literal "name:", optional whitespace, then capture non-whitespace chars
        m = _re.search(r"^\s*name:\s*(\S+)", header_block, _re.MULTILINE)
        if m:
            asset_name = m.group(1)   # e.g. "staging.stg_stations"

        # Extract  type: duckdb.sql  (we only care if it says "view")
        m2 = _re.search(r"type:\s*(\S+)", header_block, _re.MULTILINE)
        if m2:
            mat_type = m2.group(1).lower()  # e.g. "duckdb.sql", "view"

        # Remove the header block so only the SQL body remains
        close_idx = raw.index("*/") + 2
        raw = raw[close_idx:].strip()

    # STEP 2: Create the schema before the table (DuckDB requires this explicitly)
    if asset_name and "." in asset_name:
        schema = asset_name.split(".")[0]   # "staging" or "mart" or "raw"
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # STEP 3: Wrap the SELECT body in CREATE OR REPLACE TABLE/VIEW and execute
    if asset_name:
        # "view" if mat_type literally says "view", otherwise "table"
        create_kw = "VIEW" if mat_type == "view" else "TABLE"
        # Remove trailing semicolons — they'd break the wrapping syntax
        body = raw.rstrip().rstrip(";")
        sql = f"CREATE OR REPLACE {create_kw} {asset_name} AS (\n{body}\n)"
        con.execute(sql)
    else:
        # No Bruin header found — run the SQL as-is (e.g. DDL scripts)
        try:
            con.sql(raw)
        except Exception:
            # Fallback: split on semicolons and run each statement individually
            # (DuckDB doesn't always support multi-statement strings)
            for stmt in [s.strip() for s in raw.split(";") if s.strip()]:
                con.execute(stmt)


def run_python_asset(filepath):
    """
    Execute a Bruin Python asset file as a subprocess.

    WHY SUBPROCESS AND NOT import?
    Python's import system caches modules. If we imported assets directly,
    the second asset to import 'duckdb' would get the CACHED version and
    might share connection state with the first asset.

    Using subprocess.run() gives each asset:
      - Its own fresh Python process
      - Its own memory space (no shared globals)
      - Clean stdout/stderr (we can see per-asset output in the logs)
      - An independent exit code we can check for failures

    WHY cwd=PROJECT_ROOT?
    Asset scripts use relative paths like "../.." to find the database.
    Setting cwd (current working directory) to the project root ensures
    these relative paths resolve correctly regardless of where you run
    `python run_pipeline.py` from.

    PARAMETERS:
        filepath : absolute path to the Python asset .py file
    """
    result = subprocess.run(
        [PYTHON, filepath],   # Same Python interpreter as the parent process
        cwd=PROJECT_ROOT,     # Working directory for the child process
        capture_output=False, # Stream output directly to our terminal (not buffered)
    )
    if result.returncode != 0:
        # Non-zero exit code = the asset script crashed.
        # We stop the entire pipeline here — running downstream assets on
        # broken upstream data would produce meaningless results.
        print(f"\n[ERROR] {filepath} exited with code {result.returncode}")
        sys.exit(result.returncode)


# =============================================================================
# STAGE 1: INGESTION
# =============================================================================
header("STAGE 1 / 7 — Raw Ingestion (CSV → DuckDB)")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "raw", "ingest_traffic_csv.py"))


# =============================================================================
# STAGE 2: STAGING
# =============================================================================
header("STAGE 2 / 7 — Staging SQL Transforms")
con = duckdb.connect(DB_PATH)

print("  Running stg_stations.sql ...")
run_sql_asset(os.path.join(PROJECT_ROOT, "assets", "staging", "stg_stations.sql"), con)
n = con.execute("SELECT COUNT(*) FROM staging.stg_stations").fetchone()[0]
print(f"  [OK] staging.stg_stations — {n} stations")

print("  Running stg_monthly_traffic.sql ...")
run_sql_asset(os.path.join(PROJECT_ROOT, "assets", "staging", "stg_monthly_traffic.sql"), con)
n = con.execute("SELECT COUNT(*) FROM staging.stg_monthly_traffic").fetchone()[0]
print(f"  [OK] staging.stg_monthly_traffic — {n} rows")

con.close()


# =============================================================================
# STAGE 3: TRANSFORM / MART
# =============================================================================
header("STAGE 3 / 7 — Transform / Mart SQL")
con = duckdb.connect(DB_PATH)

for name, path in [
    ("traffic_features",  "assets/transform/traffic_features.sql"),
    ("lausanne_analysis", "assets/transform/lausanne_analysis.sql"),
    ("romandy_summary",   "assets/transform/romandy_summary.sql"),
]:
    print(f"  Running {path} ...")
    run_sql_asset(os.path.join(PROJECT_ROOT, path), con)
    # Probe the table that was just created
    schema = "mart"
    tbl    = name
    try:
        n = con.execute(f"SELECT COUNT(*) FROM {schema}.{tbl}").fetchone()[0]
        print(f"  [OK] {schema}.{tbl} — {n} rows")
    except Exception:
        print(f"  [OK] {path}")

con.close()


# =============================================================================
# STAGE 4: REPORTING
# =============================================================================
header("STAGE 4 / 7 — Historical Reports & Charts")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "reporting", "generate_reports.py"))


# =============================================================================
# STAGE 5: ML TRAINING
# =============================================================================
header("STAGE 5 / 7 — ML Training")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "ml", "train_model.py"))


# =============================================================================
# STAGE 6: ML EVALUATION
# =============================================================================
header("STAGE 6 / 7 — ML Evaluation & Quality Gates")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "ml", "evaluate_model.py"))


# =============================================================================
# STAGE 7: PREDICTIONS
# =============================================================================
header("STAGE 7 / 7 — Traffic Predictions")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "ml", "predict_traffic.py"))


# =============================================================================
# STAGE 8: LOOKER STUDIO CSV EXPORT
# Export mart tables to reports/looker_studio/*.csv for upload to
# Google Sheets / Looker Studio dashboards. See LOOKER_STUDIO.md.
# =============================================================================
header("STAGE 8 / 8 — Looker Studio CSV Export")
run_python_asset(os.path.join(PROJECT_ROOT, "assets", "reporting", "export_looker_studio.py"))


# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print()
print("  OUTPUTS:")

outputs = [
    ("reports/historical_report.html",                  "Historical analysis dashboard"),
    ("reports/model_report.html",                       "ML evaluation report"),
    ("reports/quality_gates.json",                      "Quality gate pass/fail log"),
    ("reports/predictions_romandy.csv",                 "Romandy Q4 + 2026 predictions"),
    ("models/best_model.pkl",                           "Trained model bundle"),
    ("mlruns/",                                         "MLflow experiment tracker"),
    ("reports/looker_studio/ls_monthly_traffic.csv",    "Looker: main fact table"),
    ("reports/looker_studio/ls_stations.csv",           "Looker: station dimension"),
    ("reports/looker_studio/ls_romandy_summary.csv",    "Looker: canton KPIs"),
    ("reports/looker_studio/ls_traffic_features.csv",   "Looker: ML features"),
    ("reports/looker_studio/ls_predictions.csv",        "Looker: model forecasts"),
    ("reports/looker_studio/ls_quality_gates.csv",      "Looker: quality gate results"),
]
for path, desc in outputs:
    full = os.path.join(PROJECT_ROOT, path)
    exists = os.path.exists(full)
    mark = "✓" if exists else "✗"
    print(f"  [{mark}] {path:50s}  {desc}")

print()
print("  LOOKER STUDIO:")
print("  Upload reports/looker_studio/*.csv to Google Sheets, then")
print("  follow LOOKER_STUDIO.md for step-by-step dashboard setup.")

print()
print("  View MLflow UI:")
print("    mlflow ui --backend-store-uri ./mlruns")
print()
