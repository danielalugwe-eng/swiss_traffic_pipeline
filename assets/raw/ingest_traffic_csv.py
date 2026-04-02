"""
@bruin
name: raw.ingest_traffic_csv
type: python

# =============================================================================
# WHAT IS A BRUIN ASSET?
# A Bruin asset is a unit of work in the pipeline. Think of it like one
# workstation on an assembly line. This particular workstation's job is to
# take raw CSV files from disk and load them into our DuckDB database.
#
# WHY IS THIS A PYTHON ASSET (not SQL)?
# The FEDRO CSV files are not "machine-friendly". They were designed to be
# opened in Excel by a human, not parsed by a computer. They have:
#   - Multi-row headers (rows 1-5 are titles, not column names)
#   - Merged cell logic: station name appears only on the FIRST row of a block,
#     and the next 4 rows are empty — continuing data for the same station
#   - Numbers formatted with thousands separators: "97,662" (not 97662)
#   - Missing values for winter-closed roads
# SQL can't handle any of this. Python with pandas can.
#
# WHAT THIS ASSET PRODUCES (in DuckDB):
#   raw.annual_results      — monthly ADT per station, all metrics
#   raw.adtwithclasses      — same + vehicle class breakdown (trucks, buses...)
#   raw.station_notes       — data quality flags (gaps, closures, estimates)
#
# HOW TO RUN THIS ASSET ALONE:
#   bruin run assets/raw/ingest_traffic_csv.py
# =============================================================================

description: "Parse all 4 FEDRO/ASTRA CSV files and load them into DuckDB raw tables. Handles multi-row Excel-style headers, forward-fill station info, and numeric cleaning."
tags:
  - raw
  - ingestion
  - fedro
@end
"""

# =============================================================================
# LIBRARY IMPORTS — WHY WE NEED EACH ONE
# =============================================================================

import os          # os  = Operating System interface.
                   # We use it to build file paths (os.path.join) so the script
                   # works on both Windows (C:\...) and Linux/Mac (/home/...).
                   # Like a GPS that gives directions no matter what city you're in.

import re          # re = Regular Expressions.
                   # We use it to strip non-numeric characters from traffic counts
                   # like "97,662" → 97662. Think of it as a search-and-replace
                   # on steroids.

import pandas as pd    # pandas = The data table workhorse.
                       # DataFrames are like Excel spreadsheets in Python:
                       # rows, columns, filters, pivots — all programmable.
                       # We use it to read CSVs and restructure data.

import duckdb          # duckdb = Our embedded analytical database.
                       # No server needed — just a file. We use it to store the
                       # cleaned tables that the next pipeline stages will read.
                       # Like a supercharged Excel file that speaks SQL.

# =============================================================================
# CONFIGURATION — PATHS AND CONSTANTS
# =============================================================================

# __file__ is Python's built-in variable that holds the path to THIS script.
# os.path.dirname(__file__) = the folder containing THIS script.
# os.path.join(..., "..", "..", "...") = navigate UP 2 levels to project root.
# This means no matter WHERE you run the script from, it always finds the CSVs.
#
# FOLDER STRUCTURE (relative to this file):
#   assets/raw/ingest_traffic_csv.py   ← THIS FILE
#   assets/raw/../../                  ← project root
#   Annual_results_2025.csv            ← target CSV
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(THIS_DIR, "..", "..")
DATA_DIR = PROJECT_ROOT  # CSVs live in the project root alongside README.md

# The 4 FEDRO CSV files, relative to project root.
# We use a dictionary so each file has a clear name → role mapping.
CSV_FILES = {
    "annual": os.path.join(DATA_DIR, "Annual_results_2025.csv"),
    "adtwithclasses": os.path.join(
        DATA_DIR,
        "Annual_results_ Measuring_station_adtwithclasses2025.csv",
    ),
    "station_notes": os.path.join(
        DATA_DIR,
        "Annual_results_ Measuring_station_2025.csv",
    ),
}

# DuckDB database file. The path is relative to the project root so it
# matches what .bruin.yml declares under connections → duckdb → path.
DB_PATH = os.path.join(PROJECT_ROOT, "traffic.duckdb")

# The first 5 rows of every FEDRO CSV are title/metadata rows, not data.
# We skip them so pandas sees the real column headers starting at row 6.
#
# Row 1: "Schweizerische Eidgenossenschaft..."
# Row 2: "Class of vehicle: volume - motor vehicle"
# Row 3: "Road network: National and principal road network"
# Row 4: "Year: 2025 ..."
# Row 5: "Monthly and annual 24-hour traffic averages (ADT)"
# Row 6: REAL COLUMN HEADERS → Nr. | Measuring station | Ct | Road | 01 | 02...
HEADER_SKIP_ROWS = 5

# The 12-month column names as they appear after skipping the header rows.
# FEDRO labels months as two-digit numbers: "01" = January, "12" = December.
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# French-speaking cantons of Switzerland (Romandy).
# VD = Vaud (Lausanne), GE = Geneva, FR = Fribourg (bilingual),
# NE = Neuchâtel, JU = Jura, VS = Valais (bilingual).
# We use this set to tag stations as is_romandy = True/False.
ROMANDY_CANTONS = {"VD", "GE", "NE", "JU", "FR", "VS"}


# =============================================================================
# HELPER FUNCTION 1: clean_number
# =============================================================================

def clean_number(value) -> float | None:
    """
    Convert a FEDRO-formatted number string to a Python float.

    WHY THIS IS NEEDED:
    FEDRO formats traffic counts with comma thousands separators: "97,662"
    Python's float() function would raise a ValueError on that string because
    it expects "97662". We need to strip the comma first.

    Also handles:
    - NaN / empty cells (pandas reads missing values as float('nan'))
    - Whitespace around numbers ("  15,106  ")
    - Already-numeric values (int or float pass through unchanged)

    EXAMPLES:
        clean_number("97,662")  → 97662.0
        clean_number("6,394")   → 6394.0
        clean_number("")        → None
        clean_number(None)      → None
        clean_number(12345)     → 12345.0

    PARAMETERS:
        value: anything — string, int, float, or NaN from pandas

    RETURNS:
        float if parseable, None if missing/unparseable
    """
    # pandas represents empty CSV cells as float('nan').
    # We check this first so we don't accidentally call str() on NaN.
    if pd.isna(value):
        return None

    # Convert to string and strip surrounding whitespace.
    s = str(value).strip()

    # Empty string → no data (sensor offline, winter closure, etc.)
    if s == "" or s == "-":
        return None

    # Remove thousands separators (commas) so "97,662" becomes "97662".
    # re.sub(pattern, replacement, string)
    s = re.sub(r",", "", s)

    # Try to parse as a float. If it still fails (e.g., "N/A" text), return None.
    try:
        return float(s)
    except ValueError:
        return None


# =============================================================================
# HELPER FUNCTION 2: parse_main_csv
# =============================================================================

def parse_main_csv(filepath: str) -> pd.DataFrame:
    """
    Parse a FEDRO annual bulletin CSV (either Annual_results or adtwithclasses).

    THE PARSING CHALLENGE — "STATION BLOCKS":
    The CSV is not a normal table. Data is organized in "blocks":

      Row:  002 | CHALET-A-GOBET | VD | H 1 | ADT     | 15106 | 16140 | ...
      Row:      |                |    |     | AWT     | 16870 | 17812 | ...
      Row:      |                |    |     | ADT Tu-Th| ...
      Row:      |                |    |     | ADT Sa  | ...
      Row:      |                |    |     | ADT Su  | ...
      Row:  003 | BRISSAGO S     | TI | H 13| ADT     | 6394  | 7069  | ...
              ↑ New station ID means new block starts

    The station ID (Nr.), name, canton, and road appear ONLY on the first row
    of each block. The following rows for the same station have empty Nr., name,
    canton, road columns — they inherit from the previous station.

    This is called a "forward-fill" or "carry-forward" pattern.
    Our solution: remember the last seen ID/name/canton/road and apply to each row.

    PARAMETERS:
        filepath: absolute path to the CSV file

    RETURNS:
        pd.DataFrame with one row per (station, metric_type) combination:
         columns: station_id, station_name, canton, road, metric_type,
                  adt_01 through adt_12, annual_avg
    """
    # ── Step 1: Read raw CSV, skipping the 5-row metadata header ─────────────
    # header=0 means: treat the first non-skipped row as column names.
    # dtype=str means: read everything as a string — we'll parse numbers ourselves.
    # This prevents pandas from misinterpreting "97,662" as a string/number error.
    raw = pd.read_csv(
        filepath,
        skiprows=HEADER_SKIP_ROWS,
        header=0,
        dtype=str,
        keep_default_na=False,   # Don't auto-convert "" to NaN yet
        encoding="latin-1",      # FEDRO files use latin-1 (Windows-1252) encoding
    )

    # ── Step 2: Normalise column names ────────────────────────────────────────
    # pandas may read the first column as "Nr." or "Unnamed: 0".
    # The second column is "Measuring station", third is blank (canton label row),
    # fourth is "Ct" (canton code), fifth is "Road", sixth is the metric type.
    # Rename to predictable names for safe access.

    # The actual column positions in the FEDRO CSV after skipping header:
    # [0]=Nr.  [1]=Measuring station  [2]="" (blank)  [3]=Ct  [4]=Road
    # [5]=metric_type  [6-17]=months 01-12  [18]=annual_avg (Year: 2025)
    cols = list(raw.columns)

    # Build a mapping from whatever pandas called the column → our clean name
    rename_map = {}
    if len(cols) > 0:
        rename_map[cols[0]] = "nr"
    if len(cols) > 1:
        rename_map[cols[1]] = "station_name"
    if len(cols) > 3:
        rename_map[cols[3]] = "canton"
    if len(cols) > 4:
        rename_map[cols[4]] = "road"
    if len(cols) > 5:
        rename_map[cols[5]] = "metric_type"

    # Month columns: the CSV header can be either:
    #   • Two-digit strings: "01", "02", ... "12"  (some FEDRO editions)
    #   • Full English/German names: "January", "February", "Mai", "June", ...
    #     (the 2025 bulletin uses English month names, with "Mai" for May)
    # We build a lookup that maps EITHER format to "adt_01" ... "adt_12".
    MONTH_NAME_MAP = {
        # English names (2025 bulletin)
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05",  "june": "06",  "july": "07",  "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        # German/French variant (FEDRO uses "Mai" for May)
        "mai": "05",
        # Two-digit numeric format (older bulletins)
        **{m: m for m in MONTHS},
    }

    month_col_map = {}
    for col in cols:
        col_str = str(col).strip()
        month_num = MONTH_NAME_MAP.get(col_str.lower())
        if month_num:
            month_col_map[col] = f"adt_{month_num}"
        elif col_str.startswith("Year") or re.match(r"^20\d{2}$", col_str):
            rename_map[col] = "annual_avg"

    rename_map.update(month_col_map)
    raw = raw.rename(columns=rename_map)

    # Drop any remaining unnamed columns (there are often trailing empty columns)
    raw = raw[[c for c in raw.columns if not str(c).startswith("Unnamed")]]

    # ── Step 3: Forward-fill station ID, name, canton, road ──────────────────
    # Replace empty strings with NaN so forward-fill works.
    for col in ["nr", "station_name", "canton", "road"]:
        if col in raw.columns:
            raw[col] = raw[col].replace("", pd.NA)

    # ffill() = "forward fill": copy the last non-null value downward.
    # This fills in the blank station rows with the parent station's info.
    for col in ["nr", "station_name", "canton", "road"]:
        if col in raw.columns:
            raw[col] = raw[col].ffill()

    # ── Step 4: Drop rows with no metric_type (they're blank separator rows) ──
    if "metric_type" in raw.columns:
        raw = raw[raw["metric_type"].notna() & (raw["metric_type"].str.strip() != "")]
    else:
        return pd.DataFrame()  # Safety: return empty if structure unexpected

    # ── Step 5: Clean and type-convert numeric columns ────────────────────────
    month_cols = [f"adt_{m}" for m in MONTHS if f"adt_{m}" in raw.columns]
    for col in month_cols:
        raw[col] = raw[col].apply(clean_number)

    if "annual_avg" in raw.columns:
        raw["annual_avg"] = raw["annual_avg"].apply(clean_number)

    # ── Step 6: Clean station ID → integer ───────────────────────────────────
    # "nr" column contains values like "002", "003", "064"
    # We strip leading zeros and convert to integer.
    if "nr" in raw.columns:
        raw["station_id"] = raw["nr"].apply(
            lambda x: int(str(x).strip()) if pd.notna(x) and str(x).strip().isdigit() else None
        )
    else:
        raw["station_id"] = None

    # ── Step 7: Clean string columns ─────────────────────────────────────────
    for col in ["station_name", "canton", "road", "metric_type"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()

    # ── Step 8: Select and order final columns ────────────────────────────────
    final_cols = ["station_id", "station_name", "canton", "road", "metric_type"] + month_cols
    if "annual_avg" in raw.columns:
        final_cols.append("annual_avg")

    result = raw[[c for c in final_cols if c in raw.columns]].copy()

    # Remove rows where station_id could not be determined (malformed rows)
    result = result[result["station_id"].notna()].copy()
    result["station_id"] = result["station_id"].astype(int)

    return result


# =============================================================================
# HELPER FUNCTION 3: parse_station_notes
# =============================================================================

def parse_station_notes(filepath: str) -> pd.DataFrame:
    """
    Parse the station data-quality notes CSV.

    This file documents:
    - "No data" periods (sensor malfunction, construction)
    - "Winter closing" periods (mountain passes closed by snow)
    - "Estimated" months (partial data, value was interpolated)

    STRUCTURE OF THIS CSV:
    Nr. | Measuring station | | from     | until     | Notes
    005 | SCHWANDEN N      | | 01.01.2025| 31.12.2025| No data
        |                  | | 01.08.2025| 31.08.2025| No data  ← same station
    010 | HOSPENTAL...     | | 01.01.2025| 15.05.2025| Winter closing

    WHY WE NEED THIS:
    Machine learning models can't use missing data. Knowing WHY data is missing
    (planned closure vs sensor fault) helps us decide whether to impute (fill in
    estimated values) or drop the station/month entirely.

    PARAMETERS:
        filepath: path to Annual_results_ Measuring_station_2025.csv

    RETURNS:
        pd.DataFrame with columns:
            station_id, station_name, gap_from, gap_until, notes
    """
    raw = pd.read_csv(
        filepath,
        skiprows=HEADER_SKIP_ROWS - 1,  # This file has 4 metadata rows, not 5
        header=0,
        dtype=str,
        keep_default_na=False,
        encoding="latin-1",
    )

    # Rename columns to predictable names
    cols = list(raw.columns)
    rename_map = {}
    if len(cols) > 0: rename_map[cols[0]] = "nr"
    if len(cols) > 1: rename_map[cols[1]] = "station_name"
    if len(cols) > 3: rename_map[cols[3]] = "gap_from"
    if len(cols) > 4: rename_map[cols[4]] = "gap_until"
    if len(cols) > 5: rename_map[cols[5]] = "notes"

    raw = raw.rename(columns=rename_map)

    # Forward-fill station ID and name
    for col in ["nr", "station_name"]:
        if col in raw.columns:
            raw[col] = raw[col].replace("", pd.NA).ffill()

    # Drop rows without notes (blank separator rows in the CSV)
    if "notes" in raw.columns:
        raw = raw[raw["notes"].notna() & (raw["notes"].str.strip() != "")]

    # Parse station ID
    if "nr" in raw.columns:
        raw["station_id"] = raw["nr"].apply(
            lambda x: int(str(x).strip()) if pd.notna(x) and str(x).strip().isdigit() else None
        )

    for col in ["station_name", "gap_from", "gap_until", "notes"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()

    final_cols = [c for c in ["station_id", "station_name", "gap_from", "gap_until", "notes"] if c in raw.columns]
    return raw[final_cols].copy()


# =============================================================================
# MAIN EXECUTION — load_to_duckdb
# =============================================================================

def load_to_duckdb(
    df: pd.DataFrame,
    table_name: str,
    con: duckdb.DuckDBPyConnection,
    schema: str = "raw",
) -> None:
    """
    Write a pandas DataFrame to a DuckDB table, replacing any existing data.

    WHY REPLACE (not APPEND)?
    In a pipeline, every run should be deterministic and idempotent.
    "Idempotent" (from mathematics) means: running twice gives the same result
    as running once. If we always REPLACE, reruns are safe. If we APPEND,
    reruns would double all the data.

    PARAMETERS:
        df          : the pandas DataFrame to write
        table_name  : e.g. "annual_results"
        con         : open DuckDB connection
        schema      : DuckDB schema name (default: "raw")
    """
    # Create the schema if it doesn't already exist.
    # A DuckDB "schema" is like a folder within the database.
    # We have: raw.*  staging.*  mart.*
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # Register the DataFrame as a temporary view so DuckDB can INSERT it.
    # This is faster than writing to CSV and re-reading.
    con.register("_tmp_df", df)

    # DROP + CREATE = always fresh data (no duplicate rows on re-run)
    full_table = f"{schema}.{table_name}"
    con.execute(f"DROP TABLE IF EXISTS {full_table}")
    con.execute(f"CREATE TABLE {full_table} AS SELECT * FROM _tmp_df")

    row_count = con.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]
    print(f"  [OK] Loaded {row_count:,} rows into {full_table}")


# =============================================================================
# ENTRY POINT — main()
# =============================================================================

def main():
    """
    Orchestrates the full ingestion:
      1. Validate that all CSV files exist
      2. Parse each CSV file
      3. Connect to DuckDB
      4. Write each DataFrame to a raw.* table
      5. Print a summary
    """
    print("=" * 65)
    print("  STAGE 1: RAW INGESTION — FEDRO CSV → DuckDB")
    print("=" * 65)
    print(f"  Database: {DB_PATH}")
    print()

    # ── Step 1: Validate input files exist ────────────────────────────────────
    for name, path in CSV_FILES.items():
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"[ERR] CSV file not found: {abs_path}\n"
                f"  Expected the file '{os.path.basename(abs_path)}' to be in the "
                f"project root folder. Please check the file name and try again."
            )
        print(f"  [FOUND] {name}: {os.path.basename(abs_path)}")

    print()

    # ── Step 2: Parse CSV files ────────────────────────────────────────────────

    print("  Parsing Annual_results_2025.csv (monthly ADT all metrics)...")
    df_annual = parse_main_csv(os.path.abspath(CSV_FILES["annual"]))
    print(f"  → {len(df_annual):,} rows parsed  ({df_annual['station_id'].nunique()} unique stations)")

    print("  Parsing adtwithclasses CSV (ADT + vehicle class breakdown)...")
    df_classes = parse_main_csv(os.path.abspath(CSV_FILES["adtwithclasses"]))
    print(f"  → {len(df_classes):,} rows parsed  ({df_classes['station_id'].nunique()} unique stations)")

    print("  Parsing station notes CSV (data gaps + closures)...")
    df_notes = parse_station_notes(os.path.abspath(CSV_FILES["station_notes"]))
    print(f"  → {len(df_notes):,} gap records parsed  ({df_notes['station_id'].nunique()} affected stations)")

    print()

    # ── Step 3: Connect to DuckDB and write tables ────────────────────────────

    print(f"  Connecting to DuckDB at: {DB_PATH}")
    # check_same_thread=False is safe here since Bruin runs assets sequentially.
    con = duckdb.connect(database=DB_PATH)

    print("  Writing to DuckDB tables...")
    load_to_duckdb(df_annual, "annual_results", con)
    load_to_duckdb(df_classes, "adtwithclasses", con)
    load_to_duckdb(df_notes, "station_notes", con)

    con.close()

    # ── Step 4: Summary ────────────────────────────────────────────────────────
    print()
    print("  DATA QUALITY OVERVIEW:")
    romandy_stations = df_annual[df_annual["canton"].isin(ROMANDY_CANTONS)]["station_id"].nunique()
    adt_rows = df_annual[df_annual["metric_type"] == "ADT"]
    complete_stations = adt_rows.dropna(subset=[f"adt_{m}" for m in MONTHS if f"adt_{m}" in adt_rows.columns]).shape[0]
    print(f"  Total stations:              {df_annual['station_id'].nunique():>6}")
    print(f"  Romandy (FR-speaking):       {romandy_stations:>6}")
    print(f"  Stations with NO gaps:       {complete_stations:>6}")
    print(f"  Stations with gap records:   {df_notes['station_id'].nunique():>6}")
    print()
    print("  STAGE 1 COMPLETE. Next step: bruin run assets/staging/")
    print("=" * 65)


# Python convention: only run main() when this file is executed directly,
# not when it's imported as a module by another script.
if __name__ == "__main__":
    main()
