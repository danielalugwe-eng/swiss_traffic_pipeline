"""
@bruin
name: reporting.generate_reports
type: python

description: >
  Generate all historical analysis visualisations and an HTML report from
  the mart layer DuckDB tables.
  Outputs:
    reports/lausanne_monthly_traffic.png
    reports/romandy_seasonal_patterns.png
    reports/station_heatmap.png
    reports/canton_comparison.png
    reports/historical_report.html

depends:
  - mart.lausanne_analysis
  - mart.romandy_summary
  - mart.traffic_features

tags:
  - reporting
  - visualization
  - romandy
@end

=============================================================================
WHAT IS THIS SCRIPT?
=============================================================================

This script is the "newspaper press" of our pipeline.
Once the data has been cleaned and summarised, this script turns it into
charts (PNG images) and a single-page HTML report that any stakeholder can
open in a browser — no Python required on their end.

WHY SEPARATE REPORTING FROM SQL?
SQL is excellent at aggregating and filtering data, but terrible at drawing
charts. Python libraries (matplotlib, seaborn) were built specifically for
visualisation and do the job far better.

LIBRARIES USED:
  • matplotlib — The foundation of Python charting.
                 Every chart type (bar, line, heatmap) is built on this.
  • seaborn    — Statistical charts on top of matplotlib.
                 Heatmaps, box-and-whisker plots, and regression charts
                 look much better with seaborn than raw matplotlib.
  • duckdb     — We connect to our database and pull the mart tables
                 into pandas DataFrames for plotting.
  • jinja2     — Template engine. We fill a HTML template with our data
                 and chart references to produce the final report file.

OUTPUT STRUCTURE:
  reports/
    ├── lausanne_monthly_traffic.png   ← Grouped bar chart by station × month
    ├── romandy_seasonal_patterns.png  ← Line chart: canton seasonal curves
    ├── station_heatmap.png            ← Heatmap: stations × months (all Romandy)
    ├── canton_comparison.png          ← Box plot: ADT distribution by canton
    └── historical_report.html         ← Full HTML report embedding all 4 charts
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os               # File path manipulation
import sys              # System exit on fatal errors
import duckdb           # Read mart tables from DuckDB
import pandas as pd     # Data manipulation before plotting
import numpy as np      # Numerical helpers (e.g., array operations for heatmap)
import matplotlib       # Core plotting engine — must set backend BEFORE pyplot import
matplotlib.use("Agg")   # "Agg" = non-interactive backend (renders to file, not screen)
                        # This is essential for running in headless pipelines/servers.
                        # Without it, matplotlib would try to open a GUI window and crash.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   # Custom number formatters on chart axes
import seaborn as sns   # Statistical visualisation (heatmaps, box plots)
from jinja2 import Template  # HTML report templating

# =============================================================================
# CONFIGURATION
# =============================================================================

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(THIS_DIR, "..", "..")
DB_PATH      = os.path.join(PROJECT_ROOT, "traffic.duckdb")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

# Create the reports/ folder if it doesn't exist yet.
# exist_ok=True = don't raise an error if the folder already exists.
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Visual style settings ---
# seaborn "whitegrid" style: white background with subtle horizontal grid lines.
# This is transport-industry standard for readability in reports.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Romandy cantons in display order (largest traffic first)
ROMANDY_CANTONS = ["VD", "GE", "VS", "NE", "FR", "JU"]

# Consistent colour palette — one colour per Romandy canton.
# These are sufficiently distinct to separate lines in a multi-canton chart.
CANTON_COLOURS = {
    "VD": "#003087",   # Deep navy  (Vaud official blue)
    "GE": "#E30613",   # Red        (Geneva official red)
    "VS": "#FFFFFF",   # White on dark bg (Valais — we override for line charts)
    "NE": "#009A44",   # Green      (Neuchâtel)
    "FR": "#000000",   # Black      (Fribourg)
    "JU": "#FF6B35",   # Orange     (Jura)
}
# Override VS white → dark gold for visibility on white backgrounds
CANTON_COLOURS["VS"] = "#C8A400"

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# HELPER: connect_db
# =============================================================================

def connect_db() -> duckdb.DuckDBPyConnection:
    """
    Open a read-only connection to the DuckDB database.

    WHY READ-ONLY?
    This is a reporting/visualisation script. It should NEVER modify data.
    Opening in read-only mode is a safety guarantee — if a bug somehow tries
    to write, DuckDB will raise an error rather than silently corrupting data.
    """
    if not os.path.exists(DB_PATH):
        print(f"[ERR] Database not found: {DB_PATH}")
        print("  Run the pipeline first:  bruin run .")
        sys.exit(1)
    return duckdb.connect(database=DB_PATH, read_only=True)


# =============================================================================
# HELPER: save_figure
# =============================================================================

def save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to the reports/ directory.

    PARAMETERS:
        fig      : the matplotlib Figure object
        filename : just the filename, e.g. "lausanne_monthly_traffic.png"

    RETURNS:
        absolute path to the saved file

    WHY DPI=150?
    DPI = "Dots Per Inch". Higher DPI = sharper image but larger file size.
    150 DPI is a good balance: crisp in reports and presentations without being
    unnecessarily large. 72 DPI looks blurry; 300 DPI is print-quality (6 MB+).

    WHY bbox_inches='tight'?
    Matplotlib sometimes clips axis labels at image edges.
    'tight' tells it to include all labels in the saved image.
    """
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)   # Free memory — matplotlib keeps figures in memory until closed
    print(f"  [SAVED] {filename}")
    return path


# =============================================================================
# CHART 1: Lausanne Monthly Traffic — Grouped Bar Chart
# =============================================================================

def chart_lausanne_monthly(con: duckdb.DuckDBPyConnection) -> str:
    """
    Grouped bar chart showing monthly ADT for each VD-canton (Lausanne area)
    measuring station across all 12 months of 2025.

    WHY A GROUPED BAR CHART?
    It lets you compare multiple stations side-by-side within the same month,
    AND compare the same station across different months.
    Perfect for answering: "Which month was busiest? Which station handles most traffic?"

    READING THIS CHART:
    - X-axis: months (Jan → Dec)
    - Y-axis: average daily traffic (vehicles/day)
    - Each colour = one measuring station
    - Tall bars = busy period.  Short bars = quiet period.
    """
    df = con.execute("""
        SELECT station_id, station_name, road, month_num, month_name,
               ROUND(adt_value, 0) AS adt_value
        FROM mart.lausanne_analysis
        ORDER BY station_id, month_num
    """).df()

    if df.empty:
        print("  [WARN] No Lausanne data found — skipping chart 1")
        return ""

    stations = df["station_name"].unique()
    n_stations = len(stations)
    months = list(range(1, 13))

    fig, ax = plt.subplots(figsize=(16, 6))

    # Bar width: total bar group width = 0.8, divided equally among stations
    bar_width = 0.8 / n_stations
    # x positions for month groups
    x = np.arange(len(months))

    for i, station in enumerate(stations):
        # Offset each station's bars so they sit side-by-side within the group
        offset = (i - n_stations / 2 + 0.5) * bar_width
        stn_df = df[df["station_name"] == station].sort_values("month_num")
        values = stn_df["adt_value"].values

        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.9,  # 0.9 = slight gap between bars for readability
            label=f"{station} ({stn_df['road'].iloc[0]})",
            alpha=0.85,
        )

    ax.set_title("Lausanne Area — Monthly Average Daily Traffic (ADT) 2025",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)

    # Format Y-axis with thousands separator: 97000 → "97,000"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.legend(title="Station", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.annotate(
        "Source: FEDRO/ASTRA Annual Bulletin 2025",
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="grey",
    )

    return save_figure(fig, "lausanne_monthly_traffic.png")


# =============================================================================
# CHART 2: Romandy Seasonal Patterns — Multi-line Chart
# =============================================================================

def chart_romandy_seasonal(con: duckdb.DuckDBPyConnection) -> str:
    """
    Line chart showing the seasonal traffic curve (Jan–Dec) for each of the
    6 French-speaking cantons, using the canton-level ADT averages.

    WHY A LINE CHART?
    Lines show trends and patterns over time better than bars.
    You can immediately see:
    - Summer peaks (July/August bump for tourist cantons like VS)
    - Stable flat lines (urban commuter cantons like GE)
    - How different the seasonal rhythm is between cantons

    READING THIS CHART:
    - X-axis: months (Jan → Dec)
    - Y-axis: canton average ADT (vehicles/day)
    - Each line = one Romandy canton
    - Sharp summer peak = tourism-dominated roads
    - Flat line = commuter/freight-dominated roads
    """
    df = con.execute("""
        SELECT t.canton, t.month_num, t.month_name,
               ROUND(AVG(t.adt_value), 0) AS avg_adt
        FROM staging.stg_monthly_traffic t
        WHERE t.is_romandy = TRUE
          AND t.metric_type = 'ADT'
          AND t.adt_value IS NOT NULL
        GROUP BY t.canton, t.month_num, t.month_name
        ORDER BY t.canton, t.month_num
    """).df()

    if df.empty:
        print("  [WARN] No Romandy summary data — skipping chart 2")
        return ""

    fig, ax = plt.subplots(figsize=(13, 6))

    for canton in ROMANDY_CANTONS:
        c_df = df[df["canton"] == canton].sort_values("month_num")
        if c_df.empty:
            continue
        ax.plot(
            c_df["month_num"],
            c_df["avg_adt"],
            marker="o",         # Dots at each data point (1 per month)
            linewidth=2.2,
            markersize=5,
            color=CANTON_COLOURS.get(canton, "#333333"),
            label=canton,
        )

    ax.set_title("Romandy Cantons — Seasonal Traffic Pattern 2025",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Shade summer season (June–August) with a light yellow background
    ax.axvspan(6, 8, alpha=0.08, color="gold", label="Summer (Jun–Aug)")

    ax.legend(title="Canton", loc="upper right", fontsize=10)
    ax.annotate(
        "Source: FEDRO/ASTRA Annual Bulletin 2025",
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="grey",
    )

    return save_figure(fig, "romandy_seasonal_patterns.png")


# =============================================================================
# CHART 3: Station Heatmap — All Romandy Stations × 12 Months
# =============================================================================

def chart_station_heatmap(con: duckdb.DuckDBPyConnection) -> str:
    """
    Heatmap where:
    - Rows = individual measuring stations (all Romandy stations)
    - Columns = months (Jan → Dec)
    - Colour intensity = ADT value (darker = more traffic)

    WHY A HEATMAP?
    A heatmap is the best way to display a matrix of values where you want to
    spot patterns visually: "Which stations are always dark? Which brighten in summer?"
    It's like a temperature map of a city, but for traffic.

    HOW TO READ IT:
    - Dark blue row = high-traffic station all year (urban motorway)
    - Dark blue only in July/Aug = tourist seasonal road
    - Light row = low-volume route (rural national road)
    - White cell = missing data (sensor offline / winter closure)

    NORMALISATION:
    We normalise each ROW (station) independently so you can compare seasonal
    patterns even between a 97,000 ADT/day urban motorway and a 3,000 ADT/day
    mountain pass. Without normalisation, the busy stations would dominate and
    all mountain passes would look identical (all very light).
    """
    df = con.execute("""
        SELECT t.station_id, t.station_name, t.month_num, ROUND(t.adt_value, 0) AS adt_value
        FROM staging.stg_monthly_traffic t
        WHERE t.is_romandy = TRUE
          AND t.metric_type = 'ADT'
        ORDER BY t.station_id, t.month_num
    """).df()

    if df.empty:
        print("  [WARN] No Romandy monthly data — skipping chart 3")
        return ""

    # Pivot to wide format: rows=stations, columns=months
    pivot = df.pivot_table(
        index="station_name",
        columns="month_num",
        values="adt_value",
        aggfunc="mean",
    )
    # Rename column numbers to month abbreviations
    pivot.columns = MONTH_LABELS[: len(pivot.columns)]

    # Row-normalise: each value becomes a 0–100 percentile within its station.
    # This reveals SEASONAL SHAPE regardless of absolute volume.
    row_min = pivot.min(axis=1)
    row_max = pivot.max(axis=1)
    pivot_norm = pivot.sub(row_min, axis=0).div(
        row_max.sub(row_min, axis=0).replace(0, 1), axis=0
    ) * 100

    fig_height = max(8, len(pivot_norm) * 0.28)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        pivot_norm,
        ax=ax,
        cmap="Blues",          # Blue palette: white=low, dark blue=high
        linewidths=0.3,         # Thin grid lines between cells
        linecolor="white",
        annot=False,            # No text inside cells (too many stations)
        cbar_kws={"label": "Relative Traffic (0=min, 100=max for each station)"},
        mask=pivot_norm.isna(), # Show missing cells as white (not coloured)
    )

    ax.set_title("Romandy Measuring Stations — Seasonal Traffic Intensity 2025\n"
                 "(Row-normalised: each row's colours relative to its own min/max)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Measuring Station", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)

    return save_figure(fig, "station_heatmap.png")


# =============================================================================
# CHART 4: Canton Comparison Box Plot
# =============================================================================

def chart_canton_comparison(con: duckdb.DuckDBPyConnection) -> str:
    """
    Box-and-whisker plot comparing traffic volume distributions across
    Romandy cantons.

    WHAT IS A BOX PLOT?
    A box plot shows 5 statistics in one compact shape:
    ─── Whisker top    = Maximum value (ignoring outliers)
    ┌───────────────┐
    │               │  ← 75th percentile (Q3): 75% of values are below this
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ← Median (50th percentile): the "middle" value
    │               │  ← 25th percentile (Q1): 25% of values are below this
    └───────────────┘
    ─── Whisker bottom = Minimum value (ignoring outliers)
    ●                  = Outliers (individual unusual stations)

    WHY THIS IS USEFUL FOR TRAFFIC:
    It answers: "Does Geneva always have high traffic, or is it variable?"
    If GE has a narrow box → consistent. Wide box → highly variable stations.
    """
    df = con.execute("""
        SELECT t.canton, t.adt_value
        FROM staging.stg_monthly_traffic t
        WHERE t.is_romandy = TRUE
          AND t.metric_type = 'ADT'
          AND t.adt_value IS NOT NULL
    """).df()

    if df.empty:
        print("  [WARN] No Romandy ADT data — skipping chart 4")
        return ""

    # Filter to only Romandy cantons in our ordered list
    df = df[df["canton"].isin(ROMANDY_CANTONS)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="canton",
        y="adt_value",
        order=ROMANDY_CANTONS,       # Consistent left-to-right order
        palette=[CANTON_COLOURS.get(c, "#999") for c in ROMANDY_CANTONS],
        width=0.5,
        fliersize=4,                 # Size of outlier dots
        ax=ax,
    )

    ax.set_title("Romandy Cantons — Traffic Volume Distribution (All Stations, All Months)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Canton", fontsize=11)
    ax.set_ylabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Add canton full names as annotations below x-tick labels
    canton_names = {
        "VD": "Vaud", "GE": "Geneva", "VS": "Valais",
        "NE": "Neuchâtel", "FR": "Fribourg", "JU": "Jura",
    }
    ax.set_xticklabels([
        f"{c}\n({canton_names.get(c, '')})" for c in ROMANDY_CANTONS
    ])

    ax.annotate(
        "Source: FEDRO/ASTRA Annual Bulletin 2025",
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="grey",
    )

    return save_figure(fig, "canton_comparison.png")


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Swiss Traffic Historical Report 2025 — Romandy</title>
  <style>
    /* ── Global reset and typography ─────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa;
           color: #222; line-height: 1.6; }

    /* ── Header ──────────────────────────────────────────────────────────── */
    header { background: linear-gradient(135deg, #003087 0%, #0057B7 100%);
             color: white; padding: 2.5rem 3rem; }
    header h1 { font-size: 2rem; font-weight: 700; }
    header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
    .badge    { display: inline-block; background: rgba(255,255,255,0.2);
                border-radius: 4px; padding: 2px 10px; font-size: 0.8rem;
                margin-top: 0.7rem; margin-right: 6px; }

    /* ── Main content area ───────────────────────────────────────────────── */
    main { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }

    /* ── Summary KPI cards ───────────────────────────────────────────────── */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem; margin-bottom: 2.5rem; }
    .kpi-card { background: white; border-radius: 8px; padding: 1.2rem 1.5rem;
                border-left: 4px solid #003087; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .kpi-card .label { font-size: 0.78rem; color: #666; text-transform: uppercase;
                       letter-spacing: 0.08em; }
    .kpi-card .value { font-size: 1.8rem; font-weight: 700; color: #003087;
                       margin-top: 0.2rem; }
    .kpi-card .sub   { font-size: 0.8rem; color: #888; margin-top: 0.1rem; }

    /* ── Section containers ──────────────────────────────────────────────── */
    .section { background: white; border-radius: 8px; padding: 1.8rem 2rem;
               margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .section h2 { font-size: 1.3rem; font-weight: 600; color: #003087;
                  border-bottom: 2px solid #e8edf5; padding-bottom: 0.7rem;
                  margin-bottom: 1.2rem; }
    .section p  { color: #444; font-size: 0.95rem; margin-bottom: 0.8rem; }

    /* ── Charts ──────────────────────────────────────────────────────────── */
    .chart-img { width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    .chart-caption { font-size: 0.82rem; color: #777; margin-top: 0.5rem;
                     text-align: center; font-style: italic; }

    /* ── Summary table ───────────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem;
            margin-top: 1rem; }
    th { background: #003087; color: white; padding: 0.6rem 0.8rem;
         text-align: left; font-weight: 600; }
    td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #e8edf5; }
    tr:hover td { background: #f0f4fb; }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    footer { text-align: center; padding: 2rem; color: #999; font-size: 0.82rem; }
  </style>
</head>
<body>
  <header>
    <h1>🚦 Swiss Traffic Historical Report — Romandy 2025</h1>
    <p>Annual bulletin analysis of French-speaking Switzerland measuring stations</p>
    <span class="badge">FEDRO / ASTRA Data</span>
    <span class="badge">2025 Annual Bulletin</span>
    <span class="badge">Generated: {{ generated_at }}</span>
  </header>

  <main>

    <!-- ── KPI Summary Cards ─────────────────────────────────────────────── -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="label">Total Romandy Stations</div>
        <div class="value">{{ total_stations }}</div>
        <div class="sub">With annual ADT data</div>
      </div>
      <div class="kpi-card">
        <div class="label">Highest Daily Traffic</div>
        <div class="value">{{ peak_station_adt }}</div>
        <div class="sub">{{ peak_station_name }}</div>
      </div>
      <div class="kpi-card">
        <div class="label">Lausanne A9 Annual ADT</div>
        <div class="value">{{ lausanne_adt }}</div>
        <div class="sub">Cont. de Lausanne (Station 064)</div>
      </div>
      <div class="kpi-card">
        <div class="label">Cantons Covered</div>
        <div class="value">6</div>
        <div class="sub">VD · GE · VS · NE · FR · JU</div>
      </div>
      <div class="kpi-card">
        <div class="label">Peak Month (Romandy avg)</div>
        <div class="value">{{ peak_month }}</div>
        <div class="sub">Highest average cross-canton</div>
      </div>
    </div>

    <!-- ── Chart 1: Lausanne Monthly ─────────────────────────────────────── -->
    <div class="section">
      <h2>📊 Lausanne Area — Monthly Traffic Breakdown</h2>
      <p>
        The Lausanne metropolitan area sits at the intersection of the A1 (Geneva–St. Gallen)
        and A9 (Lausanne–Valais) motorways. Station 064 (Contournement de Lausanne)
        is the busiest French-speaking Swiss measuring point with ~97,600 vehicles/day.
      </p>
      <p>
        <strong>How to read this chart:</strong> Each colour represents one measuring station.
        Taller bars = more traffic. Compare July (tourist peak) with January (winter low).
      </p>
      {% if chart_lausanne %}
      <img src="{{ chart_lausanne }}" alt="Lausanne Monthly Traffic" class="chart-img">
      <p class="chart-caption">Fig 1 — Monthly ADT for Lausanne/Vaud area stations, 2025</p>
      {% else %}
      <p><em>Chart not generated — run pipeline to produce data.</em></p>
      {% endif %}
    </div>

    <!-- ── Chart 2: Romandy Seasonal ─────────────────────────────────────── -->
    <div class="section">
      <h2>📈 Romandy Cantons — Seasonal Patterns</h2>
      <p>
        Each Romandy canton has a distinct seasonal traffic signature depending on
        its economic character: commuter networks (GE — flat), tourist routes
        (VS — strong summer peak), and mixed-use corridors (VD — moderate uplift).
      </p>
      <p>
        <strong>Highlight:</strong> Valais (VS) shows the highest relative summer peak
        due to Alpine tourism (Verbier, Zermatt, Saas-Fee resort access roads).
        Geneva (GE) stays flat all year — dominated by international commuters and airport traffic.
      </p>
      {% if chart_seasonal %}
      <img src="{{ chart_seasonal }}" alt="Romandy Seasonal Patterns" class="chart-img">
      <p class="chart-caption">Fig 2 — Canton-level average ADT by month, 2025. Shaded area = summer peak season.</p>
      {% endif %}
    </div>

    <!-- ── Chart 3: Heatmap ───────────────────────────────────────────────── -->
    <div class="section">
      <h2>🗺️ All Romandy Stations — Seasonal Intensity Heatmap</h2>
      <p>
        This heatmap shows every Romandy measuring station (rows) across all 12 months
        (columns). Colour intensity is row-normalised: darker blue = busier relative to
        that station's own yearly average. This reveals seasonal SHAPE, not absolute volume.
      </p>
      <p>
        <strong>Pattern to look for:</strong> Stations with a single dark column in July/August
        are alpine/resort access roads. Stations with uniform dark colour year-round are
        urban motorways and freight corridors.
      </p>
      {% if chart_heatmap %}
      <img src="{{ chart_heatmap }}" alt="Station Heatmap" class="chart-img">
      <p class="chart-caption">Fig 3 — Seasonal traffic intensity heatmap, all Romandy stations (row-normalised), 2025</p>
      {% endif %}
    </div>

    <!-- ── Chart 4: Canton Box Plot ───────────────────────────────────────── -->
    <div class="section">
      <h2>📦 Canton Traffic Distribution</h2>
      <p>
        Box plots show the spread of traffic values across all stations and all months
        within each canton. A narrow box means consistent traffic across the canton.
        A tall box means high variability — some roads are very busy, others very quiet.
      </p>
      {% if chart_box %}
      <img src="{{ chart_box }}" alt="Canton Comparison" class="chart-img">
      <p class="chart-caption">Fig 4 — ADT distribution across all Romandy stations and months by canton, 2025</p>
      {% endif %}
    </div>

    <!-- ── Canton Summary Table ───────────────────────────────────────────── -->
    <div class="section">
      <h2>📋 Canton Summary Table</h2>
      <table>
        <thead>
          <tr>
            <th>Canton</th>
            <th>Stations</th>
            <th>Avg Annual ADT</th>
            <th>Peak Month</th>
            <th>Trough Month</th>
            <th>Seasonal Ratio</th>
          </tr>
        </thead>
        <tbody>
          {% for row in canton_table %}
          <tr>
            <td><strong>{{ row.canton }}</strong></td>
            <td>{{ row.station_count }}</td>
            <td>{{ row.avg_annual_adt }}</td>
            <td>{{ row.peak_month }}</td>
            <td>{{ row.trough_month }}</td>
            <td>{{ row.seasonal_ratio }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

  </main>
  <footer>
    <p>Swiss Traffic MLOps Pipeline · Data: FEDRO/ASTRA Annual Bulletin 2025 ·
       French-speaking Switzerland (Romandy) Analysis</p>
  </footer>
</body>
</html>
"""


def generate_html_report(
    con: duckdb.DuckDBPyConnection,
    chart_paths: dict,
) -> str:
    """
    Render the Jinja2 HTML template with real data from DuckDB and the
    paths to the generated chart images.

    HOW JINJA2 TEMPLATING WORKS:
    The HTML_TEMPLATE string above contains {{ variable }} placeholders and
    {% for row in list %} loop blocks. Jinja2's Template.render() replaces
    those with real values — like a mail-merge for HTML.

    PARAMETERS:
        con         : open DuckDB connection (read-only)
        chart_paths : dict mapping chart keys to their file paths

    RETURNS:
        absolute path to the saved HTML file
    """
    from datetime import datetime

    # ── Fetch KPI values from DuckDB ──────────────────────────────────────
    kpis = con.execute("""
        SELECT
            COUNT(DISTINCT station_id) AS total_stations,
            MAX(annual_adt)            AS peak_adt
        FROM staging.stg_stations
        WHERE is_romandy = TRUE
    """).fetchone()

    peak_name_row = con.execute("""
        SELECT station_name FROM staging.stg_stations
        WHERE is_romandy = TRUE
        ORDER BY annual_adt DESC NULLS LAST LIMIT 1
    """).fetchone()

    total_stations = kpis[0] if kpis else "—"
    peak_adt       = f"{int(kpis[1]):,}" if kpis and kpis[1] else "—"
    peak_name      = peak_name_row[0] if peak_name_row else "—"

    lausanne_row = con.execute("""
        SELECT annual_adt FROM staging.stg_stations
        WHERE station_id = 64
        LIMIT 1
    """).fetchone()
    lausanne_adt = f"{int(lausanne_row[0]):,}" if lausanne_row and lausanne_row[0] else "—"

    # Peak month across all Romandy stations
    peak_month_row = con.execute("""
        SELECT month_name
        FROM (
            SELECT month_name, AVG(adt_value) AS avg_adt
            FROM staging.stg_monthly_traffic
            WHERE is_romandy = TRUE AND metric_type = 'ADT' AND adt_value IS NOT NULL
            GROUP BY month_name, month_num
            ORDER BY avg_adt DESC LIMIT 1
        )
    """).fetchone()
    peak_month = peak_month_row[0] if peak_month_row else "July"

    # Canton summary table
    canton_rows = con.execute("""
        SELECT
            canton,
            COUNT(DISTINCT station_id)             AS station_count,
            ROUND(AVG(adt_value), 0)               AS avg_annual_adt,
            NULL AS peak_month, NULL AS trough_month,
            NULL AS seasonal_ratio
        FROM staging.stg_monthly_traffic
        WHERE is_romandy = TRUE AND metric_type = 'ADT'
        GROUP BY canton
        ORDER BY AVG(adt_value) DESC
    """).df()

    canton_table = [
        {
            "canton": r.canton,
            "station_count": int(r.station_count),
            "avg_annual_adt": f"{int(r.avg_annual_adt):,}",
            "peak_month": "—",
            "trough_month": "—",
            "seasonal_ratio": "—",
        }
        for _, r in canton_rows.iterrows()
    ]

    # Resolve chart paths to relative filenames for HTML <img src>
    def rel_path(abs_path: str) -> str:
        if not abs_path:
            return ""
        # Use just the filename — report sits in the same folder as the images
        return os.path.basename(abs_path)

    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        generated_at=datetime.now().strftime("%d %B %Y %H:%M"),
        total_stations=total_stations,
        peak_station_adt=peak_adt,
        peak_station_name=peak_name,
        lausanne_adt=lausanne_adt,
        peak_month=peak_month,
        chart_lausanne=rel_path(chart_paths.get("lausanne")),
        chart_seasonal=rel_path(chart_paths.get("seasonal")),
        chart_heatmap=rel_path(chart_paths.get("heatmap")),
        chart_box=rel_path(chart_paths.get("box")),
        canton_table=canton_table,
    )

    report_path = os.path.join(REPORTS_DIR, "historical_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [SAVED] historical_report.html")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  STAGE 4: REPORTING — Charts + HTML Report")
    print("=" * 65)
    print(f"  Output directory: {REPORTS_DIR}")
    print()

    con = connect_db()

    print("  Generating charts...")
    chart_paths = {}
    chart_paths["lausanne"] = chart_lausanne_monthly(con)
    chart_paths["seasonal"] = chart_romandy_seasonal(con)
    chart_paths["heatmap"]  = chart_station_heatmap(con)
    chart_paths["box"]      = chart_canton_comparison(con)

    print()
    print("  Rendering HTML report...")
    generate_html_report(con, chart_paths)

    con.close()

    print()
    print("  STAGE 4 COMPLETE.")
    print(f"  Open the report: {os.path.join(REPORTS_DIR, 'historical_report.html')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
