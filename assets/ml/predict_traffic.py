"""
@bruin
name: ml.predict_traffic
type: python

description: >
  Use the trained best_model.pkl to generate 2026 traffic volume predictions
  for all Romandy measuring stations. Produces a predictions CSV, a
  visualisation of predicted vs 2025 actual, and writes results to DuckDB.

depends:
  - ml.evaluate_model

tags:
  - ml
  - prediction
  - romandy
@end

=============================================================================
WHAT IS THIS SCRIPT?
=============================================================================

The "graduation ceremony" of the ML pipeline.

Once the model is trained and evaluated, we use it to PREDICT FUTURE TRAFFIC.
In a production system, you would get live FEDRO sensor feeds every month,
run this script, and have predictions for the coming months.

For this project, we simulate future predictions by:
  1. Using Jan–Sep 2025 as our known feature data (the "past")
  2. Predicting Oct–Dec 2025 (which we can validate against real data)
  3. Then extrapolating to estimate Jan–Dec 2026 using a growth factor

GROWTH FACTOR APPROACH:
We don't have 2026 data yet (the annual bulletin comes out in early 2027).
So we apply a conservative annual growth factor of +0.8% to 2025 values.
This aligns with FEDRO's historical average annual traffic growth in Switzerland
(source: FEDRO traffic monitoring reports, ~0.5–1.2% per year for national roads).

OUTPUT FILES:
  reports/predictions_romandy.csv     ← All predictions in tabular form
  reports/predictions_chart.png       ← Visual comparison chart
  reports/predictions_lausanne.png    ← Lausanne-specific deep dive
  DuckDB: mart.predictions            ← Query-able predictions table

=============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import numpy as np
import pandas as pd
import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(THIS_DIR, "..", "..")
DB_PATH      = os.path.join(PROJECT_ROOT, "traffic.duckdb")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")

os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Romandy cantons we focus predictions on
ROMANDY_CANTONS = ["VD", "GE", "VS", "NE", "FR", "JU"]

# Annual traffic growth rate assumption for 2026 projections
# FEDRO historical data suggests ~0.5–1.2% annual growth for Swiss national roads.
# We use 0.8% as a conservative midpoint.
GROWTH_RATE_2026 = 0.008

TARGET_COLS  = ["adt_oct", "adt_nov", "adt_dec"]
TARGET_NAMES = {"adt_oct": "October", "adt_nov": "November", "adt_dec": "December"}


# =============================================================================
# HELPER: save_fig
# =============================================================================

def save_fig(fig, filename):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {filename}")
    return path


# =============================================================================
# PREDICTIONS CHART: 2025 Actual vs Model Predictions for Romandy
# =============================================================================

def chart_predictions_overview(df_pred: pd.DataFrame) -> str:
    """
    Horizontal bar chart comparing:
      Left side:  Actual 2025 annual ADT (known)
      Right side: Model-predicted Oct–Dec 2025, plus Jan–Sep actuals

    This allows stakeholders to see where the model's Q4 prediction differs
    from the actual (if Oct–Dec actuals are available) or to understand the
    full-year estimate.

    For each station we show:
      - 2025 Annual Avg ADT (from FEDRO official bulletin)
      - Model average of Oct + Nov + Dec prediction
    """
    # Focus: top 20 Romandy stations by volume (most policy-relevant)
    top_stations = df_pred.sort_values("annual_adt_2025", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(13, 9))

    y_pos = np.arange(len(top_stations))
    bar_h = 0.35

    bars_actual = ax.barh(
        y_pos + bar_h / 2,
        top_stations["annual_adt_2025"],
        height=bar_h,
        color="#003087",
        alpha=0.82,
        label="2025 Annual ADT (FEDRO official)",
    )
    bars_pred = ax.barh(
        y_pos - bar_h / 2,
        top_stations["pred_q4_avg"],
        height=bar_h,
        color="#E87722",
        alpha=0.82,
        label="Model: Avg Predicted Q4 ADT",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_stations["station_name"], fontsize=8)
    ax.invert_yaxis()   # Highest traffic at the top

    ax.set_xlabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.set_title("Top 20 Romandy Stations — 2025 Annual ADT vs Model Q4 Prediction",
                 fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate(
        "Source: FEDRO/ASTRA 2025 · Predictions: Swiss Traffic MLOps v1.0",
        xy=(0.99, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7.5, color="grey",
    )

    return save_fig(fig, "predictions_chart.png")


# =============================================================================
# LAUSANNE DEEP-DIVE PREDICTION CHART
# =============================================================================

def chart_lausanne_prediction(df_pred: pd.DataFrame) -> str:
    """
    Line chart showing for Lausanne area stations:
      - Actual monthly ADT Jan–Sep 2025 (known — solid line)
      - Predicted monthly ADT Oct–Dec 2025 (dashed line, different colour)
      - Estimated 2026 full year (lighter dotted line)

    This is the "flagship" chart of the project — showing the full
    Jan 2025 → Dec 2026 traffic story for the most important stations.

    HOW TO READ IT:
    - The solid part (left) = real measured data from FEDRO
    - The dashed part (right of vertical line) = what our model predicts
    - The vertical line = the train/test split point (end of September)
    """
    # Get Lausanne-area stations (VD canton with highest annual traffic)
    lausanne = df_pred[df_pred["canton"] == "VD"].sort_values(
        "annual_adt_2025", ascending=False
    ).head(4)

    if lausanne.empty:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))

    MONTH_NUMS = list(range(1, 13))
    MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    colours = ["#003087", "#E87722", "#009A44", "#C8A400"]

    for i, (_, row) in enumerate(lausanne.iterrows()):
        colour = colours[i % len(colours)]

        # ── Known months (Jan–Sep) — solid line ───────────────────────────
        known_vals = [
            row.get("adt_jan"), row.get("adt_feb"), row.get("adt_mar"),
            row.get("adt_apr"), row.get("adt_may"), row.get("adt_jun"),
            row.get("adt_jul"), row.get("adt_aug"), row.get("adt_sep"),
        ]
        ax.plot(
            MONTH_NUMS[:9], known_vals,
            color=colour, linewidth=2.2, marker="o", markersize=5,
            label=f"{row['station_name']} (actual)",
            solid_capstyle="round",
        )

        # ── Predicted months (Oct–Dec) — dashed line ──────────────────────
        predicted_vals = [
            row.get("pred_oct"), row.get("pred_nov"), row.get("pred_dec")
        ]
        # Connect the last known point to the first predicted — bridging line
        bridge_x = [9, 10]
        bridge_y = [known_vals[-1], predicted_vals[0]] if known_vals[-1] and predicted_vals[0] else [None, None]
        ax.plot(bridge_x, bridge_y, color=colour, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.plot(
            MONTH_NUMS[9:], predicted_vals,
            color=colour, linewidth=2.0, linestyle="--",
            marker="^", markersize=6,
            label=f"{row['station_name']} (predicted)",
        )

    # Mark the train/test boundary
    ax.axvline(9.5, color="#c0392b", linestyle=":", linewidth=1.8, alpha=0.8)
    ax.text(9.55, ax.get_ylim()[1] * 0.97, "  Train → Test split",
            color="#c0392b", fontsize=8.5, va="top")

    ax.set_xticks(MONTH_NUMS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month (2025)", fontsize=11)
    ax.set_ylabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.set_title("Lausanne Area Stations — Actual vs Predicted Monthly Traffic 2025\n"
                 "Solid = measured   Dashed = model prediction",
                 fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    return save_fig(fig, "predictions_lausanne.png")


# =============================================================================
# 2026 PROJECTION CHART
# =============================================================================

def chart_2026_projections(df_pred: pd.DataFrame) -> str:
    """
    Bar chart showing estimated 2026 annual ADT for each Romandy canton,
    vs the 2025 baseline, using the GROWTH_RATE_2026 factor.

    This gives planners a simple answer to: "How much more traffic should
    we expect in 2026 vs 2025?"

    The growth projection is linear (ADT_2026 = ADT_2025 × 1.008).
    A more sophisticated model would use station-type-specific growth rates
    (alpine tourist roads growing faster in summer, freight routes growing
    steadily, urban motorways already at near-capacity).
    """
    canton_2025 = (
        df_pred[df_pred["canton"].isin(ROMANDY_CANTONS)]
        .groupby("canton")["annual_adt_2025"]
        .mean()
        .reset_index()
    )
    canton_2025["adt_2026_est"] = canton_2025["annual_adt_2025"] * (1 + GROWTH_RATE_2026)
    canton_2025["growth_abs"]   = canton_2025["adt_2026_est"] - canton_2025["annual_adt_2025"]

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(canton_2025))
    w = 0.35

    ax.bar(x - w/2, canton_2025["annual_adt_2025"], w, label="2025 Actual (FEDRO)",
           color="#003087", alpha=0.85)
    ax.bar(x + w/2, canton_2025["adt_2026_est"],    w, label=f"2026 Estimate (+{GROWTH_RATE_2026*100:.1f}% growth)",
           color="#E87722", alpha=0.85)

    for i, row in canton_2025.iterrows():
        ax.text(i + w/2, row["adt_2026_est"] + 200,
                f"+{int(row['growth_abs']):,}", ha="center", fontsize=8, color="#E87722")

    ax.set_xticks(x)
    ax.set_xticklabels(canton_2025["canton"])
    ax.set_xlabel("Canton", fontsize=11)
    ax.set_ylabel("Average Daily Traffic (vehicles/day)", fontsize=11)
    ax.set_title(f"Romandy Cantons — 2025 Actual vs 2026 Projected Traffic\n"
                 f"Projection: +{GROWTH_RATE_2026*100:.1f}% annual growth (FEDRO historical average)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=10)

    return save_fig(fig, "predictions_2026_projections.png")


# =============================================================================
# WRITE PREDICTIONS TO DUCKDB
# =============================================================================

def write_predictions_to_db(df_pred: pd.DataFrame, con: duckdb.DuckDBPyConnection):
    """
    Write the predictions DataFrame to a new DuckDB table: mart.predictions.

    WHY WRITE BACK TO DUCKDB?
    Other tools (BI dashboards, analysts, future pipeline stages) can then
    query the predictions with SQL — instead of needing to re-run Python.
    This is the standard "serve predictions as data" pattern in MLOps.

    Example query after this step:
        SELECT station_name, canton, pred_oct, pred_nov, pred_dec
        FROM mart.predictions
        WHERE canton = 'VD'
        ORDER BY pred_oct DESC;
    """
    # Re-connect with write access
    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.execute("DROP TABLE IF EXISTS mart.predictions")
    con.register("_preds_tmp", df_pred)
    con.execute("CREATE TABLE mart.predictions AS SELECT * FROM _preds_tmp")
    n = con.execute("SELECT COUNT(*) FROM mart.predictions").fetchone()[0]
    print(f"  [DB] Written {n} prediction rows to mart.predictions")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  STAGE 7: TRAFFIC PREDICTIONS")
    print("=" * 65)
    print()

    # ── Load trained model ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    bundle       = joblib.load(MODEL_PATH)
    feature_cols = bundle["feature_cols"]
    metrics      = bundle.get("metrics", {})
    print("  Loaded best model bundle:")
    for target, m in metrics.items():
        print(f"    {target}: {m['model']}  R²={m['r2']:.4f}")

    # ── Load feature data ──────────────────────────────────────────────────
    con_r = duckdb.connect(database=DB_PATH, read_only=True)

    df = con_r.execute(f"""
        SELECT
            f.station_id, f.station_name, f.canton, f.road,
            f.road_category, f.is_romandy,
            {', '.join(f'f.{c}' for c in feature_cols)},
            f.adt_oct, f.adt_nov, f.adt_dec,
            s.annual_adt AS annual_adt_2025
        FROM mart.traffic_features f
        JOIN staging.stg_stations s ON f.station_id = s.station_id
        ORDER BY f.station_id
    """).df()
    con_r.close()

    print(f"\n  Predicting for {len(df)} stations")

    X = df[feature_cols].values

    # ── Generate predictions for each target month ─────────────────────────
    for target_col in TARGET_COLS:
        if target_col not in bundle:
            continue
        pipeline = bundle[target_col]
        short = target_col.split("_")[1]   # "oct", "nov", "dec"
        df[f"pred_{short}"] = pipeline.predict(X).round(0)

    # ── Q4 average (Oct + Nov + Dec) ───────────────────────────────────────
    df["pred_q4_avg"] = df[["pred_oct", "pred_nov", "pred_dec"]].mean(axis=1).round(0)

    # ── 2026 annual estimate (apply growth factor to 2025 annual) ──────────
    df["adt_2026_est"] = (df["annual_adt_2025"] * (1 + GROWTH_RATE_2026)).round(0)

    # ── Add metadata ───────────────────────────────────────────────────────
    df["prediction_date"] = datetime.now().strftime("%Y-%m-%d")
    df["model_version"]   = "v1.0"
    df["growth_rate_used"] = GROWTH_RATE_2026

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(REPORTS_DIR, "predictions_romandy.csv")
    romandy_preds = df[df["is_romandy"]].copy()
    romandy_preds.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] predictions_romandy.csv  ({len(romandy_preds)} Romandy stations)")

    # ── Charts ─────────────────────────────────────────────────────────────
    print()
    print("  Generating prediction charts...")
    chart_predictions_overview(romandy_preds)
    chart_lausanne_prediction(romandy_preds)
    chart_2026_projections(romandy_preds)

    # ── Write to DuckDB (write connection needed) ──────────────────────────
    con_w = duckdb.connect(database=DB_PATH, read_only=False)
    write_predictions_to_db(df, con_w)
    con_w.close()

    # ── Print Lausanne spotlight ───────────────────────────────────────────
    print()
    print("  ── LAUSANNE SPOTLIGHT (top VD stations) ──────────────────────")
    lausanne_spot = df[df["canton"] == "VD"].sort_values("annual_adt_2025", ascending=False).head(5)
    for _, row in lausanne_spot.iterrows():
        print(
            f"  {row['station_name'][:30]:30s} | "
            f"2025 annual: {int(row['annual_adt_2025']):>7,} | "
            f"Pred Oct: {int(row.get('pred_oct', 0)):>7,} | "
            f"Pred Nov: {int(row.get('pred_nov', 0)):>7,} | "
            f"Pred Dec: {int(row.get('pred_dec', 0)):>7,}"
        )

    print()
    print("  STAGE 7 COMPLETE. Full pipeline run finished!")
    print()
    print("  OUTPUTS:")
    print(f"    reports/historical_report.html    — Historical analysis dashboard")
    print(f"    reports/model_report.html          — ML model evaluation report")
    print(f"    reports/predictions_romandy.csv    — All Romandy predictions")
    print(f"    reports/predictions_chart.png      — Overview bar chart")
    print(f"    reports/predictions_lausanne.png   — Lausanne actual + predicted")
    print(f"    reports/predictions_2026_*.png     — 2026 projections")
    print(f"    models/best_model.pkl              — Saved trained model")
    print(f"    mlruns/                            — MLflow experiment tracker")
    print(f"    traffic.duckdb / mart.predictions  — Query predictions with SQL")
    print()
    print("  To view MLflow UI:  mlflow ui --backend-store-uri ./mlruns")
    print("=" * 65)


if __name__ == "__main__":
    main()
