"""
@bruin
name: ml.evaluate_model
type: python

description: >
  Evaluate the trained best_model.pkl on the Oct–Dec holdout set.
  Produces residual plots, actual-vs-predicted charts, and a model report HTML.
  All results are saved to reports/.

depends:
  - ml.train_model

tags:
  - ml
  - evaluation
  - reporting
@end

=============================================================================
WHAT IS MODEL EVALUATION?
=============================================================================

Training a model is like teaching a student. Evaluation is the EXAM.

The "holdout" set is data the model has NEVER seen during training.
In our case:
  • Training: months Jan–Sep for all stations
  • Holdout:  months Oct–Dec for all stations

The key principle: if the model scores well on holdout data, we trust it will
score well on genuinely new data (future months, future years).
If it only scores well on training data, it has "memorised" the answers —
this is called OVERFITTING, and means predictions will be unreliable.

KEY EVALUATION CONCEPTS:
  • Residuals        = actual − predicted. If residuals are random, model is unbiased.
                       If residuals cluster (e.g., always under-predicts high-traffic stations),
                       the model has systematic bias that needs fixing.
  • Actual vs Predicted plot = perfect model = all points on the diagonal y=x line.
  • Feature importance = which input variables matter most?
  • MAPE = Mean Absolute Percentage Error = MAE / mean(actual) × 100.
           Allows comparing error across stations of different sizes.
           A 5% MAPE on a 97,000 ADT station = 4,850 vehicles/day error.

=============================================================================
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from jinja2 import Template
from datetime import datetime

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

TARGET_COLS  = ["adt_oct", "adt_nov", "adt_dec"]
TARGET_NAMES = {"adt_oct": "October", "adt_nov": "November", "adt_dec": "December"}

# =============================================================================
# QUALITY GATE THRESHOLDS
# =============================================================================
#
# A "quality gate" is a minimum performance standard the model must meet
# before its predictions are trusted for use in transport planning decisions.
#
# ANALOGY: A factory quality inspector rejects any product below a minimum
# standard. Here, if the model's R² or MAPE fail these thresholds, we flag
# the model as "not approved" and recommend retraining.
#
# These thresholds are based on FEDRO/ASTRA transport planning guidelines and
# common practice in Swiss cantonal road-traffic demand modelling:
#   R² ≥ 0.80  → "strong" explanatory power; model captures main variance
#   MAPE ≤ 15% → average prediction within 15% of actual traffic volume
#   Per-canton MAPE ≤ 25% → no region is catastrophically mis-modelled
#   Max single-station error ≤ 50% → no completely rogue predictions
#
# To change thresholds for your organisation, edit these four constants only.

MIN_R2          = 0.80    # R² must be AT LEAST this (0=random, 1=perfect)
MAX_MAPE_PCT    = 15.0    # Overall MAPE must be BELOW this percentage
MAX_CANTON_MAPE = 25.0    # No canton's MAPE may exceed this percentage
MAX_STATION_ERR = 50.0    # No individual station MAPE may exceed this percentage


# =============================================================================
# HELPER
# =============================================================================

def save_fig(fig, filename):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {filename}")
    return path


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    Skip samples where y_true is 0 (would divide by zero).

    FORMULA:
        MAPE = (1/n) × Σ |y_true_i - y_pred_i| / y_true_i × 100

    INTERPRETATION:
        5% = On average wrong by 5% of the actual value.
        Good transport models aim for < 10%.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =============================================================================
# CHART A: Actual vs Predicted Scatter (one per target month)
# =============================================================================

def chart_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str,
    station_names: list,
) -> str:
    """
    Scatter plot: x=predicted, y=actual.

    A perfect model would have ALL points exactly on the diagonal y=x line.
    Points ABOVE the diagonal = model under-predicted (actual > predicted).
    Points BELOW the diagonal = model over-predicted (actual < predicted).

    WHY THIS PLOT MATTERS:
    It instantly reveals:
    1. Overall accuracy (how tight is the point cluster?)
    2. Bias (are points systematically above or below the line?)
    3. Outliers (which specific stations are far from the line?)
    """
    month_name = TARGET_NAMES.get(target_col, target_col)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Scatter the actual vs predicted values
    ax.scatter(y_pred, y_true, alpha=0.65, s=35, color="#0057B7", edgecolors="none")

    # Perfect prediction line (y = x diagonal)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction (y = x)")

    # Label the 3 most mis-predicted stations so the analyst can investigate
    errors = np.abs(y_true - y_pred)
    top3_idx = np.argsort(errors)[-3:]
    for i in top3_idx:
        ax.annotate(
            station_names[i][:20],  # Truncate long names
            xy=(y_pred[i], y_true[i]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color="#c0392b",
        )

    ax.set_xlabel(f"Predicted ADT — {month_name}", fontsize=11)
    ax.set_ylabel(f"Actual ADT — {month_name}", fontsize=11)
    ax.set_title(f"Actual vs Predicted Traffic — {month_name} 2025",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=9)

    fname = f"eval_actual_vs_predicted_{target_col}.png"
    return save_fig(fig, fname)


# =============================================================================
# CHART B: Residual Distribution Histogram
# =============================================================================

def chart_residuals(y_true: np.ndarray, y_pred: np.ndarray, target_col: str) -> str:
    """
    Histogram of residuals (actual − predicted).

    A GOOD model produces NORMALLY DISTRIBUTED residuals centred at 0.
    This means errors are RANDOM — sometimes high, sometimes low, no pattern.

    A BAD model produces:
    - Skewed histogram → systematic over- or under-prediction
    - Fat tails → occasional catastrophic errors
    - Multi-modal → model behaves differently for different station types

    For a transport model, residuals should ideally be within ±10% of mean ADT.
    """
    residuals = y_true - y_pred
    month_name = TARGET_NAMES.get(target_col, target_col)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Histogram of residuals ───────────────────────────────────────
    ax = axes[0]
    ax.hist(residuals, bins=20, color="#0057B7", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error line")
    ax.axvline(residuals.mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean residual = {residuals.mean():.0f}")
    ax.set_xlabel("Residual (Actual − Predicted) vehicles/day", fontsize=10)
    ax.set_ylabel("Count of stations", fontsize=10)
    ax.set_title(f"Residual Distribution — {month_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── Right: Residuals vs Predicted (check for heteroscedasticity) ───────
    # Heteroscedasticity = error size grows with predicted value.
    # If this plot shows a "fan" shape spreading wider to the right,
    # the model is less reliable for high-traffic stations.
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30, color="#E87722", edgecolors="none")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel(f"Predicted ADT ({month_name})", fontsize=10)
    ax2.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
    ax2.set_title("Residuals vs Predicted (check: is error random?)",
                  fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    fname = f"eval_residuals_{target_col}.png"
    return save_fig(fig, fname)


# =============================================================================
# CHART C: Error by Canton — which canton does the model predict best?
# =============================================================================

def chart_error_by_canton(df_eval: pd.DataFrame, target_col: str) -> str:
    """
    Bar chart showing Mean Absolute Percentage Error (MAPE) per canton.

    If certain cantons have systematically higher error, it suggests:
    - Those cantons have unusual traffic patterns not well captured by features
    - More canton-specific features (events, tourism) might help
    """
    month_name = TARGET_NAMES.get(target_col, target_col)

    canton_stats = (
        df_eval.groupby("canton")
        .apply(lambda g: pd.Series({
            "mape": mape(g["actual"].values, g["predicted"].values),
            "n": len(g),
        }))
        .reset_index()
        .sort_values("mape")
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    colours = ["#2ecc71" if m < 10 else "#e67e22" if m < 20 else "#e74c3c"
               for m in canton_stats["mape"]]

    bars = ax.bar(canton_stats["canton"], canton_stats["mape"], color=colours, width=0.6)

    # Add value labels on top of each bar
    for bar, val in zip(bars, canton_stats["mape"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(10, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.6,
               label="10% threshold (good)")
    ax.set_xlabel("Canton", fontsize=11)
    ax.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=11)
    ax.set_title(f"Prediction Error by Canton — {month_name} 2025\n"
                 "Green < 10% · Orange 10–20% · Red > 20%",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fname = f"eval_canton_error_{target_col}.png"
    return save_fig(fig, fname)


# =============================================================================
# QUALITY GATE CHECK
# =============================================================================

def check_quality_gates(metrics: dict, per_target_eval: dict) -> dict:
    """
    Run all quality gate checks and return a structured pass/fail report.

    WHAT IS A "QUALITY GATE"?
    Think of it as a checklist a pilot must complete before being allowed to
    take off. Every box must be ticked or the plane stays grounded.

    Our gates (all must pass per target month):
      1. R² gate          — Is the model explaining most of the variance?
      2. MAPE gate        — Is the overall percentage error acceptable?
      3. Worst-canton gate — Does any specific canton have unacceptably high error?
      4. Max-station gate  — Does any single station have a catastrophically bad prediction?

    WHY CANTON-LEVEL CHECKING MATTERS:
    A model that works perfectly for Lausanne but fails for Fribourg stations
    would still show a good OVERALL MAPE. The canton-level gate catches this.
    A model approved for deployment should be reliable across ALL Romandy cantons,
    not just the high-volume ones that dominate the average.

    RETURNS a dict:
    {
        "overall_pass": True/False,
        "generated_at": "ISO timestamp",
        "thresholds": {...},
        "gates": {
            "adt_oct": {
                "r2":           {"value": 0.91, "threshold": 0.80, "pass": True},
                "mape":         {"value": 8.2,  "threshold": 15.0, "pass": True},
                "worst_canton": {"canton": "JU", "mape": 18.5, "threshold": 25.0, "pass": True},
                "worst_station":{"station": "...", "mape": 41.2, "threshold": 50.0, "pass": True},
                "target_pass":  True,
            },
            ...
        }
    }
    """
    gates    = {}
    all_pass = True

    for target_col in TARGET_COLS:
        if target_col not in metrics or target_col not in per_target_eval:
            continue

        m      = metrics[target_col]
        ev     = per_target_eval[target_col]
        y_true = ev["y_true"]
        y_pred = ev["y_pred"]
        df_ev  = ev["eval_df"]

        # ── Gate 1: R² ────────────────────────────────────────────────
        r2_val  = m["r2"]
        r2_pass = float(r2_val) >= MIN_R2

        # ── Gate 2: Overall MAPE ──────────────────────────────────────
        mape_val  = mape(y_true, y_pred)
        mape_pass = mape_val <= MAX_MAPE_PCT

        # ── Gate 3: Worst-canton MAPE ─────────────────────────────────
        canton_mapes = (
            df_ev.groupby("canton")
            .apply(lambda g: mape(g["actual"].values, g["predicted"].values))
            .dropna()
        )
        if len(canton_mapes):
            worst_canton      = canton_mapes.idxmax()
            worst_canton_mape = float(canton_mapes.max())
        else:
            worst_canton      = "N/A"
            worst_canton_mape = 0.0
        canton_pass = worst_canton_mape <= MAX_CANTON_MAPE

        # ── Gate 4: Worst single-station MAPE ────────────────────────
        station_mapes = np.where(
            y_true != 0,
            np.abs((y_true - y_pred) / y_true) * 100,
            np.nan,
        )
        # Ignore NaN
        valid = ~np.isnan(station_mapes)
        if valid.sum():
            worst_idx      = int(np.nanargmax(station_mapes))
            worst_station  = df_ev["station"].iloc[worst_idx]
            worst_st_mape  = float(station_mapes[worst_idx])
        else:
            worst_station = "N/A"
            worst_st_mape = 0.0
        station_pass = worst_st_mape <= MAX_STATION_ERR

        target_pass = r2_pass and mape_pass and canton_pass and station_pass
        all_pass    = all_pass and target_pass

        gates[target_col] = {
            "r2":   {"value": round(float(r2_val), 4), "threshold": MIN_R2,     "pass": r2_pass},
            "mape": {"value": round(mape_val, 2),       "threshold": MAX_MAPE_PCT, "pass": mape_pass},
            "worst_canton": {
                "canton":    worst_canton,
                "mape":      round(worst_canton_mape, 2),
                "threshold": MAX_CANTON_MAPE,
                "pass":      canton_pass,
            },
            "worst_station": {
                "station":   worst_station,
                "mape":      round(worst_st_mape, 2),
                "threshold": MAX_STATION_ERR,
                "pass":      station_pass,
            },
            "target_pass": target_pass,
        }

    report = {
        "overall_pass": all_pass,
        "generated_at": datetime.now().isoformat(),
        "thresholds": {
            "min_r2":          MIN_R2,
            "max_mape_pct":    MAX_MAPE_PCT,
            "max_canton_mape": MAX_CANTON_MAPE,
            "max_station_err": MAX_STATION_ERR,
        },
        "gates": gates,
    }
    return report


# =============================================================================
# MODEL REPORT HTML
# =============================================================================

MODEL_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ML Model Evaluation Report — Swiss Traffic 2025</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa;
           color: #222; line-height: 1.6; }
    header { background: linear-gradient(135deg, #1a3a52 0%, #2980b9 100%);
             color: white; padding: 2rem 3rem; }
    header h1 { font-size: 1.8rem; font-weight: 700; }
    header p  { opacity: 0.85; margin-top: 0.4rem; }
    .badge { display: inline-block; background: rgba(255,255,255,0.18);
             border-radius: 4px; padding: 2px 10px; font-size: 0.8rem;
             margin-top: 0.6rem; margin-right: 6px; }
    main { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
                gap: 1rem; margin-bottom: 2rem; }
    .kpi-card { background: white; border-radius: 8px; padding: 1.2rem 1.5rem;
                border-left: 4px solid #2980b9; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .kpi-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase;
                       letter-spacing: 0.08em; }
    .kpi-card .value { font-size: 1.7rem; font-weight: 700; color: #1a3a52; margin-top: 0.2rem; }
    .kpi-card .sub   { font-size: 0.78rem; color: #888; margin-top: 0.1rem; }
    .section { background: white; border-radius: 8px; padding: 1.8rem 2rem;
               margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .section h2 { font-size: 1.2rem; font-weight: 600; color: #1a3a52;
                  border-bottom: 2px solid #e8edf5; padding-bottom: 0.6rem;
                  margin-bottom: 1rem; }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    img { width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    .cap { font-size: 0.8rem; color: #777; text-align: center; margin-top: 0.4rem;
           font-style: italic; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th { background: #1a3a52; color: white; padding: 0.6rem 0.8rem; text-align: left; }
    td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #e8edf5; }
    tr:hover td { background: #f0f4fb; }
    .good  { color: #27ae60; font-weight: 700; }
    .warn  { color: #e67e22; font-weight: 700; }
    .bad   { color: #e74c3c; font-weight: 700; }
    footer { text-align: center; padding: 2rem; color: #999; font-size: 0.82rem; }
  </style>
</head>
<body>
  <header>
    <h1>🤖 ML Model Evaluation Report — Swiss Traffic Prediction 2025</h1>
    <p>Holdout evaluation: Oct–Dec 2025 predictions vs actual FEDRO measurements</p>
    <span class="badge">Generated: {{ generated_at }}</span>
    <span class="badge">Dataset: FEDRO/ASTRA 2025</span>
    <span class="badge">Holdout: Oct · Nov · Dec</span>
  </header>
  <main>

    <div class="kpi-grid">
      {% for target, m in metrics.items() %}
      <div class="kpi-card">
        <div class="label">{{ target }} — {{ m.model }}</div>
        <div class="value {{ 'good' if m.r2 > 0.85 else 'warn' if m.r2 > 0.7 else 'bad' }}">
          R² = {{ "%.3f"|format(m.r2) }}
        </div>
        <div class="sub">MAE = {{ "{:,.0f}".format(m.mae) }} veh/day</div>
      </div>
      {% endfor %}
    </div>

    <!-- Metrics table -->
    <div class="section">
      <h2>📊 Evaluation Metrics Summary</h2>
      <p>
        <strong>R²</strong> (Coefficient of Determination): 1.0 = perfect, 0.0 = predicting the mean only.
        Good transport models typically achieve R² > 0.85. <br>
        <strong>MAE</strong>: Average absolute error in vehicles/day — the most interpretable metric. <br>
        <strong>RMSE</strong>: Like MAE but penalises large errors more heavily. <br>
        <strong>MAPE</strong>: Error as a percentage of the actual value — useful for comparing across station sizes.
      </p>
      <table>
        <thead><tr>
          <th>Target Month</th><th>Best Model</th><th>MAE (veh/day)</th>
          <th>RMSE (veh/day)</th><th>R²</th><th>Rating</th>
        </tr></thead>
        <tbody>
        {% for target, m in metrics.items() %}
        <tr>
          <td>{{ target }}</td>
          <td>{{ m.model }}</td>
          <td>{{ "{:,.0f}".format(m.mae) }}</td>
          <td>{{ "{:,.0f}".format(m.rmse) }}</td>
          <td class="{{ 'good' if m.r2 > 0.85 else 'warn' if m.r2 > 0.7 else 'bad' }}">
            {{ "%.4f"|format(m.r2) }}
          </td>
          <td>{{ '✅ Good' if m.r2 > 0.85 else '⚠️ Moderate' if m.r2 > 0.7 else '❌ Poor' }}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

    <!-- Charts -->
    {% for target in ['adt_oct', 'adt_nov', 'adt_dec'] %}
    {% if charts.get(target) %}
    <div class="section">
      <h2>📈 {{ target | upper }} — Detailed Evaluation</h2>
      <div class="chart-grid">
        {% if charts[target].get('avp') %}
        <div>
          <img src="{{ charts[target]['avp'] }}" alt="Actual vs Predicted">
          <p class="cap">Actual vs Predicted scatter — points on diagonal = perfect</p>
        </div>
        {% endif %}
        {% if charts[target].get('res') %}
        <div>
          <img src="{{ charts[target]['res'] }}" alt="Residuals">
          <p class="cap">Residual histogram — centred at 0 = unbiased model</p>
        </div>
        {% endif %}
      </div>
      {% if charts[target].get('canton') %}
      <img src="{{ charts[target]['canton'] }}" alt="Error by canton" style="margin-top:1rem">
      <p class="cap">MAPE by canton — reveals which regions the model struggles with</p>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}

  </main>
  <footer>Swiss Traffic MLOps Pipeline · ML Evaluation · FEDRO/ASTRA Data 2025</footer>
</body>
</html>
"""


# =============================================================================
# MAIN
# =============================================================================

def main(_retrain_attempt: bool = False):
    """
    Run the full evaluation suite.

    _retrain_attempt: internal flag — set to True when called from the
    auto-retrain loop so we don't retrain a second time infinitely.
    """
    print("=" * 65)
    print("  STAGE 6: ML EVALUATION" + (" [RETRAIN ATTEMPT]" if _retrain_attempt else ""))
    print("=" * 65)
    print()

    # ── Load model bundle ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERR] Model not found: {MODEL_PATH}")
        print("  Run train_model.py first.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    feature_cols = bundle["feature_cols"]
    metrics      = bundle.get("metrics", {})
    print(f"  Loaded model bundle from: {MODEL_PATH}")
    for target, m in metrics.items():
        print(f"    {target}: {m['model']}  R²={m['r2']:.4f}  MAE={m['mae']:.0f}")

    # ── Load features from DuckDB ──────────────────────────────────────────
    con = duckdb.connect(database=DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT station_id, station_name, canton, road, is_romandy,
               {', '.join(feature_cols)},
               adt_oct, adt_nov, adt_dec
        FROM mart.traffic_features
        ORDER BY station_id
    """).df()
    con.close()

    # Only evaluate on stations with actual Oct–Dec data (ground truth)
    eval_df = df[df[TARGET_COLS].notna().all(axis=1)].copy()
    print(f"\n  Evaluation set: {len(eval_df)} stations (all 3 targets non-null)")

    X_eval = eval_df[feature_cols].values
    chart_map      = {}
    per_target_eval = {}  # Collect raw arrays for quality gate checks

    for target_col in TARGET_COLS:
        if target_col not in bundle:
            print(f"  [SKIP] No pipeline in bundle for {target_col}")
            continue

        pipeline = bundle[target_col]
        y_true = eval_df[target_col].values
        y_pred = pipeline.predict(X_eval)
        station_names = eval_df["station_name"].tolist()

        print(f"\n  ── {target_col} ─────────────────────────────────────────────")
        mae_val  = np.mean(np.abs(y_true - y_pred))
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2_val   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        mape_val = mape(y_true, y_pred)

        print(f"    MAE  = {mae_val:9.0f} vehicles/day")
        print(f"    RMSE = {rmse_val:9.0f} vehicles/day")
        print(f"    R²   = {r2_val:9.4f}")
        print(f"    MAPE = {mape_val:9.2f} %")

        # Build per-row eval dataframe for canton analysis
        df_eval = pd.DataFrame({
            "canton":    eval_df["canton"].values,
            "station":   eval_df["station_name"].values,
            "actual":    y_true,
            "predicted": y_pred,
        })

        # Store for quality gate checks below
        per_target_eval[target_col] = {
            "y_true":  y_true,
            "y_pred":  y_pred,
            "eval_df": df_eval,
        }

        avp_path    = chart_actual_vs_predicted(y_true, y_pred, target_col, station_names)
        res_path    = chart_residuals(y_true, y_pred, target_col)
        canton_path = chart_error_by_canton(df_eval, target_col)

        chart_map[target_col] = {
            "avp":    os.path.basename(avp_path) if avp_path else "",
            "res":    os.path.basename(res_path) if res_path else "",
            "canton": os.path.basename(canton_path) if canton_path else "",
        }

    # ── Quality Gate Evaluation ────────────────────────────────────────────
    #
    # Now that we have actual predictions, run the formal quality gate checks.
    # This is like a final school exam — the model must pass ALL gates to be
    # considered production-ready.
    #
    # Results are:
    #   1. Printed to the console with a clear PASS / FAIL banner
    #   2. Saved to reports/quality_gates.json for audit trails
    #   3. If FAIL AND this is the first attempt → auto-retrain is triggered
    #
    # AUTO-RETRAIN LOGIC:
    # When the model fails a gate, the most likely fixes are:
    #   (a) Different hyperparameters  — RandomizedSearchCV will explore more
    #   (b) More training data         — not available here, but possible in prod
    # We cap auto-retraining at ONE attempt (retrain_attempt flag) to avoid
    # an infinite loop in case the data is inherently too noisy to pass gates.
    print()
    print("  ── QUALITY GATE EVALUATION ────────────────────────────────────")
    gate_report = check_quality_gates(metrics, per_target_eval)

    # Print per-target gate results
    for target_col, gates in gate_report["gates"].items():
        g = gates
        status = "✅ PASS" if g["target_pass"] else "❌ FAIL"
        print(f"  {target_col}  {status}")
        r2g = g["r2"]
        print(f"    R²   gate : {r2g['value']:.4f}  (need ≥{r2g['threshold']})  "
              f"{'✅' if r2g['pass'] else '❌'}")
        mg = g["mape"]
        print(f"    MAPE gate : {mg['value']:.2f}%  (need ≤{mg['threshold']}%)  "
              f"{'✅' if mg['pass'] else '❌'}")
        cg = g["worst_canton"]
        print(f"    Canton gate: worst={cg['canton']} {cg['mape']:.1f}%  "
              f"(need ≤{cg['threshold']}%)  {'✅' if cg['pass'] else '❌'}")
        sg = g["worst_station"]
        print(f"    Station gate: worst={sg['station'][:25]} {sg['mape']:.1f}%  "
              f"(need ≤{sg['threshold']}%)  {'✅' if sg['pass'] else '❌'}")

    # Overall banner
    print()
    if gate_report["overall_pass"]:
        print("  ╔══════════════════════════════════════╗")
        print("  ║   ✅  ALL QUALITY GATES PASSED        ║")
        print("  ║   Model approved for predictions.    ║")
        print("  ╚══════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════╗")
        print("  ║   ❌  QUALITY GATES FAILED            ║")
        print("  ║   Model NOT approved for deployment. ║")
        print("  ╚══════════════════════════════════════╝")

    # Save gate report to JSON (for CI/CD systems, audit logs, dashboards)
    gate_json_path = os.path.join(REPORTS_DIR, "quality_gates.json")
    with open(gate_json_path, "w", encoding="utf-8") as f:
        # Convert numpy bools/floats to plain Python types for JSON serialisation
        import json
        json.dump(gate_report, f, indent=2, default=str)
    print(f"  [SAVED] reports/quality_gates.json")

    # ── Auto-retrain if gates failed ──────────────────────────────────────
    #
    # HOW AUTO-RETRAIN WORKS:
    # 1. We import train_model.main() (Python import, not subprocess)
    # 2. Call it — this re-runs the full training pipeline:
    #     • Loads the same data from DuckDB
    #     • Runs RandomizedSearchCV again (random seed 42 is fixed, but a
    #       wider N_ITER_SEARCH is used on retry to explore more)
    #     • Overwrites models/best_model.pkl with the new best
    # 3. We reload the bundle and re-run ALL evaluation steps above
    # 4. The new gate report replaces quality_gates.json
    # 5. If STILL failing: we report the failure and exit — do NOT loop again
    #    (the _retrain_attempt parameter prevents a second auto-retrain)
    #
    # WHY ONLY ONE AUTO-RETRAIN?
    # Two reasons:
    #   (a) If the model fails after tuning, the issue may be fundamental
    #       (too little data, too much noise) — more retraining won't help.
    #   (b) Prevent runaway pipeline loops in unattended CI/CD runs.
    #
    # In production, persistent gate failures should trigger a human alert:
    # a Slack message, a Jira ticket, or a PagerDuty incident.

    if not gate_report["overall_pass"] and not _retrain_attempt:
        print()
        print("  [AUTO-RETRAIN] Quality gates failed. Triggering retraining ...")
        print("  [AUTO-RETRAIN] train_model.py will run with wider hyperparameter")
        print("  [AUTO-RETRAIN] search (N_ITER_SEARCH × 2) before re-evaluating.")
        print()

        # Dynamically import to avoid circular dependency issues
        import importlib.util
        train_spec = importlib.util.spec_from_file_location(
            "train_model",
            os.path.join(THIS_DIR, "train_model.py"),
        )
        train_mod = importlib.util.module_from_spec(train_spec)
        # Temporarily double the search budget for the retry attempt
        train_mod.N_ITER_SEARCH = 40   # 2× normal search
        train_spec.loader.exec_module(train_mod)
        train_mod.main()               # Overwrites best_model.pkl

        print()
        print("  [AUTO-RETRAIN] Retraining complete. Re-evaluating ...")
        print()
        main(_retrain_attempt=True)    # Evaluate the new model; won't retrain again
        return                         # Exit this (original) call cleanly

    elif not gate_report["overall_pass"] and _retrain_attempt:
        print()
        print("  [WARN] Model still fails quality gates after auto-retrain.")
        print("  [WARN] Manual investigation required:")
        print("    1. Check reports/quality_gates.json for failing gates.")
        print("    2. Inspect reports/eval_*.png charts for systematic bias.")
        print("    3. Consider adding more training features or collecting more data.")
        print("    4. Adjust MIN_R2 / MAX_MAPE_PCT thresholds if data is inherently noisy.")
        # Do NOT sys.exit() — we still produce the HTML report for diagnostics

    # ── Render model report HTML ───────────────────────────────────────────
    report_path = os.path.join(REPORTS_DIR, "model_report.html")
    template = Template(MODEL_REPORT_TEMPLATE)
    html = template.render(
        generated_at=datetime.now().strftime("%d %B %Y %H:%M"),
        metrics=metrics,
        charts=chart_map,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [SAVED] reports/model_report.html")

    print()
    print("  STAGE 6 COMPLETE. Next: bruin run assets/ml/predict_traffic.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
