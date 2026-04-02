"""
@bruin
name: ml.train_model
type: python

description: >
  Train three ML models (Linear Regression, Random Forest, Gradient Boosting)
  on Jan–Sep traffic data to predict Oct–Dec ADT for all Swiss measuring stations.
  Logs all experiments to MLflow. Saves the best model to models/best_model.pkl.

depends:
  - mart.traffic_features

tags:
  - ml
  - training
  - mlflow
@end

=============================================================================
WHAT IS THIS SCRIPT? (THE 10-YEAR-OLD EXPLANATION)
=============================================================================

Imagine you are studying for a maths test. You look at 9 months of past tests
(January through September) and practise answering them. Then you sit the real
test — October, November, December — and see how well you do.

This script does exactly that, but for traffic prediction:
  INPUT  (features X): Monthly traffic data Jan–Sept for every station
  OUTPUT (target    y): Monthly traffic data Oct–Dec for every station

It trains 3 different "students" (models) and picks the best one by comparing
their test scores.

=============================================================================
THE MACHINE LEARNING APPROACH IN DETAIL
=============================================================================

CROSS-SECTIONAL STRATEGY (not time-series):
We only have ONE year of data (2025). Traditional time-series models like
ARIMA or LSTM need at least 3–5 years of data per station. We have ~200
stations × 12 months each.

Solution: treat every station-month as an independent observation.
  • Training data: all stations, months 1–9  (~200 × 9 = 1,800 rows after imputation)
  • Test data:     all stations, months 10–12 (~200 × 3 = 600 rows)

The model learns: "A station with this Jan–Sep pattern will have this Oct–Dec level."
This uses correlations ACROSS stations, not time-series correlations within one station.

THREE MODELS COMPARED:
  1. LinearRegression — The baseline. Assumes a straight-line relationship between
     features and target. Fast to train. Interpretable ("each extra 1000 ADT in
     July predicts X more ADT in October"). Not great with non-linear patterns.

  2. RandomForestRegressor — An ensemble of 200 decision trees. Each tree learns
     different patterns; their average prediction is more accurate than any single tree.
     Handles non-linear relationships and missing values well. Like asking 200 experts
     to vote on the answer.

  3. GradientBoostingRegressor — Builds trees SEQUENTIALLY: each new tree corrects
     the errors of all previous trees. Often the most accurate model for tabular data.
     Slower to train but achieves the lowest error in most competitions.

METRICS USED:
  • MAE  = Mean Absolute Error. Average |predicted − actual|. Easy to interpret:
           "On average we're off by X vehicles/day."
  • RMSE = Root Mean Squared Error. Like MAE but punishes large errors more.
           Good for catching catastrophic predictions.
  • R²   = Coefficient of Determination. 1.0 = perfect. 0.0 = as good as guessing
           the mean. Range: 0 to 1. Aim for > 0.85 for transport modelling.

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")   # Suppress sklearn convergence warnings in output

import numpy as np
import pandas as pd
import duckdb
import joblib       # joblib saves Python objects (our trained model) to a .pkl file
                    # .pkl = "pickle" file — a binary serialisation of Python objects.
                    # Like taking a photograph of your trained model so you can
                    # reload it later without retraining.

# scikit-learn — the standard Python ML library
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
                    # StandardScaler transforms features so they all have
                    # mean=0 and standard deviation=1.
                    # WHY? LinearRegression is sensitive to feature scale.
                    # Without scaling, adt_jan (~80,000) would dominate canton_code (~1–18).
                    # Tree-based models (RF, GBM) don't need scaling, but it doesn't hurt.

from sklearn.pipeline import Pipeline
                    # A Pipeline chains preprocessing + model into one object.
                    # Benefit: when you save the pipeline, it saves the scaler too.
                    # This prevents "training-serving skew" — using raw values at
                    # prediction time when the model was trained on scaled values.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
                    # SimpleImputer fills in missing values (NaN).
                    # We use strategy="median" — replace missing with the median
                    # of that column. Median is robust to outliers; mean is not.

from sklearn.model_selection import RandomizedSearchCV, KFold
                    # RandomizedSearchCV: instead of trying EVERY combination of
                    # hyperparameters (GridSearchCV), it randomly samples N combinations.
                    # Much faster when the search space is large.
                    # Example: 4 × 5 × 3 × 3 = 180 RF combos → we sample 20 at random.
                    #
                    # KFold: splits training data into K equal chunks.
                    # Train on K-1 chunks, validate on the remaining 1 chunk.
                    # Repeat K times (each chunk gets to be the validation set once).
                    # Final score = average of K validation scores.
                    # This gives a more reliable estimate than a single train/val split.

import mlflow
import mlflow.sklearn
                    # mlflow tracks every training experiment automatically.
                    # It logs: parameters, metrics, model artifacts.
                    # You can view results at: http://localhost:5000  (after `mlflow ui`)

# =============================================================================
# CONFIGURATION
# =============================================================================

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(THIS_DIR, "..", "..")
DB_PATH      = os.path.join(PROJECT_ROOT, "traffic.duckdb")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# MLflow experiment name — groups all our training runs together in the UI.
# Each call to train_and_log() creates a new "run" within this experiment.
MLFLOW_EXPERIMENT = "swiss-traffic-adt-prediction"

# ── Feature columns (X) ────────────────────────────────────────────────────
# These are the input columns the model will be trained on.
# They come from mart.traffic_features.
RAW_MONTH_FEATURES = [
    "adt_jan", "adt_feb", "adt_mar", "adt_apr", "adt_may",
    "adt_jun", "adt_jul", "adt_aug", "adt_sep",
]
ENGINEERED_FEATURES = [
    "summer_peak_ratio",
    "weekday_weekend_ratio",
    "hgv_pct_jul",
    "winter_depression_ratio",
    "mean_adt_jan_sep",
    "canton_code",
    "road_type_code",
    "is_romandy_int",
]
ALL_FEATURES = RAW_MONTH_FEATURES + ENGINEERED_FEATURES

# ── Target columns (y) ─────────────────────────────────────────────────────
# We train a SEPARATE model for each target month.
# Multi-output regression (one model for all 3 targets) is also possible,
# but separate models allow per-month hyperparameter tuning.
TARGET_COLS = ["adt_oct", "adt_nov", "adt_dec"]

# ── Model definitions ───────────────────────────────────────────────────────
# Each model is defined as a dict with:
#   name      : human-readable name for logging
#   estimator : the scikit-learn model class
#   params    : hyperparameters
#   needs_scale: whether to apply StandardScaler before training
#
# HYPERPARAMETERS EXPLAINED:
# n_estimators = how many trees to build (more = better but slower)
# max_depth    = how deep each tree can grow (deeper = more complex patterns,
#                but risk of overfitting = memorising training data too well)
# random_state = fixed random seed: ensures reproducible results every run
MODEL_DEFINITIONS = [
    {
        "name": "LinearRegression",
        "estimator": LinearRegression(),
        "params": {},
        "needs_scale": True,     # Linear models require feature scaling
    },
    {
        "name": "RandomForest",
        "estimator": RandomForestRegressor(
            n_estimators=200,   # 200 decision trees in the forest
            max_depth=8,        # Each tree can split at most 8 times
            min_samples_leaf=3, # Each leaf must have at least 3 training samples
                                # (prevents overfitting on tiny datasets like ours)
            random_state=42,    # 42 is the conventional "random seed" in ML
        ),
        "params": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 3},
        "needs_scale": False,   # Tree models don't need scaling
    },
    {
        "name": "GradientBoosting",
        "estimator": GradientBoostingRegressor(
            n_estimators=150,   # 150 sequential boosting rounds
            learning_rate=0.08, # Step size: small = more conservative, less overfitting
                                # Think of it as: how much attention to pay to each mistake
            max_depth=4,        # Shallow trees work best for boosting (weak learners)
            subsample=0.8,      # Use 80% of training rows per round (like dropout in deep learning)
                                # Adds randomness = better generalisation
            random_state=42,
        ),
        "params": {
            "n_estimators": 150, "learning_rate": 0.08,
            "max_depth": 4, "subsample": 0.8,
        },
        "needs_scale": False,
    },
]

# Train/test split month boundary.
# We train on months 1–9 (Jan–Sep) and test on months 10–12 (Oct–Dec).
# For stations where Oct/Nov/Dec values are NULL (winter closures), we skip
# them from the test set but still use them in training.
TRAIN_MONTHS_END  = 9
TEST_MONTHS_START = 10

# =============================================================================
# HYPERPARAMETER SEARCH GRIDS
# =============================================================================
#
# For RandomForest and GradientBoosting we define a SEARCH SPACE — a dict of
# hyperparameter names → lists of candidate values to try.
#
# The keys use sklearn Pipeline naming convention:
#   "model__n_estimators" means: the "n_estimators" param of the step named "model"
#   (the last step of our Pipeline is always named "model").
#
# Only RF and GBM are tuned. LinearRegression has no hyperparameters to tune.

PARAM_GRIDS = {
    "RandomForest": {
        # Number of trees: more trees = more stable predictions but slower training.
        # We cap at 300; beyond that gains are marginal on our ~180-row dataset.
        "model__n_estimators":     [100, 150, 200, 300],
        # Max depth: restricts how complex each tree can get.
        # None = grow until leaves are pure (risks overfitting on small data).
        "model__max_depth":        [5, 6, 8, 10, None],
        # Min samples per leaf: prevents trees from learning patterns from < N stations.
        # Larger = more generalisation, smaller = more expressiveness.
        "model__min_samples_leaf": [2, 3, 5],
        # max_features: fraction of features considered at each split.
        # "sqrt" = standard RF default; 0.5 and 0.7 = more features per split.
        "model__max_features":     ["sqrt", 0.5, 0.7],
    },
    "GradientBoosting": {
        # More rounds = potentially better fit, but risk of overfitting.
        "model__n_estimators":     [100, 150, 200],
        # Learning rate: step size each round takes toward reducing error.
        # Lower rate needs more rounds but often generalises better.
        "model__learning_rate":    [0.05, 0.08, 0.10, 0.15],
        # Depth of each weak learner (individual tree). 3–5 is typical for boosting.
        "model__max_depth":        [3, 4, 5],
        # Subsample: fraction of training rows used in each boosting round.
        # <1.0 adds randomness → helps prevent overfitting.
        "model__subsample":        [0.70, 0.80, 0.90],
        "model__min_samples_leaf": [2, 3, 5],
    },
}

# How many random hyperparameter combinations to try per model.
# 20 iterations × 5-fold CV = 100 total model fits per target per model type.
# This is fast enough (~30s) on our small dataset.
N_ITER_SEARCH = 20
CV_FOLDS      = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load the ML feature matrix from mart.traffic_features.

    Returns a DataFrame where:
    - Each ROW   = one measuring station
    - Each COLUMN = one feature (Jan–Sep ADT) or target (Oct–Dec ADT)

    IMPORTANT: Some stations have NULL values for some months (closures, faults).
    We keep them in the dataset and let SimpleImputer handle them.
    Dropping all rows with any NULL would discard ~30% of usable stations.
    """
    df = con.execute(f"""
        SELECT
            station_id,
            station_name,
            canton,
            road,
            road_category,
            is_romandy,
            {', '.join(ALL_FEATURES)},
            adt_oct, adt_nov, adt_dec,
            months_with_data
        FROM mart.traffic_features
        ORDER BY station_id
    """).df()

    print(f"  Loaded {len(df)} stations from mart.traffic_features")
    print(f"  Feature columns: {len(ALL_FEATURES)}")
    print(f"  Training data (has ≥7 months Jan-Sep): {len(df)} rows")

    # Report data completeness
    complete = df[ALL_FEATURES + TARGET_COLS].notna().all(axis=1).sum()
    print(f"  Fully complete rows (no NULLs at all): {complete}")

    return df


# =============================================================================
# TRAIN + LOG ONE MODEL
# =============================================================================

def train_and_log(
    model_def: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_col: str,
    imputer: SimpleImputer,
) -> dict:
    """
    Train one model for one target column, log everything to MLflow,
    return the trained pipeline and its test metrics.

    MLFLOW CONCEPTS:
    • Experiment = a named group of runs. We use "swiss-traffic-adt-prediction".
    • Run        = one specific training attempt (one model × one target × one set of params).
    • Params     = hyperparameters logged (n_estimators, learning_rate, etc.)
    • Metrics    = evaluation scores (MAE, RMSE, R²)
    • Artifact   = files saved alongside the run (the model .pkl file, feature importances)

    PIPELINE STRUCTURE:
    The sklearn Pipeline links two steps:
      Step 1: imputer   — fill missing values with median
      Step 2: estimator — the actual regression model

    WHY PIPELINE AND NOT SEPARATE STEPS?
    When you call pipeline.predict(X_new), it automatically:
      1. Runs the imputer on X_new
      2. Feeds the result to the model
    This guarantees the same preprocessing at both training AND prediction time.

    PARAMETERS:
        model_def  : dict with 'name', 'estimator', 'params'
        X_train    : numpy array, shape (n_train, n_features)
        y_train    : numpy array, shape (n_train,)  — one target column
        X_test     : numpy array, shape (n_test,  n_features)
        y_test     : numpy array, shape (n_test,)
        target_col : name of the target ('adt_oct', 'adt_nov', 'adt_dec')
        imputer    : fitted SimpleImputer (fitted on train, applied to test)

    RETURNS:
        dict with keys: name, target, pipeline, mae, rmse, r2
    """
    name      = model_def["name"]
    estimator = model_def["estimator"]

    # ── Build base pipeline ───────────────────────────────────────────
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_def["needs_scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    base_pipeline = Pipeline(steps)

    # ── Hyperparameter tuning (RandomizedSearchCV) ────────────────────
    # LinearRegression has no tunable hyperparameters → skip search.
    # RF and GBM have PARAM_GRIDS defined → sample 20 random combos,
    # evaluate each with 5-fold cross-validation, pick the best.
    #
    # HOW RANDOMIZED SEARCH WORKS:
    #   1. Randomly draw a combination: e.g. {n_estimators: 200, max_depth: 8, ...}
    #   2. Split training data into 5 equal folds
    #   3. Train on 4 folds, validate on 1 fold → get MAE
    #   4. Repeat for all 5 fold rotations → average MAE = CV score
    #   5. Repeat steps 1–4 for N_ITER_SEARCH = 20 combinations
    #   6. The combination with the lowest average CV-MAE wins
    #   7. Refit the winner on the FULL training set (refit=True)
    #
    # Why not GridSearch? With 4×5×3×3 = 180 RF combos × 5 folds = 900 model
    # fits just for RandomForest. Randomized search covers the space almost
    # as well in 20×5 = 100 fits — 9× faster.

    param_grid  = PARAM_GRIDS.get(name)
    best_params = model_def["params"].copy()   # Start with defaults
    cv_mae      = None

    if param_grid:
        print(f"    [{name:20s}] Tuning: {N_ITER_SEARCH} combos × {CV_FOLDS}-fold CV ...")
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_grid,
            n_iter=N_ITER_SEARCH,
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",   # sklearn maximises → negate MAE
            n_jobs=-1,                            # Use all CPU cores
            random_state=42,
            refit=True,    # After search, refit best estimator on full X_train
            verbose=0,
        )
        search.fit(X_train, y_train)
        tuned_pipeline = search.best_estimator_  # Final model, fitted on full data
        # Strip the "model__" prefix so params read cleanly in MLflow UI
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        cv_mae = -search.best_score_             # Convert back from negative
        print(f"    [{name:20s}] Best CV-MAE={cv_mae:.0f}  Params: {best_params}")
    else:
        tuned_pipeline = None   # Will train base_pipeline inside mlflow context

    with mlflow.start_run(run_name=f"{name}__{target_col}"):
        # ── Log hyperparameters ───────────────────────────────────────
        mlflow.log_param("model_type",      name)
        mlflow.log_param("target",          target_col)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples",  len(X_test))
        mlflow.log_param("n_features",      X_train.shape[1])
        mlflow.log_param("hypertuned",      bool(param_grid))
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        if cv_mae is not None:
            mlflow.log_metric("cv_mae", round(cv_mae, 2))

        # ── Use tuned or base pipeline ────────────────────────────────
        if tuned_pipeline is not None:
            pipeline = tuned_pipeline   # Already fitted by RandomizedSearchCV.refit
        else:
            # LinearRegression: no search, fit directly here
            base_pipeline.fit(X_train, y_train)
            pipeline = base_pipeline

        # ── Evaluate on test set ──────────────────────────────────────
        y_pred = pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # ── Log metrics ───────────────────────────────────────────────
        mlflow.log_metric("mae",  round(mae,  2))
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("r2",   round(r2,   4))

        # ── Log the trained model as an MLflow artifact ───────────────
        # This saves the pipeline to MLflow's artifact store so you can
        # reload it later with: mlflow.sklearn.load_model("runs:/<run_id>/model")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"    [{name:20s}] target={target_col} | "
              f"MAE={mae:8.0f} | RMSE={rmse:8.0f} | R²={r2:.4f}")

    return {
        "name": name,
        "target": target_col,
        "pipeline": pipeline,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  STAGE 5: ML TRAINING")
    print("=" * 65)
    print()

    # ── Connect to DuckDB ──────────────────────────────────────────────────
    if not os.path.exists(DB_PATH):
        print(f"[ERR] Database not found: {DB_PATH}")
        print("  Run ingestion + staging + transform stages first.")
        sys.exit(1)

    con = duckdb.connect(database=DB_PATH, read_only=True)
    df = load_features(con)
    con.close()

    # ── Setup MLflow ───────────────────────────────────────────────────────
    # mlruns/ will be created in the project root (same directory as this repo)
    from pathlib import Path
    mlruns_dir = os.path.join(PROJECT_ROOT, 'mlruns')
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(Path(mlruns_dir).as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"\n  MLflow tracking: {os.path.join(PROJECT_ROOT, 'mlruns')}")
    print(f"  View UI after run: mlflow ui --backend-store-uri {os.path.join(PROJECT_ROOT, 'mlruns')}\n")

    # ── Prepare feature matrix X and target matrix Y ──────────────────────
    X = df[ALL_FEATURES].values   # Shape: (n_stations, 17 features)
    Y = df[TARGET_COLS].values    # Shape: (n_stations, 3 targets)

    # Split: training rows = stations where all 3 target months exist
    # (We want to evaluate only on stations with real Oct–Dec ground truth)
    test_mask  = df[TARGET_COLS].notna().all(axis=1).values
    train_mask = np.ones(len(df), dtype=bool)   # All stations in training

    X_train = X[train_mask]
    X_test  = X[test_mask]

    print(f"  Training samples : {X_train.shape[0]} stations")
    print(f"  Test samples     : {X_test.shape[0]} stations (have full Oct–Dec data)")
    print()

    # ── Train all models for each target ──────────────────────────────────
    all_results = []
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)   # Fit imputer on training data only (no test leakage!)

    for target_idx, target_col in enumerate(TARGET_COLS):
        print(f"  ── Target: {target_col} ─────────────────────────────────────")
        y_train = Y[train_mask, target_idx]
        y_test  = Y[test_mask,  target_idx]

        # Remove rows from training where BOTH feature AND target are NaN
        # (can't learn from completely missing data)
        valid_train = ~np.isnan(y_train)
        X_tr = X_train[valid_train]
        y_tr = y_train[valid_train]

        # Remove test rows where target is NaN (nothing to evaluate against)
        valid_test = ~np.isnan(y_test)
        X_te = X_test[valid_test]
        y_te = y_test[valid_test]

        if len(y_tr) < 10:
            print(f"    [SKIP] Not enough training samples for {target_col}")
            continue

        for model_def in MODEL_DEFINITIONS:
            result = train_and_log(
                model_def, X_tr, y_tr, X_te, y_te, target_col, imputer
            )
            all_results.append(result)

    # ── Select best model per target (by R²) ──────────────────────────────
    print()
    print("  ── BEST MODELS BY TARGET ──────────────────────────────────────")
    best_models = {}
    for target_col in TARGET_COLS:
        target_results = [r for r in all_results if r["target"] == target_col]
        if not target_results:
            continue
        best = max(target_results, key=lambda r: r["r2"])
        best_models[target_col] = best
        print(f"  {target_col}: {best['name']:20s}  R²={best['r2']:.4f}  MAE={best['mae']:.0f}")

    # ── Save best model bundle ─────────────────────────────────────────────
    # We save a dict containing all three best models (one per target month)
    # and the feature column list so the prediction script can use it.
    #
    # The dict structure:
    # {
    #   "adt_oct": <fitted Pipeline>,
    #   "adt_nov": <fitted Pipeline>,
    #   "adt_dec": <fitted Pipeline>,
    #   "feature_cols": [...],
    #   "metrics": {...},
    # }
    model_bundle = {
        target: result["pipeline"]
        for target, result in best_models.items()
    }
    model_bundle["feature_cols"] = ALL_FEATURES
    model_bundle["metrics"] = {
        target: {"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"], "model": r["name"]}
        for target, r in best_models.items()
    }

    bundle_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(model_bundle, bundle_path)
    print()
    print(f"  [SAVED] models/best_model.pkl  ({os.path.getsize(bundle_path)/1024:.0f} KB)")

    # Also save metrics as JSON (human-readable, used by evaluate_model and reports)
    metrics_path = os.path.join(REPORTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(model_bundle["metrics"], f, indent=2)
    print(f"  [SAVED] reports/training_metrics.json")

    # ── Feature importance (for RandomForest target=adt_oct as reference) ──
    rf_oct = next(
        (r for r in all_results if r["name"] == "RandomForest" and r["target"] == "adt_oct"),
        None
    )
    if rf_oct:
        pipeline = rf_oct["pipeline"]
        # Extract the actual model from the pipeline (last step)
        rf_model = pipeline.named_steps["model"]
        importances = rf_model.feature_importances_
        feat_imp = pd.DataFrame(
            {"feature": ALL_FEATURES, "importance": importances}
        ).sort_values("importance", ascending=False)

        print()
        print("  ── FEATURE IMPORTANCE (RandomForest → adt_oct) ───────────────")
        for _, row in feat_imp.head(8).iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:30s} {row['importance']:.4f}  {bar}")

        feat_imp.to_csv(
            os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False
        )
        print(f"  [SAVED] reports/feature_importance.csv")

    print()
    print("  STAGE 5 COMPLETE. Next: bruin run assets/ml/evaluate_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
