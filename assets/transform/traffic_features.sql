/* =============================================================================
   @bruin
   name: mart.traffic_features
   type: duckdb.sql

   -- WHAT IS THE MART LAYER?
   -- The mart ("data mart") is the last SQL layer — the "store shelves".
   -- Everything here is pre-computed, analytics-ready, and machine-learning-ready.
   -- Analysts and ML models query this layer, never the raw or staging layers.
   --
   -- WHAT THIS ASSET PRODUCES:
   -- mart.traffic_features — ML feature matrix (one row per station)
   --
   -- THIS IS THE INPUT TABLE FOR THE ML PIPELINE.
   -- It is a WIDE table (many columns per row) because scikit-learn expects
   -- a feature matrix in the shape: [n_samples  ×  n_features]
   --
   -- FEATURE ENGINEERING PHILOSOPHY:
   -- Raw ADT values (Jan, Feb, ..., Sep) are already useful features.
   -- But adding derived features helps the model learn faster:
   --   • Seasonal ratios reveal summer tourism vs commuter patterns
   --   • Heavy-vehicle ratio identifies freight corridors
   --   • Weekday/weekend ratio separates leisure from business routes
   -- These features encode domain knowledge (from transport engineering)
   -- into numbers the model can use.
   --
   -- TRAIN / TEST SPLIT DESIGN:
   -- Features (X): months Jan–Sep (the "visible" data a model would have mid-year)
   -- Targets  (y): months Oct, Nov, Dec (what we want to PREDICT)
   -- This simulates a real forecast: given 9 months of data, predict Q4.

   materialization:
     type: table

   depends:
     - staging.stg_stations
     - staging.stg_monthly_traffic

   tags:
     - mart
     - ml
     - features
   @end
   ============================================================================= */

-- =============================================================================
-- CTE 1: Base ADT monthly values for every station
-- CTE = "Common Table Expression" — a named sub-query.
-- Think of it as a temporary table only available within this query.
-- CTEs make complex SQL readable by breaking it into named steps.
-- =============================================================================

WITH adt_wide AS (
    -- Pivot from long format back to wide format for the feature matrix.
    -- We use conditional aggregation: MAX(CASE WHEN month=X THEN value END)
    -- This is equivalent to a PIVOT but works on all SQL dialects.
    -- MAX() is used because there's only one row per (station, month) for ADT.
    SELECT
        station_id,
        station_name,
        canton,
        road,
        road_category,
        is_romandy,
        months_with_data,

        -- ── TRAINING FEATURES (X): Jan–Sep ──────────────────────────────────
        MAX(CASE WHEN month_num = 1  AND metric_type = 'ADT' THEN adt_value END) AS adt_jan,
        MAX(CASE WHEN month_num = 2  AND metric_type = 'ADT' THEN adt_value END) AS adt_feb,
        MAX(CASE WHEN month_num = 3  AND metric_type = 'ADT' THEN adt_value END) AS adt_mar,
        MAX(CASE WHEN month_num = 4  AND metric_type = 'ADT' THEN adt_value END) AS adt_apr,
        MAX(CASE WHEN month_num = 5  AND metric_type = 'ADT' THEN adt_value END) AS adt_may,
        MAX(CASE WHEN month_num = 6  AND metric_type = 'ADT' THEN adt_value END) AS adt_jun,
        MAX(CASE WHEN month_num = 7  AND metric_type = 'ADT' THEN adt_value END) AS adt_jul,
        MAX(CASE WHEN month_num = 8  AND metric_type = 'ADT' THEN adt_value END) AS adt_aug,
        MAX(CASE WHEN month_num = 9  AND metric_type = 'ADT' THEN adt_value END) AS adt_sep,

        -- ── TEST / VALIDATION TARGETS (y): Oct–Dec ──────────────────────────
        MAX(CASE WHEN month_num = 10 AND metric_type = 'ADT' THEN adt_value END) AS adt_oct,
        MAX(CASE WHEN month_num = 11 AND metric_type = 'ADT' THEN adt_value END) AS adt_nov,
        MAX(CASE WHEN month_num = 12 AND metric_type = 'ADT' THEN adt_value END) AS adt_dec,

        -- ── WEEKDAY TRAFFIC (AWT = Average Workday Traffic) ──────────────────
        -- AWT is always higher than ADT because it excludes quiet weekends.
        -- AWT/ADT ratio = how much commuter-dominated the station is.
        MAX(CASE WHEN month_num = 1  AND metric_type = 'AWT' THEN adt_value END) AS awt_jan,
        MAX(CASE WHEN month_num = 7  AND metric_type = 'AWT' THEN adt_value END) AS awt_jul,

        -- ── WEEKEND TRAFFIC (Saturday and Sunday ADT) ────────────────────────
        MAX(CASE WHEN month_num = 7  AND metric_type = 'ADT Sa' THEN adt_value END) AS adt_sa_jul,
        MAX(CASE WHEN month_num = 7  AND metric_type = 'ADT Su' THEN adt_value END) AS adt_su_jul,

        -- ── HEAVY VEHICLE TRAFFIC ─────────────────────────────────────────────
        -- HV = Heavy Vehicles (buses + all trucks: classes 1, 8, 9, 10)
        -- HGV = Heavy Goods Vehicles (freight trucks only: classes 8, 9, 10)
        -- Source: adtwithclasses table via the stg_monthly_traffic view
        MAX(CASE WHEN month_num = 1  AND metric_type = 'ADT HV'  THEN adt_value END) AS hv_jan,
        MAX(CASE WHEN month_num = 7  AND metric_type = 'ADT HV'  THEN adt_value END) AS hv_jul,
        MAX(CASE WHEN month_num = 1  AND metric_type = 'ADT HGV' THEN adt_value END) AS hgv_jan,
        MAX(CASE WHEN month_num = 7  AND metric_type = 'ADT HGV' THEN adt_value END) AS hgv_jul

    FROM staging.stg_monthly_traffic
    GROUP BY station_id, station_name, canton, road, road_category, is_romandy, months_with_data
),

-- =============================================================================
-- CTE 2: Derived / Engineered Features
-- These features encode domain knowledge into numeric form.
-- An ML model could in theory learn these from raw data given enough samples,
-- but with only ~200 stations, giving the model pre-computed ratios
-- is more efficient (it's called "feature engineering").
-- =============================================================================

engineered AS (
    SELECT
        *,

        -- PEAK_MONTH_ADT: the highest ADT across any of the 9 training months.
        -- High peak = tourist/holiday route (alpine passes peak in summer).
        -- Low peak relative to average = stable commuter road.
        GREATEST(
            COALESCE(adt_jan, 0), COALESCE(adt_feb, 0), COALESCE(adt_mar, 0),
            COALESCE(adt_apr, 0), COALESCE(adt_may, 0), COALESCE(adt_jun, 0),
            COALESCE(adt_jul, 0), COALESCE(adt_aug, 0), COALESCE(adt_sep, 0)
        ) AS peak_month_adt,

        -- TROUGH_MONTH_ADT: the lowest ADT across training months.
        -- For winter-closed alpine passes, this would be NULL/0.
        -- For urban motorways, this stays relatively high even in January.
        LEAST(
            COALESCE(adt_jan, 999999), COALESCE(adt_feb, 999999), COALESCE(adt_mar, 999999),
            COALESCE(adt_apr, 999999), COALESCE(adt_may, 999999), COALESCE(adt_jun, 999999),
            COALESCE(adt_jul, 999999), COALESCE(adt_aug, 999999), COALESCE(adt_sep, 999999)
        ) AS trough_month_adt,

        -- MEAN JAN–SEP ADT: average of the first 9 months.
        -- Used to normalise seasonal ratios.
        (
            COALESCE(adt_jan, 0) + COALESCE(adt_feb, 0) + COALESCE(adt_mar, 0) +
            COALESCE(adt_apr, 0) + COALESCE(adt_may, 0) + COALESCE(adt_jun, 0) +
            COALESCE(adt_jul, 0) + COALESCE(adt_aug, 0) + COALESCE(adt_sep, 0)
        ) / NULLIF(
            (adt_jan IS NOT NULL)::INT + (adt_feb IS NOT NULL)::INT + (adt_mar IS NOT NULL)::INT +
            (adt_apr IS NOT NULL)::INT + (adt_may IS NOT NULL)::INT + (adt_jun IS NOT NULL)::INT +
            (adt_jul IS NOT NULL)::INT + (adt_aug IS NOT NULL)::INT + (adt_sep IS NOT NULL)::INT,
            0
        ) AS mean_adt_jan_sep,

        -- SUMMER PEAK RATIO: July ADT / January ADT.
        -- > 2.0 = strong seasonal tourism (alpine passes, lake resorts)
        -- ~ 1.0 = stable year-round commuter traffic (urban motorways)
        -- NULLIF prevents division-by-zero; result is NULL if Jan is NULL/0.
        ROUND(COALESCE(adt_jul, 0) / NULLIF(adt_jan, 0), 3) AS summer_peak_ratio,

        -- WEEKDAY vs WEEKEND RATIO: AWT_Jul / ADT_Jul
        -- > 1.3 = strongly commuter-dominated
        -- ~ 1.0 = balanced traffic (or tourism-dominated in summer)
        ROUND(COALESCE(awt_jul, 0) / NULLIF(adt_jul, 0), 3) AS weekday_weekend_ratio,

        -- HEAVY VEHICLE PERCENTAGE in July (peak month for trucks before holidays)
        -- HGV% > 15% = major freight corridor (e.g., A2 Basel-Lugano)
        -- HGV% < 3%  = urban commuter road
        ROUND(COALESCE(hgv_jul, 0) / NULLIF(adt_jul, 0) * 100, 2) AS hgv_pct_jul,

        -- WINTER DEPRESSION RATIO: Jan ADT / Mean Jan–Sep ADT
        -- Measures how much traffic drops in January vs the yearly average.
        -- <0.7 = significant winter depression (seasonal tourism roads)
        -- >0.9 = nearly constant throughout year (commuter/freight)
        ROUND(
            COALESCE(adt_jan, 0) / NULLIF(
                (
                    COALESCE(adt_jan, 0) + COALESCE(adt_feb, 0) + COALESCE(adt_mar, 0) +
                    COALESCE(adt_apr, 0) + COALESCE(adt_may, 0) + COALESCE(adt_jun, 0) +
                    COALESCE(adt_jul, 0) + COALESCE(adt_aug, 0) + COALESCE(adt_sep, 0)
                ) / NULLIF(
                    (adt_jan IS NOT NULL)::INT + (adt_feb IS NOT NULL)::INT + (adt_mar IS NOT NULL)::INT +
                    (adt_apr IS NOT NULL)::INT + (adt_may IS NOT NULL)::INT + (adt_jun IS NOT NULL)::INT +
                    (adt_jul IS NOT NULL)::INT + (adt_aug IS NOT NULL)::INT + (adt_sep IS NOT NULL)::INT,
                    0
                ),
                0
            ), 3
        ) AS winter_depression_ratio,

        -- CANTON CODE (numeric encoding for ML models)
        -- ML algorithms require numbers, not strings.
        -- One-hot encoding would create a column per canton — too many for 200 rows.
        -- Label encoding (integer) is sufficient for tree-based models (RF, GBM).
        CASE canton
            WHEN 'VD' THEN 1   -- Vaud (Lausanne)
            WHEN 'GE' THEN 2   -- Geneva
            WHEN 'NE' THEN 3   -- Neuchâtel
            WHEN 'FR' THEN 4   -- Fribourg
            WHEN 'JU' THEN 5   -- Jura
            WHEN 'VS' THEN 6   -- Valais
            WHEN 'BE' THEN 7   -- Berne (bilingual)
            WHEN 'AG' THEN 8   -- Aargau
            WHEN 'ZH' THEN 9   -- Zurich
            WHEN 'LU' THEN 10  -- Lucerne
            WHEN 'BS' THEN 11  -- Basel-Stadt
            WHEN 'BL' THEN 12  -- Basel-Landschaft
            WHEN 'SG' THEN 13  -- St. Gallen
            WHEN 'GR' THEN 14  -- Graubünden
            WHEN 'TI' THEN 15  -- Ticino (Italian-speaking)
            WHEN 'SO' THEN 16  -- Solothurn
            WHEN 'TG' THEN 17  -- Thurgau
            WHEN 'SH' THEN 18  -- Schaffhausen
            ELSE 0             -- Unknown / Other cantons
        END AS canton_code,

        -- ROAD TYPE CODE (0 = National Road, 1 = Motorway)
        CASE road_category
            WHEN 'Motorway'       THEN 1
            WHEN 'National Road'  THEN 0
            ELSE 0
        END AS road_type_code,

        -- ROMANDY BINARY FLAG (0 or 1 for ML models)
        is_romandy::INT AS is_romandy_int

    FROM adt_wide
)

-- =============================================================================
-- FINAL SELECT: filter to stations with sufficient training data
-- We require at least 7 of 9 training months to have ADT values.
-- Stations with more missing data would add noise, not signal, to the model.
-- =============================================================================

SELECT
    -- Identifiers (not features — ML model ignores these)
    station_id,
    station_name,
    canton,
    road,

    -- ── FEATURES (X) — what the model SEES as input ──────────────────────────
    -- Raw monthly ADT values Jan–Sep
    adt_jan, adt_feb, adt_mar, adt_apr, adt_may,
    adt_jun, adt_jul, adt_aug, adt_sep,

    -- Engineered features
    summer_peak_ratio,
    weekday_weekend_ratio,
    hgv_pct_jul,
    winter_depression_ratio,
    mean_adt_jan_sep,

    -- Categorical encodings
    canton_code,
    road_type_code,
    is_romandy_int,

    -- ── TARGETS (y) — what the model PREDICTS ────────────────────────────────
    adt_oct,
    adt_nov,
    adt_dec,

    -- ── METADATA (for reporting, not for training) ────────────────────────────
    road_category,
    is_romandy,
    months_with_data,
    peak_month_adt,
    trough_month_adt

FROM engineered

-- Only keep stations that have enough training-month data
-- (at least 7/9 months Jan–Sep must be non-NULL)
WHERE (
    (adt_jan IS NOT NULL)::INT + (adt_feb IS NOT NULL)::INT +
    (adt_mar IS NOT NULL)::INT + (adt_apr IS NOT NULL)::INT +
    (adt_may IS NOT NULL)::INT + (adt_jun IS NOT NULL)::INT +
    (adt_jul IS NOT NULL)::INT + (adt_aug IS NOT NULL)::INT +
    (adt_sep IS NOT NULL)::INT
) >= 7

ORDER BY station_id
