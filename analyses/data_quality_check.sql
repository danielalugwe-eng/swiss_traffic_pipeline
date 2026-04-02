-- =============================================================================
-- DATA QUALITY CHECK
-- analyses/data_quality_check.sql
-- =============================================================================
--
-- PURPOSE: Ad-hoc diagnostic queries for the Swiss Traffic dataset.
--           Run these in DuckDB or via Bruin's `bruin query` command to
--           understand data completeness before committing to ML training.
--
-- HOW TO RUN:
--   bruin query --asset analyses/data_quality_check.sql
--   OR: duckdb traffic.duckdb < analyses/data_quality_check.sql
--
-- WHAT IS "DATA QUALITY" IN THIS CONTEXT?
--   We have measuring stations across Switzerland, each reporting Average Daily
--   Traffic (ADT) for every month of 2025. Quality issues arise when:
--     • A station is closed in winter (e.g. alpine passes like the Furka)
--     • A sensor malfunctions and reports NULL for some months
--     • Administrative estimates replace sensor readings (*-flagged values)
--   Before training an ML model, we need to understand the extent of these
--   gaps so we can decide: impute vs drop vs flag.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- CHECK 1: Overall dataset dimensions
-- ---------------------------------------------------------------------------
-- How many rows, stations, cantons, and months of data do we have?

SELECT
    COUNT(DISTINCT station_id)   AS total_stations,
    COUNT(DISTINCT canton)       AS total_cantons,
    COUNT(*)                     AS total_rows_in_staging,
    -- Of those rows, how many have a valid ADT value?
    SUM(CASE WHEN adt IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_adt,
    -- And how many are NULL/missing?
    SUM(CASE WHEN adt IS NULL     THEN 1 ELSE 0 END) AS rows_without_adt,
    -- Percentage complete
    ROUND(100.0 * SUM(CASE WHEN adt IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_complete
FROM staging.stg_monthly_traffic
WHERE metric_type = 'ADT'   -- Focus on Average Daily Traffic only
;


-- ---------------------------------------------------------------------------
-- CHECK 2: NULL rate per canton
-- ---------------------------------------------------------------------------
-- Which cantons have the most missing data?
-- High NULL rates → fewer usable stations → weaker model for that canton.
-- Romandy cantons (VD, GE, VS, NE, FR, JU) should have good coverage given
-- the A1/A9 national road network.

SELECT
    canton,
    COUNT(DISTINCT station_id)                        AS n_stations,
    COUNT(*)                                          AS n_month_records,
    SUM(CASE WHEN adt IS NOT NULL THEN 1 ELSE 0 END)  AS n_present,
    SUM(CASE WHEN adt IS NULL     THEN 1 ELSE 0 END)  AS n_missing,
    ROUND(100.0 * SUM(CASE WHEN adt IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing,
    -- Flag: >30% missing means the canton is under-instrumented
    CASE WHEN ROUND(100.0 * SUM(CASE WHEN adt IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) > 30
         THEN '⚠️  High missingness'
         ELSE '✅  Acceptable'
    END AS data_quality_flag
FROM staging.stg_monthly_traffic
WHERE metric_type = 'ADT'
GROUP BY canton
ORDER BY pct_missing DESC
;


-- ---------------------------------------------------------------------------
-- CHECK 3: NULL rate per month across all stations
-- ---------------------------------------------------------------------------
-- Are certain months systematically missing?
-- December/January should be high on alpine closure stations.
-- If a non-alpine motorway shows NULLs in summer, that's a sensor fault,
-- not a seasonal closure — worth investigating.

SELECT
    month_name,
    month_num,
    COUNT(*)                                          AS n_stations_reporting,
    SUM(CASE WHEN adt IS NOT NULL THEN 1 ELSE 0 END)  AS n_present,
    SUM(CASE WHEN adt IS NULL     THEN 1 ELSE 0 END)  AS n_missing,
    ROUND(100.0 * SUM(CASE WHEN adt IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing
FROM staging.stg_monthly_traffic
WHERE metric_type = 'ADT'
GROUP BY month_name, month_num
ORDER BY month_num
;


-- ---------------------------------------------------------------------------
-- CHECK 4: Winter-closure stations
-- ---------------------------------------------------------------------------
-- Which stations have 3+ consecutive winter NULLs (Nov–Mar)?
-- These are almost certainly alpine passes with seasonal road closures.
-- These stations should be handled carefully: we can still use their summer
-- data for ML, but we cannot predict their winter values.

SELECT
    s.station_id,
    s.station_name,
    s.canton,
    s.road,
    s.has_winter_closing,   -- Flag set in stg_stations.sql
    -- Count months with NULL ADT
    s.months_with_data,
    (12 - s.months_with_data) AS months_missing,
    -- Classify closure type
    CASE
        WHEN s.months_with_data <= 6  THEN 'Major closure (≥6 months missing)'
        WHEN s.months_with_data <= 9  THEN 'Partial closure (3-5 months missing)'
        WHEN s.months_with_data <= 11 THEN 'Minor gap (1-2 months missing)'
        ELSE 'Complete data'
    END AS closure_category
FROM staging.stg_stations s
WHERE s.months_with_data < 12
ORDER BY s.months_with_data ASC
;


-- ---------------------------------------------------------------------------
-- CHECK 5: Stations with estimated values (flagged data)
-- ---------------------------------------------------------------------------
-- FEDRO flags some monthly values as estimates when sensor data is
-- unavailable (e.g. the sensor was offline, so they extrapolated using
-- nearby sensors or historical averages).
-- These are *not* NULL but are less reliable than direct measurements.
-- We note them but still use them — ML models are robust to small
-- amounts of imprecision.

SELECT
    s.station_id,
    s.station_name,
    s.canton,
    s.has_estimated_months,
    s.has_data_gaps
FROM staging.stg_stations s
WHERE s.has_estimated_months = TRUE OR s.has_data_gaps = TRUE
ORDER BY s.canton, s.station_name
;


-- ---------------------------------------------------------------------------
-- CHECK 6: Expected station counts per Romandy canton
-- ---------------------------------------------------------------------------
-- Cross-reference: How many stations does each Romandy canton contribute?
-- We expect VD to dominate (A1, A9, A12 all pass through it).
-- GE and NE have fewer motorway km, so fewer stations is normal.

SELECT
    canton,
    COUNT(*)                                          AS n_stations,
    SUM(CASE WHEN road_category = 'Motorway' THEN 1 ELSE 0 END) AS motorway_stations,
    SUM(CASE WHEN road_category = 'National Road' THEN 1 ELSE 0 END) AS national_road_stations,
    ROUND(AVG(annual_adt), 0)                         AS avg_annual_adt,
    MAX(annual_adt)                                   AS max_annual_adt,
    MIN(annual_adt)                                   AS min_annual_adt
FROM staging.stg_stations
WHERE canton IN ('VD', 'GE', 'VS', 'NE', 'FR', 'JU')
GROUP BY canton
ORDER BY n_stations DESC
;


-- ---------------------------------------------------------------------------
-- CHECK 7: Lausanne station sanity check
-- ---------------------------------------------------------------------------
-- Verify the flagship analysis station is present and has complete data.
-- Station 064 is the CONT. DE LAUSANNE on the A9 — our primary focus station.

SELECT
    s.station_id,
    s.station_name,
    s.canton,
    s.road,
    s.road_category,
    s.annual_adt,
    s.months_with_data,
    s.has_data_gaps,
    s.has_winter_closing,
    s.has_estimated_months
FROM staging.stg_stations s
WHERE s.canton = 'VD'
ORDER BY s.annual_adt DESC
LIMIT 10
;


-- ---------------------------------------------------------------------------
-- CHECK 8: Training data sufficiency for ML
-- ---------------------------------------------------------------------------
-- How many stations will actually make it into the ML feature matrix?
-- Recall: in traffic_features.sql we require ≥7 of the 9 training months
-- (Jan–Sep) to be non-NULL. Check how many pass and fail this threshold.

SELECT
    canton,
    COUNT(*)                                                          AS total_stations,
    SUM(CASE WHEN months_with_data >= 7  THEN 1 ELSE 0 END)          AS pass_threshold,
    SUM(CASE WHEN months_with_data < 7   THEN 1 ELSE 0 END)          AS fail_threshold,
    ROUND(100.0 * SUM(CASE WHEN months_with_data >= 7 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_usable
FROM staging.stg_stations
GROUP BY canton
ORDER BY pct_usable ASC
;


-- ---------------------------------------------------------------------------
-- CHECK 9: Summary verdict
-- ---------------------------------------------------------------------------
-- A single-row summary suitable for including in a data quality report.

SELECT
    (SELECT COUNT(DISTINCT station_id) FROM staging.stg_stations)         AS total_stations,
    (SELECT COUNT(DISTINCT station_id) FROM staging.stg_stations
     WHERE canton IN ('VD','GE','VS','NE','FR','JU'))                      AS romandy_stations,
    (SELECT COUNT(DISTINCT station_id) FROM staging.stg_stations
     WHERE months_with_data = 12)                                          AS stations_complete_data,
    (SELECT COUNT(DISTINCT station_id) FROM staging.stg_stations
     WHERE months_with_data >= 7)                                          AS stations_ml_usable,
    (SELECT ROUND(100.0 * COUNT(DISTINCT station_id) /
     (SELECT COUNT(*) FROM staging.stg_stations), 1)
     FROM staging.stg_stations WHERE months_with_data >= 7)               AS pct_ml_usable,
    (SELECT ROUND(AVG(annual_adt), 0) FROM staging.stg_stations
     WHERE canton IN ('VD','GE','VS','NE','FR','JU'))                      AS avg_romandy_adt,
    (SELECT station_name FROM staging.stg_stations
     WHERE canton = 'VD' ORDER BY annual_adt DESC LIMIT 1)                AS highest_traffic_station
;
