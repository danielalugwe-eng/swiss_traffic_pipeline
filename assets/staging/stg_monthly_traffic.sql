/* =============================================================================
   @bruin
   name: staging.stg_monthly_traffic
   type: duckdb.sql

   -- WHAT IS THIS ASSET?
   -- It transforms the raw data from WIDE format to LONG format.
   --
   -- RIGHT NOW (WIDE format) - one row per station per metric:
   -- station_id | metric | adt_01 | adt_02 | adt_03 | ... | adt_12
   -- 64         | ADT    | 82000  | 79000  | 85000  | ... | 88000
   --
   -- AFTER (LONG format) - one row per station per metric per month:
   -- station_id | metric | month_num | month_name | adt_value
   -- 64         | ADT    | 1         | January    | 82000
   -- 64         | ADT    | 2         | February   | 79000
   -- 64         | ADT    | 3         | March      | 85000
   -- ...        | ...    | ...       | ...        | ...
   -- 64         | ADT    | 12        | December   | 88000
   --
   -- WHY LONG FORMAT?
   -- Long format is the "tidy data" standard for analytics and visualization.
   -- It's much easier to:
   --  • Filter by month: WHERE month_num = 7   (July)
   --  • Plot time-series: x=month_num, y=adt_value
   --  • Compute seasonal averages: GROUP BY month_num
   -- Wide format is good for ML feature matrices (handled in mart layer).
   --
   -- TECHNIQUE USED: UNPIVOT (or manual UNION ALL)
   -- DuckDB supports UNPIVOT natively. We use UNION ALL for clarity and
   -- compatibility — it makes the transformation explicit and easy to understand.

   materialization:
     type: table

   depends:
     - raw.ingest_traffic_csv
     - staging.stg_stations

   tags:
     - staging
     - monthly
     - fact
   @end
   ============================================================================= */

-- =============================================================================
-- UNPIVOT: One SELECT per month, all UNIONed together.
-- UNION ALL combines the results of multiple SELECT statements.
-- The difference between UNION and UNION ALL:
--   UNION removes duplicate rows (slow — has to scan everything).
--   UNION ALL keeps all rows including duplicates (fast — no scanning).
-- We use UNION ALL because there are no duplicates here (each SELECT produces
-- a different month_num).
--
-- COALESCE(r.canton, s.canton) = use raw canton if available, else staging's.
-- This is a safety fallback in case the join fails for any row.
-- =============================================================================

SELECT
    r.station_id,
    r.station_name,
    r.canton,
    r.road,
    r.metric_type,
    s.road_category,
    s.is_romandy,
    s.months_with_data,
    1                          AS month_num,
    'January'                  AS month_name,
    'Q1'                       AS quarter,
    'Winter'                   AS season,          -- Northern Hemisphere meteorological season
    r.adt_01                   AS adt_value         -- The actual measurement for this month
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       2, 'February', 'Q1', 'Winter', r.adt_02
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       3, 'March', 'Q1', 'Spring', r.adt_03
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       4, 'April', 'Q2', 'Spring', r.adt_04
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       5, 'May', 'Q2', 'Spring', r.adt_05
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       6, 'June', 'Q2', 'Summer', r.adt_06
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       7, 'July', 'Q3', 'Summer', r.adt_07
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       8, 'August', 'Q3', 'Summer', r.adt_08
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       9, 'September', 'Q3', 'Autumn', r.adt_09
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       10, 'October', 'Q4', 'Autumn', r.adt_10
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       11, 'November', 'Q4', 'Autumn', r.adt_11
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

UNION ALL

SELECT r.station_id, r.station_name, r.canton, r.road, r.metric_type,
       s.road_category, s.is_romandy, s.months_with_data,
       12, 'December', 'Q4', 'Winter', r.adt_12
FROM raw.annual_results r
JOIN staging.stg_stations s ON r.station_id = s.station_id
WHERE r.metric_type IN ('ADT', 'AWT', 'ADT Tu-Th', 'ADT Sa', 'ADT Su',
                        'ADT HV', 'ADT HGV', 'AWT HV', 'AWT HGV')

ORDER BY station_id, metric_type, month_num
