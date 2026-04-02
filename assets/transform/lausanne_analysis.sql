/* =============================================================================
   @bruin
   name: mart.lausanne_analysis
   type: duckdb.sql

   -- WHAT IS THIS ASSET?
   -- Deep analysis of Lausanne and the Vaud (VD) canton stations.
   --
   -- WHY LAUSANNE?
   -- Among all Romandy stations, Lausanne has:
   --  • The highest traffic volume (~97,000 vehicles/day)
   --  • 3 measuring stations providing spatial coverage
   --  • Complete 12-month data (no gaps)
   --  • Position on A9 motorway (European route E62 Basel–Genoa)
   --
   -- KEY LAUSANNE STATIONS:
   --  • Station 064: CONT. DE LAUSANNE (A9 motorway)      ~97,662 ADT/year
   --  • Station 043: PREVERENGES         (A1 motorway)    ~97,195 ADT/year
   --  • Station 149: MEX                 (A1 motorway)    ~74,395 ADT/year
   --  • Station 083: VILLENEUVE          (A9 motorway)    ~67,693 ADT/year
   --
   -- WHAT THIS TABLE IS USED FOR:
   --  • Lausanne-specific charts in the historical report
   --  • Peak-hour / peak-month identification
   --  • Weekday vs weekend traffic comparison
   --  • Seasonal variation analysis (ski season, summer tourism)

   materialization:
     type: table

   depends:
     - staging.stg_monthly_traffic
     - staging.stg_stations

   tags:
     - mart
     - lausanne
     - romandy
     - analysis
   @end
   ============================================================================= */

WITH

-- All VD canton stations with their monthly long-format traffic data
vd_monthly AS (
    SELECT
        t.station_id,
        t.station_name,
        t.canton,
        t.road,
        t.road_category,
        t.metric_type,
        t.month_num,
        t.month_name,
        t.quarter,
        t.season,
        t.adt_value
    FROM staging.stg_monthly_traffic t
    WHERE t.canton = 'VD'
),

-- ============================================================
-- Focus: ADT (primary metric) only, pivoted to wide analysis
-- One row per (station × month) with all derived fields.
-- ============================================================

adt_only AS (
    SELECT
        station_id,
        station_name,
        road,
        month_num,
        month_name,
        quarter,
        season,
        adt_value,

        -- Year average for this station (average of available months)
        AVG(adt_value) OVER (PARTITION BY station_id) AS station_year_avg,

        -- Monthly deviation from the station's own yearly average.
        -- Positive = above average (summer peak)
        -- Negative = below average (winter trough)
        adt_value - AVG(adt_value) OVER (PARTITION BY station_id) AS deviation_from_avg,

        -- Percentage deviation: useful for comparing stations of different sizes.
        -- e.g., +15% in July means 15% more traffic than the annual average.
        ROUND(
            (adt_value - AVG(adt_value) OVER (PARTITION BY station_id))
            / NULLIF(AVG(adt_value) OVER (PARTITION BY station_id), 0) * 100,
            1
        ) AS pct_deviation,

        -- Rank of this month (1 = busiest month for this station)
        RANK() OVER (PARTITION BY station_id ORDER BY adt_value DESC NULLS LAST)
            AS month_rank_by_traffic,

        -- Rolling 3-month moving average (smooths out noise).
        -- LAG(1) = previous month's value   LAG(2) = two months ago
        -- LEAD(1) = next month's value
        AVG(adt_value) OVER (
            PARTITION BY station_id
            ORDER BY month_num
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ) AS rolling_3mo_avg

    FROM vd_monthly
    WHERE metric_type = 'ADT'
),

-- ============================================================
-- Weekday vs Weekend comparison for each station × month
-- ============================================================

weekday_weekend AS (
    SELECT
        station_id,
        month_num,
        -- Weekday traffic (Tue–Thu average = most representative of work days)
        MAX(CASE WHEN metric_type = 'ADT Tu-Th' THEN adt_value END) AS adt_tue_thu,
        -- Saturday
        MAX(CASE WHEN metric_type = 'ADT Sa'    THEN adt_value END) AS adt_saturday,
        -- Sunday
        MAX(CASE WHEN metric_type = 'ADT Su'    THEN adt_value END) AS adt_sunday,
        -- Primary ADT (all days average)
        MAX(CASE WHEN metric_type = 'ADT'       THEN adt_value END) AS adt_all_days
    FROM vd_monthly
    GROUP BY station_id, month_num
),

-- ============================================================
-- Peak and trough identification per station
-- ============================================================

peak_trough AS (
    SELECT
        station_id,
        MAX(adt_value)          AS peak_adt,
        MIN(adt_value)          AS trough_adt,
        MAX(month_name) FILTER (WHERE month_rank_by_traffic = 1) AS peak_month,
        MAX(month_name) FILTER (WHERE month_rank_by_traffic = 12) AS trough_month,
        ROUND(MAX(adt_value) / NULLIF(MIN(adt_value), 0), 2) AS peak_to_trough_ratio,
        -- Coefficient of Variation: standard deviation / mean
        -- Low CV (<0.1) = stable route. High CV (>0.3) = highly seasonal.
        ROUND(STDDEV(adt_value) / NULLIF(AVG(adt_value), 0), 3) AS cv_seasonality
    FROM adt_only
    GROUP BY station_id
)

-- ============================================================
-- FINAL SELECT: combine all analysis dimensions
-- ============================================================

SELECT
    a.station_id,
    a.station_name,
    a.road,
    a.month_num,
    a.month_name,
    a.quarter,
    a.season,

    -- Core traffic values
    ROUND(a.adt_value, 0)             AS adt_value,
    ROUND(a.station_year_avg, 0)       AS station_year_avg,
    ROUND(a.rolling_3mo_avg, 0)        AS rolling_3mo_avg_adt,

    -- Deviation analysis (how unusual is this month?)
    ROUND(a.deviation_from_avg, 0)     AS deviation_from_annual_avg,
    a.pct_deviation                    AS pct_deviation_from_avg,
    a.month_rank_by_traffic            AS month_busy_rank,

    -- Weekday vs weekend breakdown
    ROUND(w.adt_tue_thu, 0)            AS adt_tue_thu,
    ROUND(w.adt_saturday, 0)           AS adt_saturday,
    ROUND(w.adt_sunday, 0)             AS adt_sunday,

    -- Weekday/weekend ratio: > 1.3 means weekday dominated (commuter)
    ROUND(w.adt_tue_thu / NULLIF(w.adt_sunday, 0), 2)  AS tue_thu_vs_sunday_ratio,

    -- Station-level summary metrics (same for all months of a station)
    pt.peak_adt,
    pt.trough_adt,
    pt.peak_month,
    pt.trough_month,
    pt.peak_to_trough_ratio,
    pt.cv_seasonality,

    -- Classify station type based on seasonality coefficient of variation:
    -- This is a rule-based classification using domain knowledge.
    CASE
        WHEN pt.cv_seasonality < 0.08  THEN 'Urban Commuter — flat seasonal pattern'
        WHEN pt.cv_seasonality < 0.15  THEN 'Mixed Urban/Tourist — moderate variation'
        WHEN pt.cv_seasonality < 0.25  THEN 'Tourist — strong summer peak'
        ELSE                                'Alpine/Seasonal — extreme seasonal swings'
    END AS route_type_classification

FROM adt_only a
JOIN weekday_weekend w ON a.station_id = w.station_id AND a.month_num = w.month_num
JOIN peak_trough     pt ON a.station_id = pt.station_id

ORDER BY a.station_id, a.month_num
