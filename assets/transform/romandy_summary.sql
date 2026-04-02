/* =============================================================================
   @bruin
   name: mart.romandy_summary
   type: duckdb.sql

   -- WHAT IS THIS ASSET?
   -- Canton-level aggregate summary for all 6 French-speaking (Romandy) cantons:
   --   VD = Vaud       (capital: Lausanne)
   --   GE = Geneva     (capital: Geneva)
   --   NE = Neuchâtel  (capital: Neuchâtel)
   --   FR = Fribourg   (bilingual, capital: Fribourg)
   --   JU = Jura       (capital: Delémont)
   --   VS = Valais     (bilingual, capital: Sion)
   --
   -- WHY CANTON-LEVEL AGGREGATION?
   -- Individual station data is granular (sensor point on one road).
   -- Canton-level data answers higher-level questions:
   --   • Which Romandy canton has the highest total traffic load?
   --   • Which is most seasonal (ski/summer tourism)?
   --   • Which has the highest freight-truck share?
   -- This table feeds the "Romandy Overview" section of the HTML report.
   --
   -- OUTPUT: ~6 rows (one per Romandy canton) × many KPI columns.

   materialization:
     type: table

   depends:
     - staging.stg_monthly_traffic
     - staging.stg_stations
     - mart.traffic_features

   tags:
     - mart
     - romandy
     - reporting
     - kpi
   @end
   ============================================================================= */

WITH

-- ── Step 1: ADT data for all Romandy stations (long format) ──────────────────
romandy_long AS (
    SELECT
        t.station_id,
        t.station_name,
        t.canton,
        t.road,
        t.road_category,
        t.month_num,
        t.month_name,
        t.season,
        t.adt_value
    FROM staging.stg_monthly_traffic t
    WHERE t.is_romandy = TRUE
      AND t.metric_type = 'ADT'
),

-- ── Step 2: Monthly averages per canton ──────────────────────────────────────
-- For each canton and month, average the ADT across all stations in that canton.
-- This gives a "canton typical day in month X" figure.
canton_monthly AS (
    SELECT
        canton,
        month_num,
        month_name,
        season,
        COUNT(DISTINCT station_id)     AS station_count,
        ROUND(AVG(adt_value), 0)       AS avg_adt,
        ROUND(SUM(adt_value), 0)       AS total_adt,       -- raw sum (not physically meaningful but useful for ranking)
        ROUND(MAX(adt_value), 0)       AS max_station_adt, -- busiest station in canton that month
        ROUND(MIN(adt_value), 0)       AS min_station_adt
    FROM romandy_long
    WHERE adt_value IS NOT NULL
    GROUP BY canton, month_num, month_name, season
),

-- ── Step 3: Annual KPIs per canton ───────────────────────────────────────────
canton_annual AS (
    SELECT
        canton,

        -- Total number of measuring stations in this canton
        COUNT(DISTINCT station_id)                    AS total_stations,

        -- Average annual ADT across all stations in this canton
        ROUND(AVG(adt_value), 0)                      AS canton_avg_adt,

        -- Peak month: which month has the highest average ADT in this canton?
        -- ARG_MAX is a DuckDB function: returns month_num where adt_value is max.
        -- Equivalent to: SELECT month_name WHERE adt = MAX(adt) LIMIT 1
        ARG_MAX(month_name, adt_value)                AS peak_month_name,
        ROUND(MAX(adt_value), 0)                      AS peak_month_adt,

        -- Trough month: lowest traffic month
        ARG_MIN(month_name, adt_value)                AS trough_month_name,
        ROUND(MIN(adt_value), 0)                      AS trough_month_adt,

        -- Seasonality ratio: peak / trough. Higher = more seasonal.
        -- Urban Geneva might be 1.15. Alpine Valais might be 2.5+.
        ROUND(MAX(adt_value) / NULLIF(MIN(adt_value), 0), 2)  AS seasonality_ratio,

        -- Summer (Jul) average ADT across canton stations
        ROUND(AVG(CASE WHEN month_num = 7 THEN adt_value END), 0)   AS avg_adt_july,

        -- Winter (Jan) average ADT across canton stations
        ROUND(AVG(CASE WHEN month_num = 1 THEN adt_value END), 0)   AS avg_adt_jan,

        -- Q1 average (Jan–Mar): represents winter baseline
        ROUND(AVG(CASE WHEN month_num IN (1,2,3) THEN adt_value END), 0) AS avg_adt_q1,

        -- Q3 average (Jul–Sep): represents summer peak
        ROUND(AVG(CASE WHEN month_num IN (7,8,9) THEN adt_value END), 0) AS avg_adt_q3,

        -- Standard deviation of monthly canton averages — measures how much
        -- traffic fluctuates month to month across this canton's stations.
        ROUND(STDDEV(adt_value), 0)                    AS monthly_std_dev,

        -- Coefficient of variation (CV): std / mean — normalized seasonality.
        -- <0.1 = very stable (Geneva/urban)
        -- >0.3 = highly seasonal (alpine passes)
        ROUND(STDDEV(adt_value) / NULLIF(AVG(adt_value), 0), 3)  AS cv_seasonality

    FROM romandy_long
    WHERE adt_value IS NOT NULL
    GROUP BY canton
),

-- ── Step 4: Heavy vehicle share from the ML features table ───────────────────
-- mart.traffic_features has pre-computed hgv_pct_jul per station.
-- We aggregate to canton level here.
canton_hgv AS (
    SELECT
        canton,
        ROUND(AVG(hgv_pct_jul), 2)      AS avg_hgv_pct_jul,
        ROUND(MAX(hgv_pct_jul), 2)      AS max_hgv_pct_jul
    FROM mart.traffic_features
    WHERE is_romandy = TRUE
      AND hgv_pct_jul IS NOT NULL
    GROUP BY canton
),

-- ── Step 5: Count fully-complete stations (12/12 months ADT present) ─────────
canton_completeness AS (
    SELECT
        canton,
        COUNT(*) FILTER (WHERE months_with_data = 12)  AS stations_12mo_complete,
        COUNT(*) FILTER (WHERE months_with_data >= 9)  AS stations_9mo_plus
    FROM staging.stg_stations
    WHERE is_romandy = TRUE
    GROUP BY canton
),

-- ── Step 6: Full canton name (descriptive label) ─────────────────────────────
canton_labels AS (
    SELECT * FROM (VALUES
        ('VD', 'Vaud',      'Lausanne',    'French'),
        ('GE', 'Geneva',    'Geneva',      'French'),
        ('NE', 'Neuchâtel', 'Neuchâtel',   'French'),
        ('FR', 'Fribourg',  'Fribourg',    'Bilingual FR/DE'),
        ('JU', 'Jura',      'Delémont',    'French'),
        ('VS', 'Valais',    'Sion',        'Bilingual FR/DE')
    ) AS t(canton, canton_name, capital, language)
)

-- ── Final SELECT: join all CTEs ───────────────────────────────────────────────
SELECT
    ca.canton,
    cl.canton_name,
    cl.capital,
    cl.language,

    -- Station inventory
    ca.total_stations,
    cc.stations_12mo_complete,
    cc.stations_9mo_plus,

    -- Annual traffic KPIs
    ca.canton_avg_adt,
    ca.avg_adt_july                     AS avg_adt_summer_peak,
    ca.avg_adt_jan                      AS avg_adt_winter_trough,
    ca.avg_adt_q1,
    ca.avg_adt_q3,

    -- Seasonality indicators
    ca.peak_month_name,
    ca.peak_month_adt,
    ca.trough_month_name,
    ca.trough_month_adt,
    ca.seasonality_ratio,
    ca.cv_seasonality,

    -- Classify seasonality profile
    CASE
        WHEN ca.cv_seasonality < 0.08  THEN 'Stable Urban'
        WHEN ca.cv_seasonality < 0.18  THEN 'Moderate Seasonal'
        WHEN ca.cv_seasonality < 0.30  THEN 'Strong Seasonal'
        ELSE                                'Highly Alpine/Seasonal'
    END AS seasonality_profile,

    -- Heavy vehicle / freight share
    COALESCE(hv.avg_hgv_pct_jul, 0)    AS avg_hgv_pct_july,
    COALESCE(hv.max_hgv_pct_jul, 0)    AS max_hgv_pct_july,

    -- Summer/Winter lift factor (how much does summer beat winter?)
    ROUND(ca.avg_adt_q3 / NULLIF(ca.avg_adt_q1, 0), 2)   AS summer_winter_lift,

    -- Rank within Romandy by average ADT (1 = busiest canton)
    RANK() OVER (ORDER BY ca.canton_avg_adt DESC)          AS romandy_traffic_rank

FROM canton_annual ca
JOIN canton_labels     cl ON ca.canton = cl.canton
LEFT JOIN canton_hgv   hv ON ca.canton = hv.canton
LEFT JOIN canton_completeness cc ON ca.canton = cc.canton

ORDER BY romandy_traffic_rank
