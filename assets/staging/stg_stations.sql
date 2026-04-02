/* =============================================================================
   @bruin
   name: staging.stg_stations
   type: duckdb.sql

   -- WHAT IS THE STAGING LAYER?
   -- Think of it as the "Quality Control Room" of our data factory.
   -- Raw data has messy formatting, duplicates, and varying conventions.
   -- The staging layer cleans all of that and creates a single reliable
   -- master record for each measuring station.
   --
   -- WHAT THIS ASSET PRODUCES:
   -- staging.stg_stations — ONE row per measuring station with:
   --   • Clean station metadata (ID, name, canton, road)
   --   • Computed flags: is_romandy, road_category
   --   • Quality metric: months_with_data (out of 12)
   --
   -- WHY "ONE ROW PER STATION"?
   -- The raw table has 5 rows per station (ADT, AWT, ADT Tu-Th, ADT Sa, ADT Su).
   -- We collapse those to 1 station row. This becomes our "dimension table"
   -- in data warehouse terminology — the master list of stations.
   --
   -- THIS IS A BRUIN "TABLE" MATERIALISATION:
   -- Bruin will execute this SQL and save the results as a real DuckDB table
   -- (not just a view). Every pipeline run recreates this table fresh.

   materialization:
     type: table

   depends:
     - raw.ingest_traffic_csv

   tags:
     - staging
     - stations
     - dimension
   @end
   ============================================================================= */

-- =============================================================================
-- STEP 1: Deduplicate to one row per station using only the ADT metric.
-- WHY ADT only?
-- ADT (Average Daily Traffic) is the primary metric — it counts ALL vehicles
-- for ALL days (weekdays + weekends averaged together). It is the standard
-- traffic measurement used by transport engineers worldwide.
-- We filter on metric_type = 'ADT' to get exactly one row per station.
-- =============================================================================

WITH base AS (
    SELECT
        -- Core station identifiers (from raw ingestion)
        station_id,
        station_name,
        canton,
        road,

        -- Remove anything in parentheses from station names.
        -- e.g. "CONT. DE LAUSANNE (AR)" → "CONT. DE LAUSANNE"
        -- The "(AR)" suffix means "Autoroutière" (on motorway).
        -- REGEXP_REPLACE(text, pattern, replacement) is a DuckDB function.
        REGEXP_REPLACE(station_name, '\s*\(.*\)\s*$', '') AS station_name_clean,

        -- Road category: roads starting with "A" are motorways (Autoroutes),
        -- roads starting with "H" are national roads (Routes nationales).
        -- CASE WHEN ... THEN ... ELSE ... END = SQL's if/else statement.
        CASE
            WHEN road LIKE 'A %' OR road LIKE 'A%'  THEN 'Motorway'
            WHEN road LIKE 'H %' OR road LIKE 'H%'  THEN 'National Road'
            ELSE 'Other'
        END AS road_category,

        -- Is this station in a French-speaking (Romandy) canton?
        -- VD = Vaud (Lausanne), GE = Geneva, NE = Neuchâtel,
        -- FR = Fribourg (bilingual), JU = Jura, VS = Valais (bilingual).
        CASE
            WHEN canton IN ('VD', 'GE', 'NE', 'JU', 'FR', 'VS') THEN TRUE
            ELSE FALSE
        END AS is_romandy,

        -- Count how many of the 12 monthly ADT values are NOT NULL.
        -- A station with 12/12 has full-year data — ideal for ML.
        -- A station with 0/12 had total sensor failure all year.
        -- COALESCE(x, 0) returns x if not null, else 0. We use this pattern
        -- to count non-null values: (adt IS NOT NULL)::INT = 1 or 0.
        (
            (adt_01 IS NOT NULL)::INT + (adt_02 IS NOT NULL)::INT +
            (adt_03 IS NOT NULL)::INT + (adt_04 IS NOT NULL)::INT +
            (adt_05 IS NOT NULL)::INT + (adt_06 IS NOT NULL)::INT +
            (adt_07 IS NOT NULL)::INT + (adt_08 IS NOT NULL)::INT +
            (adt_09 IS NOT NULL)::INT + (adt_10 IS NOT NULL)::INT +
            (adt_11 IS NOT NULL)::INT + (adt_12 IS NOT NULL)::INT
        ) AS months_with_data,

        -- Annual average from FEDRO official bulletin.
        annual_avg

    FROM raw.annual_results

    -- Filter: only the primary ADT metric row per station
    WHERE metric_type = 'ADT'
),

-- =============================================================================
-- STEP 2: Add FEDRO-defined data quality flags from the station notes table.
-- LEFT JOIN = include ALL stations, even those WITHOUT any noted gaps.
-- (A station not in station_notes is CLEAN — no known issues.)
-- =============================================================================

with_quality AS (
    SELECT
        b.*,

        -- Does this station have ANY gap noted in the quality file?
        -- COUNT(*) > 0 means at least one "No data" or "Winter closing" entry.
        CASE
            WHEN sn.gap_count > 0 THEN TRUE
            ELSE FALSE
        END AS has_data_gaps,

        -- Did any gap record mention winter closing? (alpine pass stations)
        COALESCE(sn.has_winter_close, FALSE) AS has_winter_closing,

        -- Were any months estimated/interpolated by FEDRO?
        COALESCE(sn.has_estimate, FALSE) AS has_estimated_months

    FROM base b
    LEFT JOIN (
        -- Aggregate gap notes per station in a subquery.
        -- We do this here so the main join stays clean.
        SELECT
            station_id,
            COUNT(*)                                             AS gap_count,
            MAX(CASE WHEN notes ILIKE '%winter%' THEN TRUE ELSE FALSE END) AS has_winter_close,
            MAX(CASE WHEN notes ILIKE '%estimated%' THEN TRUE ELSE FALSE END) AS has_estimate
        FROM raw.station_notes
        GROUP BY station_id
    ) sn ON b.station_id = sn.station_id
)

-- =============================================================================
-- FINAL SELECT: output all columns in logical order
-- =============================================================================

SELECT
    -- Identifiers
    station_id,
    station_name,
    station_name_clean,
    canton,
    road,

    -- Derived categorical features
    road_category,
    is_romandy,

    -- Data quality metrics
    months_with_data,
    has_data_gaps,
    has_winter_closing,
    has_estimated_months,

    -- The annual average ADT as reported by FEDRO (vehicles/day)
    annual_avg AS annual_adt

FROM with_quality
ORDER BY station_id
