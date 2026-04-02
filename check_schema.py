import duckdb
con = duckdb.connect("traffic.duckdb")
cols = con.sql("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='raw' AND table_name='annual_results' ORDER BY ordinal_position").df()
print(cols.to_string())
print()
print("First row columns:")
row = con.sql("SELECT * FROM raw.annual_results LIMIT 1").df()
print(list(row.columns))
con.close()
