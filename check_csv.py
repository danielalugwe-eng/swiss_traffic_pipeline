import pandas as pd
df = pd.read_csv('Annual_results_2025.csv', skiprows=5, header=0, dtype=str,
                 keep_default_na=False, encoding='latin-1', nrows=2)
print('MAIN CSV COLUMNS:')
for i, c in enumerate(df.columns):
    print(i, repr(c))
print()
print('First data row:')
print(df.iloc[0].to_dict())
