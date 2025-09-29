"""
insights_code.py
Auto-generated exploratory insights script for the dataset in 'Apple Inc.xlsx'.
- This script calculates summary statistics, correlations, missing data report, and basic trend detection.
- Run with: python insights_code.py
"""
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:/Users/Cain Antony/Time Series Stock Forecasting/Apple Inc.xlsx", sheet_name=0)

def print_section(title):
    print("\n" + "="*30)
    print(title)
    print("="*30 + "\n")

print_section("Basic info")
print(df.info())

print_section("Missing values per column")
print(df.isnull().sum())

print_section("Summary statistics (numeric columns)")
print(df.describe(include=[np.number]).T)

# Correlation matrix (numeric)
if not df.select_dtypes(include=[np.number]).empty:
    print_section("Correlation matrix (Pearson)")
    corr = df.select_dtypes(include=[np.number]).corr()
    print(corr)

# Identify potential anomalies: z-score > 3
from scipy import stats
numeric = df.select_dtypes(include=[np.number])
if not numeric.empty:
    zscores = np.abs(stats.zscore(numeric.dropna()))
    if zscores.size:
        anomaly_rows = (zscores > 3).any(axis=1)
        print_section("Anomaly detection (z-score > 3) - sample rows")
        print(df.loc[numeric.dropna().index[anomaly_rows]].head(10))

# If there's a date column, compute monthly aggregates for numeric columns
for c in df.columns:
    if 'date' in c.lower() or 'time' in c.lower():
        try:
            df[c] = pd.to_datetime(df[c])
            datecol = c
            print_section(f"Time-based aggregates using {datecol}")
            df2 = df.set_index(datecol)
            monthly = df2.resample('M').mean(numeric_only=True)
            print("Monthly means (first 10 rows):")
            print(monthly.head(10))
            break
        except:
            pass

print_section("Top value counts for object columns")
for col in df.select_dtypes(include=['object','category']).columns:
    print("\nColumn:", col)
    print(df[col].value_counts().nlargest(20))
