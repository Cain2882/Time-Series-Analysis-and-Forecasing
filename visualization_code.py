"""
visualization_code.py
Auto-generated visualization script for the dataset in 'Apple Inc.xlsx'.
Instructions:
- This script uses pandas and matplotlib only.
- Each chart is plotted in its own figure (no subplots) as requested.
- It does not set custom colors.
- Run with: python visualization_code.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load the workbook's first sheet by default
df = pd.read_excel(r"C:/Users/Cain Antony/Time Series Stock Forecasting/Apple Inc.xlsx", sheet_name=0)

def save_or_show(fig, fname):
    # Save figure to file and also show
    fig.savefig(fname, bbox_inches='tight')
    print(f"Saved: {fname}")

# Quick info
print("Columns detected:", df.columns.tolist())
print("Dtypes:\n", df.dtypes)

# --- Example plots generated based on detected columns ---

# Ensure Date column is datetime and sort
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df_sorted = df.sort_values(by="Date")

    # Time series plots for selected columns
    for col in ['Open', 'Close', 'Volume']:
        if col in df.columns:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(df_sorted["Date"].values, df_sorted[col].values)
            ax.set_title(f"Time series: {col} vs Date")
            ax.set_xlabel("Date")
            ax.set_ylabel(col)
            fig.autofmt_xdate()
            save_or_show(fig, f"plot_{col}_vs_Date.png")

# Bar charts for top values in numerical columns (frequency counts)
for col in ['High', 'Low', 'Adj Close']:
    if col in df.columns:
        top = df[col].value_counts().nlargest(10)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(top.index.astype(str), top.values)
        ax.set_title(f"Top categories in {col} (top 10)")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        save_or_show(fig, f"bar_top_{col}.png")

print("✅ All visualizations generated and saved successfully.")
