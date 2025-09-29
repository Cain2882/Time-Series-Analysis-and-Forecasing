# -----------------------------
# 📊 Exploratory Data Analysis
# -----------------------------
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "Apple Inc.xlsx"
df = pd.read_excel(file_path, sheet_name="Apple Inc")

# -----------------------------
# 🔧 Fix numeric-format issues (e.g. "204,50" or "1.234,56")
# -----------------------------
def clean_number(value):
    """
    Convert values like '204,50', '1.234,56', '43,804,400', '$204,50' to a float.
    Keeps numeric values untouched. Returns np.nan when conversion fails.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    # Keep only digits, dot, comma, minus
    s = re.sub(r'[^\d\.,\-]', '', s)
    if s == '':
        return np.nan
    # If both '.' and ',' present -> assume '.' is thousands sep and ',' is decimal
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s and '.' not in s:
        # If the last group after comma has length <= 3, treat comma as decimal,
        # else treat commas as thousands separators
        if len(s.split(',')[-1]) <= 3:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    # else: only '.' present (standard decimal) -> leave as is
    try:
        return float(s)
    except:
        # fallback to pandas numeric coercion
        return pd.to_numeric(s, errors='coerce')

# Apply cleaning to numeric columns if they exist
numeric_columns = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
for col in numeric_columns:
    df[col] = df[col].apply(clean_number)

# Convert Volume to integer-like if possible (nullable Int64 to preserve NaNs)
if "Volume" in df.columns:
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("Int64")

# Convert Date to datetime format and sort
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")

# -----------------------------
# 🔎 Basic Info
# -----------------------------
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# -----------------------------
# 📈 Visualizations
# -----------------------------

# 1. Closing Price over time
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Close"], label="Closing Price")
plt.title("Apple Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# 2. Volume traded over time
plt.figure(figsize=(12,6))
plt.bar(df["Date"], df["Volume"])
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

# 3. Distribution of Daily Returns
df["Daily Return"] = df["Close"].pct_change()
plt.figure(figsize=(10,5))
sns.histplot(df["Daily Return"].dropna(), bins=50, kde=True)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[["Open","High","Low","Close","Adj Close","Volume"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Moving Averages
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.plot(df["Date"], df["MA50"], label="50-day MA")
plt.plot(df["Date"], df["MA200"], label="200-day MA")
plt.title("Apple Close Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
