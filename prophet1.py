import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# --- Load Excel ---
file_path = "C:/Users/Cain Antony/Time Series Stock Forecasting/Apple Inc.xlsx"
# If sheet name unknown, read the first sheet
xls = pd.ExcelFile(file_path)
df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

# --- Try to detect date and value columns ---
date_cols = [c for c in df_raw.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()]
value_cols = [c for c in df_raw.columns if c.lower() in ('close','adj close','adj_close','price','close_price','y')]

if len(date_cols) == 0:
    # fallback: first column
    date_col = df_raw.columns[0]
else:
    date_col = date_cols[0]

if len(value_cols) == 0:
    # fallback: last numeric column
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found for forecasting target.")
    value_col = numeric_cols[-1]
else:
    value_col = value_cols[0]

df = df_raw[[date_col, value_col]].dropna()
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

# Prophet expects 'ds' and 'y'
prophet_df = df.rename(columns={date_col: 'ds', value_col: 'y'}).reset_index(drop=True)

# --- Train / test split (keep last 30 days for testing if daily)
horizon = 30  # forecast horizon
train = prophet_df.iloc[:-horizon].copy()
future_df = prophet_df.iloc[:-horizon].copy()

# --- Fit model ---
m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
m.fit(train)

# --- Make future dataframe and forecast ---
future = m.make_future_dataframe(periods=horizon, freq='D')  # adapt freq if you have business days or months
forecast = m.predict(future)

# --- Plot ---
fig1 = m.plot(forecast)
plt.title("Prophet forecast")
plt.show()

fig2 = m.plot_components(forecast)
plt.show()

# --- Save forecast results ---
out = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon)
out.to_csv("C:/Users/Cain Antony/prophet_forecast.csv", index=False)
print("Saved forecast to C:/Users/Cain Antony/prophet_forecast.csv")
