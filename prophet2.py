import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Try fbprophet, fallback to prophet
try:
    from fbprophet import Prophet
except ImportError:
    from prophet import Prophet

file_path = "C:/Users/Cain Antony/Time Series Stock Forecasting/Apple Inc.xlsx"
xls = pd.ExcelFile(file_path)
df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

date_cols = [c for c in df_raw.columns if 'date' in c.lower()]
value_cols = [c for c in df_raw.columns if c.lower() in ('close','adj close','adj_close','price','y')]

date_col = date_cols[0] if date_cols else df_raw.columns[0]
value_col = value_cols[0] if value_cols else df_raw.select_dtypes(include=[np.number]).columns[-1]

df = df_raw[[date_col, value_col]].dropna()
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)
df_prophet = df.rename(columns={date_col:'ds', value_col:'y'}).reset_index(drop=True)

h = 30
train = df_prophet.iloc[:-h]

m = Prophet(daily_seasonality=True, yearly_seasonality=True)
m.fit(train)
future = m.make_future_dataframe(periods=h, freq='D')
forecast = m.predict(future)

m.plot(forecast)
plt.title("Prophet forecast")
plt.show()

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(h).to_csv(
    "C:/Users/Cain Antony/fbprophet_forecast.csv", index=False
)
print("Saved forecast to C:/Users/Cain Antony/fbprophet_forecast.csv")
