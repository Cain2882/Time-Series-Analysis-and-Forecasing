import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

file_path = "C:/Users/Cain Antony/Time Series Stock Forecasting/Apple Inc.xlsx"
xls = pd.ExcelFile(file_path)
df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

date_cols = [c for c in df_raw.columns if 'date' in c.lower() or 'time' in c.lower()]
value_cols = [c for c in df_raw.columns if c.lower() in ('close','adj close','adj_close','price','y')]

date_col = date_cols[0] if date_cols else df_raw.columns[0]
value_col = value_cols[0] if value_cols else df_raw.select_dtypes(include=[np.number]).columns[-1]

df = df_raw[[date_col, value_col]].dropna()
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)
series = df[value_col].asfreq('D').fillna(method='ffill')  # daily freq fallback

# Simple train/test
h = 30
train, test = series[:-h], series[-h:]

# Grid search for p,d,q (small ranges to limit runtime)
best_aic = np.inf
best_order = None
for p in range(0,3):
    for d in range(0,2):
        for q in range(0,3):
            try:
                model = ARIMA(train, order=(p,d,q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p,d,q)
            except Exception:
                continue

print("Best ARIMA order:", best_order, "AIC:", best_aic)

# Fit best model
model = ARIMA(train, order=best_order)
res = model.fit()
# Forecast
fc = res.get_forecast(steps=h)
mean_fc = fc.predicted_mean
conf_int = fc.conf_int()

# Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(mean_fc.index, mean_fc, label='Forecast')
plt.fill_between(mean_fc.index, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.2)
plt.legend()
plt.title(f"ARIMA{best_order} forecast")
plt.show()

# Save
out = pd.DataFrame({
    'date': mean_fc.index,
    'yhat': mean_fc.values,
    'lower': conf_int.iloc[:,0].values,
    'upper': conf_int.iloc[:,1].values
})
out.to_csv("C:/Users/Cain Antony/arima_forecast.csv", index=False)
print("Saved forecast to C:/Users/Cain Antony/arima_forecast.csv")
