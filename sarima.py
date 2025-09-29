import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
series = df[value_col].asfreq('D').fillna(method='ffill')

# If data is daily but has weekly seasonality set seasonal_period=7; if monthly -> 12
seasonal_period = 7

h = 30
train, test = series[:-h], series[-h:]

# search small grid for (p,d,q) x (P,D,Q,s)
best_aic = np.inf
best_cfg = None
for p in range(0,2):
    for d in range(0,2):
        for q in range(0,2):
            for P in range(0,2):
                for D in range(0,2):
                    for Q in range(0,2):
                        try:
                            mod = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,seasonal_period),
                                          enforce_stationarity=False, enforce_invertibility=False)
                            res = mod.fit(disp=False)
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_cfg = (p,d,q,P,D,Q,seasonal_period)
                        except Exception:
                            continue

print("Best SARIMA config:", best_cfg, "AIC:", best_aic)

# Fit final model
mod = SARIMAX(train, order=best_cfg[:3], seasonal_order=best_cfg[3:], 
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False)

pred = res.get_forecast(steps=h)
mean_pred = pred.predicted_mean
ci = pred.conf_int()

plt.figure(figsize=(10,5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(mean_pred.index, mean_pred, label='SARIMA Forecast')
plt.fill_between(mean_pred.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2)
plt.legend()
plt.title(f"SARIMA{best_cfg[:3]}x{best_cfg[3:6]} forecasts (s={seasonal_period})")
plt.show()

out = pd.DataFrame({'date': mean_pred.index, 'yhat': mean_pred.values, 'lower': ci.iloc[:,0].values, 'upper': ci.iloc[:,1].values})
out.to_csv("C:/Users/Cain Antony/sarima_forecast.csv", index=False)
print("Saved forecast to C:/Users/Cain Antony/sarima_forecast.csv")
