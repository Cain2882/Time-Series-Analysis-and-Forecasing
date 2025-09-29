import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

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
series = df[value_col].astype('float')

# Resample/ensure uniform frequency (daily)
series = series.asfreq('D').fillna(method='ffill')

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(series.values.reshape(-1,1))

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 60  # lookback window
X, y = create_sequences(scaled, SEQ_LEN)

# train/test split
h = 30
train_X, train_y = X[:-h], y[:-h]
test_X, test_y = X[-h:], y[-h:]

# Build LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[es], verbose=2)

# Predict next h steps using iterative forecasting
last_seq = scaled[-SEQ_LEN:]  # last available sequence
preds_scaled = []
current_seq = last_seq.reshape(1, SEQ_LEN, 1)

for i in range(h):
    p = model.predict(current_seq, verbose=0)[0,0]
    preds_scaled.append(p)
    # append p and slide window
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0, -1, 0] = p

preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()

# Prepare index for forecast
last_date = series.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')

# Plot real last portion and forecast
plt.figure(figsize=(10,5))
plt.plot(series[-200:].index, series[-200:].values, label='History (last 200)')
plt.plot(forecast_index, preds, label='LSTM Forecast')
plt.legend()
plt.title("LSTM Forecast")
plt.show()

out = pd.DataFrame({'date': forecast_index, 'yhat': preds})
out.to_csv("C:/Users/Cain Antony/lstm_forecast.csv", index=False)
print("Saved forecast to C:/Users/Cain Antony/lstm_forecast.csv")
