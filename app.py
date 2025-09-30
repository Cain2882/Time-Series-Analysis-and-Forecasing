# app_with_more_forecasts.py
# Original app expanded with SARIMA, Prophet1, Prophet2, and LSTM forecasting modules.
# Designed to be optional-deps friendly: heavy libraries are imported inside functions with fallbacks.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import io

warnings.filterwarnings("ignore")

# Try to import seaborn for better plots (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# -------------------------
# Utility: Clean numeric values
# -------------------------
def clean_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    s = re.sub(r'[^\d\.,\-]', '', s)
    if s == '':
        return np.nan
    # handle thousand separators / decimal commas
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s and '.' not in s:
        # guess: comma as decimal if last group length <=3
        if len(s.split(',')[-1]) <= 3:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    try:
        return float(s)
    except:
        return pd.to_numeric(s, errors='coerce')


# -------------------------
# Main App Function
# -------------------------
def main():
    st.set_page_config(page_title="📊 Stock Analysis & Forecasting App", layout="wide")
    st.title("📊 Stock Analysis & Forecasting App (lightweight)")

    uploaded_file = st.file_uploader("Upload Excel file (sheet with Date & Close recommended)", type=["xlsx", "xls", "csv"])
    if not uploaded_file:
        st.info("👆 Upload an Excel/CSV file to begin. Example columns: Date, Open, High, Low, Close, Adj Close, Volume")
        return

    # Load file (Excel or CSV)
    try:
        if str(uploaded_file.name).lower().endswith((".xls", ".xlsx")):
            df_raw = pd.read_excel(uploaded_file, sheet_name=0)
        else:
            df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Clean numeric columns if present
    numeric_columns = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df_raw.columns]
    for col in numeric_columns:
        df_raw[col] = df_raw[col].apply(clean_number)

    # Volume to nullable int if exists
    if "Volume" in df_raw.columns:
        df_raw["Volume"] = pd.to_numeric(df_raw["Volume"], errors="coerce").astype("Int64")

    # Date parsing
    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
        df_raw = df_raw.sort_values("Date")

    # -------------------------
    # Sidebar Navigation
    # -------------------------
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Select a module",
        [
            "EDA",
            "Insights",
            "ARIMA Forecasting",
            "SARIMA Forecasting",
            "Prophet1 Forecasting",
            "Prophet2 Forecasting",
            "LSTM Forecasting",
            "Linear Regression Model",
            "Visualizations",
        ],
    )

    # -------------------------
    # EDA Section
    # -------------------------
    if choice == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Dataset shape:", df_raw.shape)
        st.write("Columns:", df_raw.columns.tolist())
        st.write("Missing values:", df_raw.isnull().sum())

        if "Close" in df_raw.columns and "Date" in df_raw.columns:
            st.subheader("Closing Price Over Time")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_raw["Date"], df_raw["Close"], label="Closing Price")
            ax.set_title("Closing Price Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

        if "Volume" in df_raw.columns and "Date" in df_raw.columns:
            st.subheader("Trading Volume Over Time")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(df_raw["Date"], df_raw["Volume"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Volume")
            st.pyplot(fig)

        if "Close" in df_raw.columns:
            df_raw["Daily Return"] = df_raw["Close"].pct_change()
            st.subheader("Distribution of Daily Returns")
            fig, ax = plt.subplots()
            if HAS_SEABORN:
                sns.histplot(df_raw["Daily Return"].dropna(), bins=50, kde=True, ax=ax)
            else:
                ax.hist(df_raw["Daily Return"].dropna(), bins=50)
                ax.set_xlabel("Daily Return")
            st.pyplot(fig)

        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols = [c for c in cols if c in df_raw.columns]
        if len(cols) >= 2:
            st.subheader("Correlation Heatmap")
            corr = df_raw[cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            if HAS_SEABORN:
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            else:
                cax = ax.matshow(corr, cmap="RdYlBu")
                fig.colorbar(cax)
                ax.set_xticks(range(len(cols)))
                ax.set_xticklabels(cols, rotation=45)
                ax.set_yticks(range(len(cols)))
                ax.set_yticklabels(cols)
                for (i, j), val in np.ndenumerate(corr.values):
                    ax.text(j, i, f"{val:.2f}", va='center', ha='center')
            st.pyplot(fig)

    # -------------------------
    # Insights Section
    # -------------------------
    elif choice == "Insights":
        st.header("Exploratory Insights")

        buf = io.StringIO()
        df_raw.info(buf=buf)
        st.text(buf.getvalue())

        st.subheader("Missing Values per Column")
        st.write(df_raw.isnull().sum())

        st.subheader("Summary Statistics")
        st.write(df_raw.describe(include=[np.number]).T)

        st.subheader("Correlation Matrix")
        if not df_raw.select_dtypes(include=[np.number]).empty:
            st.write(df_raw.corr())

        # Anomaly detection (z-score via numpy)
        numeric = df_raw.select_dtypes(include=[np.number])
        if not numeric.empty:
            zscores = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
            anomaly_rows = (zscores > 3).any(axis=1)
            if anomaly_rows.any():
                st.subheader("Anomaly detection (z-score > 3) — showing up to 10 rows")
                st.write(df_raw.loc[anomaly_rows].head(10))
            else:
                st.info("No numeric anomalies found (z-score > 3).")

    # -------------------------
    # ARIMA Forecasting (optional library)
    # -------------------------
    elif choice == "ARIMA Forecasting":
        run_arima(df_raw)

    # -------------------------
    # SARIMA Forecasting (optional)
    # -------------------------
    elif choice == "SARIMA Forecasting":
        run_sarima(df_raw)

    # -------------------------
    # Prophet1 Forecasting (optional)
    # -------------------------
    elif choice == "Prophet1 Forecasting":
        run_prophet1(df_raw)

    # -------------------------
    # Prophet2 Forecasting (optional, different seasonalities / changepoints)
    # -------------------------
    elif choice == "Prophet2 Forecasting":
        run_prophet2(df_raw)

    # -------------------------
    # LSTM Forecasting (optional)
    # -------------------------
    elif choice == "LSTM Forecasting":
        run_lstm(df_raw)

    # -------------------------
    # Linear Regression (lightweight fallback)
    # -------------------------
    elif choice == "Linear Regression Model":
        run_linear_regression(df_raw)

    elif choice == "Visualizations":
        st.header("Visualizations")
        st.write("Use EDA -> Visualizations above. For custom visualizations, add code or install optional `visualization_code` module.")


# -------------------------
# ARIMA helper (optional statsmodels)
# -------------------------
def run_arima(df_raw):
    st.header("ARIMA Time Series Forecasting (optional)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("ARIMA needs 'Date' and 'Close' columns.")
        return

    series = df_raw.set_index("Date")["Close"].dropna()
    periods = int(st.number_input("Forecast periods (days)", min_value=1, value=30))

    # Try statsmodels ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        order_input = st.text_input("ARIMA order p,d,q (comma separated)", value="5,1,0")
        p, d, q = [int(x.strip()) for x in order_input.split(",")]
        with st.spinner("Fitting ARIMA model (statsmodels)..."):
            model = ARIMA(series, order=(p, d, q)).fit()
            forecast = model.forecast(steps=periods)
            # ensure forecast has a datetime index
            try:
                forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
                forecast = pd.Series(forecast, index=forecast_index)
            except Exception:
                forecast = pd.Series(forecast)
            st.success("ARIMA forecast complete (statsmodels).")
    except Exception as e:
        st.warning("statsmodels not available or ARIMA failed. Falling back to simple moving-average forecast. (Install statsmodels to enable ARIMA).")
        # Moving-average fallback
        window = min(10, max(1, len(series) // 10))
        last_mean = series.tail(window).mean()
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast = pd.Series([last_mean] * periods, index=forecast_index)

    # Plot observed + forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, label="Observed")
    ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
    ax.set_title("Observed vs Forecast")
    ax.legend()
    st.pyplot(fig)

    # Show forecast table
    st.subheader("Forecast values")
    st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))


# -------------------------
# SARIMA helper (optional statsmodels.statespace)
# -------------------------
def run_sarima(df_raw):
    st.header("SARIMA Forecasting (optional)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("SARIMA needs 'Date' and 'Close' columns.")
        return

    series = df_raw.set_index("Date")["Close"].dropna()
    periods = int(st.number_input("Forecast periods (days) for SARIMA", min_value=1, value=30))

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order_text = st.text_input("Non-seasonal order (p,d,q)", value="1,1,1")
        seasonal_text = st.text_input("Seasonal order (P,D,Q,s) e.g. 1,1,1,12", value="0,1,1,12")
        p, d, q = [int(x.strip()) for x in order_text.split(",")]
        P, D, Q, s = [int(x.strip()) for x in seasonal_text.split(",")]
        with st.spinner("Fitting SARIMA (statsmodels.statespace.sarimax)..."):
            model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=periods)
            forecast = pred.predicted_mean
            try:
                forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
                forecast = pd.Series(forecast.values, index=forecast_index)
            except Exception:
                forecast = pd.Series(forecast.values)
            st.success("SARIMA forecasting complete.")
    except Exception as e:
        st.warning("statsmodels SARIMAX not available or fit failed. Falling back to seasonal moving-average forecast.")
        # Seasonal moving-average fallback using last seasonal cycle if possible
        freq = 7
        if len(series) >= freq:
            last_cycle = series.tail(freq).mean()
        else:
            last_cycle = series.tail(10).mean()
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast = pd.Series([last_cycle] * periods, index=forecast_index)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, label="Observed")
    ax.plot(forecast.index, forecast.values, linestyle="--", label="SARIMA Forecast")
    ax.set_title("SARIMA Observed vs Forecast")
    ax.legend()
    st.pyplot(fig)

    st.subheader("SARIMA Forecast values")
    st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))


# -------------------------
# Prophet1 helper (optional: prophet or fbprophet)
# -------------------------
def run_prophet1(df_raw):
    st.header("Prophet Forecasting — Model 1 (optional)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("Prophet needs 'Date' and 'Close' columns.")
        return

    periods = int(st.number_input("Forecast periods (days) for Prophet1", min_value=1, value=30))
    cap = None

    # Prepare df for Prophet
    df = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
    # If Prophet installed, use it; else fallback
    try:
        try:
            # new package name
            from prophet import Prophet
        except Exception:
            # older package name
            from fbprophet import Prophet

        # simple model
        model = Prophet(daily_seasonality=True)
        with st.spinner("Fitting Prophet model (this may take a moment)..."):
            model.fit(df)
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            # extract only the future part
            future_forecast = forecast.set_index('ds')['yhat'].tail(periods)
            st.success("Prophet forecast complete.")
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.set_index('ds').index, df.set_index('ds')['y'], label='Observed')
            ax.plot(future_forecast.index, future_forecast.values, linestyle='--', label='Prophet Forecast')
            ax.set_title("Prophet Model 1: Observed vs Forecast")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Prophet Forecast values (last {} rows)".format(periods))
            st.write(future_forecast.rename("Forecast").to_frame().reset_index().rename(columns={"ds": "Date"}))
    except Exception as e:
        st.warning("Prophet not available. Falling back to naive persistence forecast.")
        series = df.set_index('ds')['y']
        last_val = series.dropna().iloc[-1]
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast = pd.Series([last_val] * periods, index=forecast_index)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series.values, label="Observed")
        ax.plot(forecast.index, forecast.values, linestyle="--", label="Naive Forecast")
        ax.legend()
        st.pyplot(fig)
        st.subheader("Forecast values")
        st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))


# -------------------------
# Prophet2 helper (optional) — different defaults (e.g., yearly seasonality + changepoints)
# -------------------------
def run_prophet2(df_raw):
    st.header("Prophet Forecasting — Model 2 (optional, different seasonal settings)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("Prophet needs 'Date' and 'Close' columns.")
        return

    periods = int(st.number_input("Forecast periods (days) for Prophet2", min_value=1, value=30))

    df = df_raw[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})

    try:
        try:
            from prophet import Prophet
        except Exception:
            from fbprophet import Prophet

        # Model 2: yearly seasonality + more changepoints
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, n_changepoints=50)
        with st.spinner("Fitting Prophet model 2..."):
            model.fit(df)
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            future_forecast = forecast.set_index('ds')['yhat'].tail(periods)
            st.success("Prophet2 forecast complete.")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.set_index('ds').index, df.set_index('ds')['y'], label='Observed')
            ax.plot(future_forecast.index, future_forecast.values, linestyle='--', label='Prophet2 Forecast')
            ax.set_title("Prophet Model 2: Observed vs Forecast")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Prophet2 Forecast values")
            st.write(future_forecast.rename("Forecast").to_frame().reset_index().rename(columns={"ds": "Date"}))
    except Exception as e:
        st.warning("Prophet not available. Falling back to simple moving-average forecast.")
        series = df.set_index('ds')['y']
        window = min(20, max(1, len(series) // 10))
        last_mean = series.tail(window).mean()
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast = pd.Series([last_mean] * periods, index=forecast_index)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series.values, label='Observed')
        ax.plot(forecast.index, forecast.values, linestyle='--', label='Fallback Forecast')
        ax.legend()
        st.pyplot(fig)
        st.subheader("Forecast values")
        st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))


# -------------------------
# LSTM helper (optional: tensorflow.keras), lightweight fallback
# -------------------------
def run_lstm(df_raw):
    st.header("LSTM Forecasting (optional TensorFlow/Keras)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("LSTM needs 'Date' and 'Close' columns.")
        return

    df = df_raw[["Date", "Close"]].dropna().set_index("Date").asfreq('D').fillna(method='ffill')
    series = df['Close'].astype(float)

    forecast_days = int(st.number_input("Forecast days for LSTM", min_value=1, value=30))
    lookback = int(st.number_input("LSTM lookback (timesteps)", min_value=1, value=10))
    test_fraction = float(st.slider("Test fraction for LSTM", 0.05, 0.5, 0.2))

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        # Prepare data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

        def make_sequences(data, lookback):
            X, y = [], []
            for i in range(len(data) - lookback):
                X.append(data[i:i+lookback])
                y.append(data[i+lookback])
            return np.array(X), np.array(y)

        X, y = make_sequences(scaled, lookback)
        if len(X) < 5:
            st.error("Not enough data points for the chosen lookback. Reduce lookback or provide more data.")
            return

        split = int(len(X) * (1 - test_fraction))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(32, input_shape=(lookback, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        with st.spinner("Training LSTM (this may take time depending on your machine)..."):
            model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # forecasting iteratively
        last_seq = scaled[-lookback:].tolist()
        preds_scaled = []
        for _ in range(forecast_days):
            x_in = np.array(last_seq[-lookback:]).reshape((1, lookback, 1))
            p = model.predict(x_in, verbose=0)[0][0]
            preds_scaled.append(p)
            last_seq.append(p)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        forecast = pd.Series(preds, index=forecast_index)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series.values, label='Observed')
        ax.plot(forecast.index, forecast.values, linestyle='--', label='LSTM Forecast')
        ax.set_title("LSTM Observed vs Forecast")
        ax.legend()
        st.pyplot(fig)

        st.subheader("LSTM Forecast values")
        st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))

    except Exception as e:
        st.warning("TensorFlow/Keras or sklearn not available or LSTM failed. Falling back to simple linear extrapolation.")
        # fallback: linear trend extrapolation using numpy.polyfit on recent window
        window = min(60, len(series))
        y_recent = series.tail(window)
        x = np.arange(len(y_recent))
        if len(x) < 2:
            st.error("Not enough data for fallback forecasting.")
            return
        coeffs = np.polyfit(x, y_recent.values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        future_x = np.arange(len(x), len(x) + forecast_days)
        preds = slope * future_x + intercept
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        forecast = pd.Series(preds, index=forecast_index)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series.values, label='Observed')
        ax.plot(forecast.index, forecast.values, linestyle='--', label='Fallback Linear Forecast')
        ax.legend()
        st.pyplot(fig)
        st.subheader("Fallback Forecast values")
        st.write(forecast.rename("Forecast").to_frame().reset_index().rename(columns={"index": "Date"}))


# -------------------------
# Linear regression helper (sklearn optional, else numpy polyfit)
# -------------------------
def run_linear_regression(df_raw):
    st.header("Linear Regression Stock Price Prediction (date -> close)")
    if "Date" not in df_raw.columns or "Close" not in df_raw.columns:
        st.error("This module requires 'Date' and 'Close' columns.")
        return

    # Prepare data: convert date to ordinal
    df = df_raw[["Date", "Close"]].dropna().copy()
    df["x"] = df["Date"].map(pd.Timestamp.toordinal)
    X = df["x"].values.reshape(-1, 1)
    y = df["Close"].values

    test_size = float(st.slider("Test size (fraction)", 0.05, 0.5, 0.2))
    forecast_days = int(st.number_input("Forecast days (after last date)", min_value=1, value=30))

    # Try sklearn
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        st.write(f"R^2 on test set: {score:.4f}")
        # Forecast
        last_x = df["x"].iloc[-1]
        future_index = np.arange(last_x + 1, last_x + forecast_days + 1).reshape(-1, 1)
        future_preds = model.predict(future_index)
    except Exception:
        st.warning("scikit-learn not available. Using numpy.polyfit fallback (no train/test split).")
        coeffs = np.polyfit(df["x"].astype(float), df["Close"].astype(float), deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        st.write(f"Linear fit slope={slope:.6f}, intercept={intercept:.2f}")
        last_x = df["x"].iloc[-1]
        future_index = np.arange(last_x + 1, last_x + forecast_days + 1)
        future_preds = slope * future_index + intercept

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df["Close"], label="Observed")
    future_dates = [pd.Timestamp.fromordinal(int(x)) for x in future_index.flatten()]
    ax.plot(future_dates, future_preds, linestyle="--", label="Forecast")
    ax.set_title("Linear regression forecast (date -> close)")
    ax.legend()
    st.pyplot(fig)

    out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})
    st.subheader("Predicted future prices")
    st.write(out_df)

# -------------------------
# Run only if executed
# -------------------------
if __name__ == "__main__":
    main()
