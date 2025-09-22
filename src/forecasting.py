"""
forecasting.py
Time series forecasting using ARIMA (baseline) and deep LSTM (CPU-friendly).
Saves: artifacts/forecast_lstm.npy, artifacts/forecast_arima.csv, artifacts/forecast_metrics.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from preprocessing import load_spain_energy, ensure_dirs


def train_test_split_series(series: pd.Series, test_size: int = 24 * 3):
    """Split last `test_size` points as test."""
    series = series.dropna()
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    return train, test


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def arima_forecast(series, steps=24):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def lstm_forecast_train_eval(series, look_back=48, epochs=10, steps=24):
    """
    Train LSTM on train split, evaluate on last `steps` (test), then roll-forward forecast `steps`.
    """
    series = series.dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    def make_xy(data, lb):
        X, y = [], []
        for i in range(len(data) - lb):
            X.append(data[i:i + lb])
            y.append(data[i + lb])
        X = np.array(X)
        y = np.array(y)
        # Safety check for empty arrays
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {lb + 1} data points.")
    
        # Reshape X to 3D: (samples, timesteps, features) 
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
    
        return X, y

    # Split
    test_size = steps
    train_scaled = scaled[:-test_size]
    test_scaled = scaled[-(test_size + look_back):]  # we need look_back context before test

    X_train, y_train = make_xy(train_scaled, look_back)
    X_test, y_test = make_xy(test_scaled, look_back)

    # Build model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1, callbacks=[early_stop])

    # Evaluate on test slice
    test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    test_pred = scaler.inverse_transform(test_pred_scaled).flatten()
    test_true = scaler.inverse_transform(y_test).flatten()

    test_mape = mape(test_true, test_pred)
    test_rmse = rmse(test_true, test_pred)

    # Roll-forward forecast for next `steps` using last look_back window
    context = scaled[-look_back:].flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(context[-look_back:]).reshape((1, look_back, 1))
        yhat = model.predict(x_input, verbose=0)[0][0]
        preds.append(yhat)
        context.append(yhat)
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    return preds_inv, {"mape": test_mape, "rmse": test_rmse}


if __name__ == "__main__":
    ensure_dirs()
    df = load_spain_energy()
    if "real_demand" not in df.columns:
        raise ValueError("Expected 'real_demand' in Spain dataset.")

    demand_series = df["real_demand"]

    # ARIMA (baseline) forecast for 24h
    arima_out = arima_forecast(demand_series, steps=24)
    Path("artifacts").mkdir(exist_ok=True)
    pd.DataFrame({"arima_forecast": arima_out}).to_csv("artifacts/forecast_arima.csv", index=False)
    print("ARIMA forecast (first 5):")
    print(arima_out.head())

    # LSTM (train/eval + 24h forecast)
    lstm_out, metrics = lstm_forecast_train_eval(demand_series, look_back=48, epochs=10, steps=24)
    np.save("artifacts/forecast_lstm.npy", lstm_out)
    with open("artifacts/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nLSTM 24-step forecast (first 10):", lstm_out[:10])
    print("LSTM test metrics:", metrics)
