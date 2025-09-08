"""
main.py
Pipeline runner: forecasting -> stability -> optimization -> RL controller -> evaluation
"""

from preprocessing import ensure_dirs, load_spain_energy
from forecasting import lstm_forecast_train_eval
from stability_classifier import train_stability_classifier
from optimization import run_optimization_from_forecast
from rl_controller import simulate_controller
from evaluation import summarize
import numpy as np
import json


def main():
    ensure_dirs()

    # 1) Forecasting
    df = load_spain_energy()
    demand_series = df["real_demand"].dropna()
    lstm_out, metrics = lstm_forecast_train_eval(demand_series, look_back=48, epochs=10, steps=24)
    np.save("artifacts/forecast_lstm.npy", lstm_out)
    with open("artifacts/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 2) Stability classification
    train_stability_classifier()

    # 3) Optimization (dispatch over the LSTM forecast)
    run_optimization_from_forecast("artifacts/forecast_lstm.npy")

    # 4) RL controller (adjusts dispatch dynamically)
    arr = np.load("artifacts/forecast_lstm.npy")
    simulate_controller(arr)

    # 5) Evaluation summary
    summarize()


if __name__ == "__main__":
    main()
