# src/streamlit_app.py
"""
SmartGrid-AI — Streamlit Dashboard
Usage:
    streamlit run src/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Local project imports
from preprocessing import load_spain_energy, ensure_dirs
from forecasting import arima_forecast, lstm_forecast_train_eval
from optimization import optimize_dispatch_dynamic
from evaluation import summarize as eval_summarize

import plotly.graph_objects as go

# ========== Setup ==========
ensure_dirs()
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

st.set_page_config(page_title="SmartGrid-AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# ========== Helper Functions ==========
@st.cache_data(ttl=600)
def load_packaged_spain():
    try:
        return load_spain_energy()
    except Exception as e:
        st.error(f"Failed to load packaged Spain dataset: {e}")
        return pd.DataFrame()

def try_load_forecast():
    if (ARTIFACTS / "forecast_lstm.npy").exists():
        return np.load(ARTIFACTS / "forecast_lstm.npy"), "lstm"
    if (ARTIFACTS / "forecast_arima.csv").exists():
        df = pd.read_csv(ARTIFACTS / "forecast_arima.csv")
        col = df.select_dtypes(include='number').columns[0]
        return df[col].values, "arima"
    return None, None

def plot_series_with_forecast(history_series, forecast, title="Demand & Forecast", key=None):
    if history_series is None or history_series.empty:
        st.warning("No historical series available to plot.")
        return

    hist = history_series.reset_index(drop=True).astype(float)
    H = len(forecast)
    x_hist = np.arange(len(hist))
    x_fore = np.arange(len(hist), len(hist) + H)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=hist, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=x_fore, y=forecast, mode="lines+markers", name="Forecast"))
    fig.update_layout(title=title, xaxis_title="Time index", yaxis_title="Demand")

    st.plotly_chart(fig, use_container_width=True, key=key or f"plot_{np.random.randint(1e9)}")

def plot_dispatch(df_dispatch, key=None):
    if df_dispatch is None or df_dispatch.empty:
        st.warning("No dispatch data to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_dispatch.hour, y=df_dispatch.renewable, name="Renewable"))
    fig.add_trace(go.Bar(x=df_dispatch.hour, y=df_dispatch.thermal, name="Thermal"))
    fig.add_trace(go.Bar(x=df_dispatch.hour, y=df_dispatch.shed, name="Shed"))
    fig.update_layout(barmode='stack', title="Hourly Dispatch", xaxis_title="Hour", yaxis_title="Energy")

    st.plotly_chart(fig, use_container_width=True, key=key or f"plot_{np.random.randint(1e9)}")

# ========== Sidebar Controls ==========
st.sidebar.title("⚡ SmartGrid-AI")
st.sidebar.write("Forecasting & Dispatch Optimization")

data_option = st.sidebar.radio("Select data source:", ["Spain dataset", "Upload CSV", "Manual 24h demand"])
forecast_mode = st.sidebar.selectbox("Forecasting method:", ["Load existing", "ARIMA", "LSTM"])
run_pipeline = st.sidebar.button("Run Full Pipeline")

questions = [
    "What will the demand look like for the next 24 hours?",
    "How accurate was our last forecast?",
    "What is the optimal dispatch plan for tomorrow?",
    "What happens if renewable capacity is doubled?",
    "How much load shedding if demand spikes 10%?",
    "Summarize overall system performance"
]
choice = st.sidebar.selectbox("Ask a predefined question:", questions)

st.sidebar.write("---")
st.sidebar.write("Download artifacts:")
if (ARTIFACTS / "forecast_lstm.npy").exists():
    st.sidebar.download_button("Download LSTM forecast", data=open(ARTIFACTS / "forecast_lstm.npy", "rb"), file_name="forecast_lstm.npy")
if (ARTIFACTS / "dispatch_plan_dynamic.csv").exists():
    st.sidebar.download_button("Download Dispatch Plan", data=open(ARTIFACTS / "dispatch_plan_dynamic.csv", "rb"), file_name="dispatch_plan_dynamic.csv")

# ========== Main Layout ==========
st.title("SmartGrid-AI — Interactive Forecast & Dispatch Studio")

col1, col2 = st.columns([2, 1])

# --- Data Load ---
with col1:
    st.header("Data")
    if data_option == "Spain dataset":
        df = load_packaged_spain()
        st.write("Using packaged Spain dataset")
    elif data_option == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write("Custom dataset uploaded")
        else:
            df = pd.DataFrame()
    else:  # Manual
        st.info("Enter 24 hourly demand values")
        cols = st.columns(6)
        manual_vals = []
        for i in range(24):
            val = cols[i % 6].number_input(f"H{i}", value=float(50 + 30*np.sin((i/24)*2*np.pi)), key=f"h{i}")
            manual_vals.append(val)
        df = pd.DataFrame({"hour": np.arange(24), "real_demand": manual_vals})

    if not df.empty:
        st.dataframe(df.head())

# --- Forecast ---
    st.header("Forecasting")
    forecast_arr, forecast_type = try_load_forecast()

    if forecast_mode == "ARIMA":
        if st.button("Run ARIMA Forecast"):
            if "real_demand" in df.columns:
                series = df["real_demand"].dropna()
                arr = arima_forecast(series, steps=24)
                forecast_arr = np.array(arr)
                pd.DataFrame({"forecast": forecast_arr}).to_csv(ARTIFACTS / "forecast_arima.csv", index=False)
                st.success("ARIMA forecast generated")
    elif forecast_mode == "LSTM":
        epochs = st.slider("LSTM epochs", 1, 50, 10)
        look_back = st.number_input("LSTM look_back", 6, 168, 48)
        if st.button("Train LSTM Forecast"):
            if "real_demand" in df.columns:
                series = df["real_demand"].dropna()
                preds, metrics = lstm_forecast_train_eval(series, look_back=look_back, epochs=epochs, steps=24)
                np.save(ARTIFACTS / "forecast_lstm.npy", preds)
                with open(ARTIFACTS / "forecast_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                forecast_arr = preds
                st.success(f"LSTM forecast generated (metrics: {metrics})")

    if forecast_arr is not None:
        if "real_demand" in df.columns:
            hist_series = df["real_demand"]
        else:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                hist_series = df[num_cols[0]]
            else:
                st.warning("No numeric column found in dataset for forecasting.")
                hist_series = pd.Series(dtype=float)
        plot_series_with_forecast(hist_series, forecast_arr, f"Forecast ({forecast_type}) vs history", key="main_forecast_plot")

# --- Problem → Solution ---
with col2:
    st.header("Problem → Solution")
    problem_text = st.text_area("Describe your problem", placeholder="E.g., demand spike, optimize renewables, etc.")
    if st.button("Generate Solution"):
        if forecast_arr is not None:
            df_dispatch, obj = optimize_dispatch_dynamic(forecast_arr, thermal_cost=1.0,
                                                         renewable_cost=0.2, shed_penalty=200.0,
                                                         iters=200, particles=80)
            st.write(f"Objective value: {obj:.2f}")
            plot_dispatch(df_dispatch, key="solution_dispatch")
        else:
            st.warning("No forecast available. Please run forecasting first.")

    st.header("Custom Optimization")
    custom_demand_input = st.text_area("Enter 24 values (comma separated)")
    if st.button("Run Custom Optimization"):
        try:
            arr = np.array([float(x) for x in custom_demand_input.split(",")])
            if arr.size == 24:
                df_dispatch, obj = optimize_dispatch_dynamic(arr, thermal_cost=1.0,
                                                             renewable_cost=0.2, shed_penalty=200.0,
                                                             iters=200, particles=80)
                st.write(f"Custom optimization complete (Objective={obj:.2f})")
                plot_dispatch(df_dispatch, key="custom_dispatch")
            else:
                st.error("Please enter exactly 24 values")
        except Exception as e:
            st.error(f"Invalid input: {e}")

# --- Q&A Section ---
st.header("Predefined Questions")
if choice == questions[0]:
    st.subheader("Forecast Next 24 Hours")
    if forecast_arr is not None:
        if "real_demand" in df.columns:
            hist_series = df["real_demand"]
        else:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                hist_series = df[num_cols[0]]
            else:
                hist_series = pd.Series(dtype=float)

        plot_series_with_forecast(hist_series, forecast_arr, "Next 24h Demand Forecast", key="q_forecast")
elif choice == questions[1]:
    st.subheader("Forecast Accuracy")
    metrics_file = ARTIFACTS / "forecast_metrics.json"
    if metrics_file.exists():
        st.json(json.load(open(metrics_file)))
    else:
        st.info("No metrics found. Train LSTM first.")
elif choice == questions[2]:
    st.subheader("Optimal Dispatch Plan")
    if forecast_arr is not None:
        df_dispatch, obj = optimize_dispatch_dynamic(forecast_arr, thermal_cost=1.0,
                                                     renewable_cost=0.2, shed_penalty=200.0,
                                                     iters=200, particles=80)
        plot_dispatch(df_dispatch, key="q_dispatch_optimal")
        st.write(f"Objective={obj:.2f}")
elif choice == questions[3]:
    st.subheader("Renewable Capacity Doubled")
    if forecast_arr is not None:
        df_dispatch, obj = optimize_dispatch_dynamic(forecast_arr, renew_max_scale=2.0,
                                                     thermal_cost=1.0, renewable_cost=0.2,
                                                     shed_penalty=200.0, iters=200, particles=80)
        plot_dispatch(df_dispatch, key="q_dispatch_renew")
        st.write(f"Objective={obj:.2f}")
elif choice == questions[4]:
    st.subheader("10% Demand Spike")
    if forecast_arr is not None:
        arr_spike = forecast_arr * 1.1
        df_dispatch, obj = optimize_dispatch_dynamic(arr_spike, thermal_cost=1.0,
                                                     renewable_cost=0.2, shed_penalty=200.0,
                                                     iters=200, particles=80)
        plot_dispatch(df_dispatch, key="q_dispatch_spike")
        st.write(f"Objective={obj:.2f}")
elif choice == questions[5]:
    st.subheader("System Performance Summary")
    if (ARTIFACTS / "summary.json").exists():
        st.json(json.load(open(ARTIFACTS / "summary.json")))
    else:
        try:
            summary = eval_summarize()
            st.json(summary)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

# --- Footer ---
st.markdown("---")
st.caption("SmartGrid-AI Dashboard — built for forecasting, optimization, and evaluation")
