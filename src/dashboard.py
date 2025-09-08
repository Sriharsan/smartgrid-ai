"""
Streamlit Dashboard — SmartGrid-AI (Client Edition)

Runs on Streamlit Cloud (worldwide, mobile-friendly).
Consumes artifacts only (no local preprocessing or training code).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SmartGrid-AI Dashboard",
    page_icon="⚡",
    layout="wide",
)

ART = Path("artifacts")
DATA = Path("data")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_json(path: Path) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_text(path: Path) -> str:
    try:
        return Path(path).read_text()
    except Exception:
        return ""

# ---------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------
forecast_lstm_path = ART / "forecast_lstm.npy"
forecast_metrics_path = ART / "forecast_metrics.json"
forecast_arima_path = ART / "forecast_arima.csv"

dispatch_path = ART / "dispatch_plan_dynamic.csv"
if not dispatch_path.exists():
    dispatch_path = ART / "dispatch_plan.csv"

stability_report_path = ART / "stability_report.txt"
rl_summary_path = ART / "rl_summary.json"

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("⚡ SmartGrid-AI Dashboard")
st.caption("Forecast → Stability → Dispatch → Control")

# KPIs
col1, col2, col3, col4 = st.columns(4)
metrics = load_json(forecast_metrics_path)
if metrics:
    col1.metric("Forecast MAPE (%)", f"{metrics.get('mape', 0):.2f}")
    col2.metric("Forecast RMSE", f"{metrics.get('rmse', 0):.3f}")
else:
    col1.metric("Forecast MAPE (%)", "–")
    col2.metric("Forecast RMSE", "–")

rl = load_json(rl_summary_path)
if rl:
    col3.metric("Avg RE frac", f"{rl.get('avg_re_frac', 0):.3f}")
    col4.metric("Avg TH frac", f"{rl.get('avg_th_frac', 0):.3f}")
else:
    col3.metric("Avg RE frac", "–")
    col4.metric("Avg TH frac", "–")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
TAB_OVERVIEW, TAB_FORECAST, TAB_DISPATCH, TAB_STABILITY, TAB_CONTROLLER, TAB_DATA = st.tabs([
    "Overview",
    "Forecast",
    "Dispatch",
    "Stability",
    "Controller",
    "Data Explorer",
])

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
with TAB_OVERVIEW:
    st.subheader("Operational Snapshot")
    fc = load_npy(forecast_lstm_path)
    disp = load_csv(dispatch_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Next-24h Demand Forecast (LSTM)**")
        if fc is None:
            st.info("No LSTM forecast available.")
        else:
            df_fc = pd.DataFrame({"hour": np.arange(len(fc)), "LSTM Forecast": fc})
            if _HAS_PLOTLY:
                fig = px.line(df_fc, x="hour", y="LSTM Forecast", markers=True)
                st.plotly_chart(fig, use_container_width=True, key="overview_forecast")
            else:
                st.line_chart(df_fc.set_index("hour"))

    with col2:
        st.markdown("**Economic Dispatch Mix**")
        if disp.empty:
            st.info("No dispatch plan available.")
        else:
            total_demand = disp["demand"].sum()
            ren_share = 100 * disp["renewable"].sum() / max(total_demand, 1e-9)
            shed_share = 100 * disp["shed"].sum() / max(total_demand, 1e-9)
            th_share = 100 * disp["thermal"].sum() / max(total_demand, 1e-9)

            c1, c2, c3 = st.columns(3)
            c1.metric("Renewable Share (%)", f"{ren_share:.2f}")
            c2.metric("Thermal Share (%)", f"{th_share:.2f}")
            c3.metric("Load Shed (%)", f"{shed_share:.4f}")

            if _HAS_PLOTLY:
                pie = px.pie(
                    values=[disp["renewable"].sum(), disp["thermal"].sum(), disp["shed"].sum()],
                    names=["Renewable", "Thermal", "Shed"],
                    hole=0.45,
                )
                st.plotly_chart(pie, use_container_width=True, key="overview_dispatch_pie")
            else:
                st.bar_chart(disp[["renewable", "thermal", "shed"]].sum())

# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------
with TAB_FORECAST:
    st.subheader("Forecast Quality & Trajectories")
    fc = load_npy(forecast_lstm_path)
    ar = load_csv(forecast_arima_path)
    mets = load_json(forecast_metrics_path)

    if fc is None and ar.empty:
        st.info("No forecast data available.")
    else:
        df = pd.DataFrame({"hour": np.arange(len(fc)) if fc is not None else [], "LSTM Forecast": fc})
        if _HAS_PLOTLY and not df.empty:
            fig = px.line(df, x="hour", y="LSTM Forecast", markers=True)
            st.plotly_chart(fig, use_container_width=True, key="forecast_lstm")
        elif not df.empty:
            st.line_chart(df.set_index("hour"))

    st.markdown("**Metrics**")
    if mets:
        st.metric("MAPE (%)", f"{mets.get('mape', 0):.2f}")
        st.metric("RMSE", f"{mets.get('rmse', 0):.3f}")
    else:
        st.write("–")

# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------
with TAB_DISPATCH:
    st.subheader("Economic Dispatch")
    disp = load_csv(dispatch_path)
    if disp.empty:
        st.info("No dispatch plan available.")
    else:
        long_df = disp.melt(id_vars=["hour", "demand"], value_vars=["renewable", "thermal", "shed"],
                            var_name="source", value_name="MWh")
        if _HAS_PLOTLY:
            fig = px.area(long_df, x="hour", y="MWh", color="source")
            st.plotly_chart(fig, use_container_width=True, key="dispatch_area")
        else:
            st.area_chart(disp.set_index("hour")[["renewable", "thermal", "shed"]])

        st.download_button(
            "⬇️ Download Dispatch CSV",
            data=disp.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_plan.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------
with TAB_STABILITY:
    st.subheader("Grid Stability Classifier")
    rep = load_text(stability_report_path)
    if not rep:
        st.info("No stability report available.")
    else:
        st.code(rep, language="text")

# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------
with TAB_CONTROLLER:
    st.subheader("RL-like Controller Summary")
    rl = load_json(rl_summary_path)
    if not rl:
        st.info("No controller results available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Reward", f"{rl.get('avg_reward', 0):.3f}")
        c2.metric("Avg Shed", f"{rl.get('avg_shed', 0):.4f}")
        c3.metric("Avg RE frac", f"{rl.get('avg_re_frac', 0):.3f}")
        c4.metric("Avg TH frac", f"{rl.get('avg_th_frac', 0):.3f}")

# ---------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------
with TAB_DATA:
    st.subheader("Data Explorer")
    if forecast_lstm_path.exists():
        st.write("**LSTM forecast (first 10)**")
        st.write(load_npy(forecast_lstm_path)[:10])
    if dispatch_path.exists():
        st.write("**Dispatch plan**")
        st.dataframe(load_csv(dispatch_path), use_container_width=True)
