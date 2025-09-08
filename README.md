# ⚡ SmartGrid-AI

**SmartGrid-AI** is an end-to-end AI system for **smart grid operations** that integrates:

- **Load Forecasting** → Predict short-term electricity demand (LSTM, ARIMA).
- **Stability Classification** → Detect grid stability using ML (XGBoost + SMOTE).
- **Economic Dispatch** → Optimize renewable vs thermal generation using metaheuristics.
- **RL-like Control** → Reinforcement-inspired policy for dispatch under uncertainty.
- **Interactive Dashboard** → Streamlit app for real-time exploration and visualization.

---

## 📊 Features

- **Forecasting** → LSTM-based demand forecasting with metrics (MAPE, RMSE).
- **Stability** → Robust classification with >98% accuracy (XGBoost).
- **Optimization** → Economic dispatch balancing renewables, thermal, and load shedding.
- **Control** → RL-inspired controller with reward-based evaluation.
- **Dashboard** → Mobile-friendly Streamlit interface with tabs:
  - Overview
  - Forecast
  - Dispatch
  - Stability
  - Controller
  - Data Explorer

---

## 🚀 Deployment

This project is deployed on **Streamlit Cloud**.  
To run the dashboard locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/dashboard.py
