"""
evaluation.py
Consolidates metrics:
- Forecasting: MAPE, RMSE (from artifacts/forecast_metrics.json)
- Stability: accuracy, F1 (from artifacts/stability_report.txt)
- Optimization: dispatch KPIs (renewable share, shed%)
Saves: artifacts/summary.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def summarize():
    summary = {}

    # Forecast metrics
    try:
        with open("artifacts/forecast_metrics.json") as f:
            summary["forecast"] = json.load(f)
    except Exception:
        summary["forecast"] = {}

    # Stability metrics
    try:
        with open("artifacts/stability_report.txt") as f:
            txt = f.read()
        # crude parse
        lines = txt.splitlines()
        acc = [l for l in lines if l.lower().startswith("accuracy")][0].split(":")[1].strip()
        f1 = [l for l in lines if l.lower().startswith("f1")][0].split(":")[1].strip()
        summary["stability"] = {"accuracy": float(acc), "f1_weighted": float(f1)}
    except Exception:
        summary["stability"] = {}

    # Dispatch KPIs
    try:
        df = pd.read_csv("artifacts/dispatch_plan.csv")
        total = df["demand"].sum()
        ren = df["renewable"].sum()
        shed = df["shed"].sum()
        summary["dispatch"] = {
            "renewable_share_pct": float(100.0 * ren / max(total, 1e-9)),
            "shed_pct": float(100.0 * shed / max(total, 1e-9))
        }
    except Exception:
        summary["dispatch"] = {}

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summarize()
