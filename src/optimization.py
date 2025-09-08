"""
optimization.py

Dynamic economic dispatch optimizer (PSO-style) that treats shedding as an outcome (not a fixed reserve).
Inputs:
 - demand_forecast (numpy array) from artifacts/forecast_lstm.npy (24-hour forecast assumed)
Outputs:
 - artifacts/dispatch_plan_dynamic.csv (hourly demand, renewable_supply, thermal_supply, shed)
 - printed objective value and small preview

How it works (summary):
 - For each hour h, we optimize two decision variables in [0,1]:
     re_frac[h] : fraction of demand assigned to renewables (bounded by R_avail[h])
     th_frac[h] : fraction of demand assigned to thermal (bounded by 1.0)
   The actual supplied fractions are clipped by availability; unmet fraction (shed) = max(0, 1 - (re_used + th_used)).
 - Cost per hour = renewable_cost * re_used * demand + thermal_cost * th_used * demand + shed_penalty * shed * demand
 - PSO-like swarm searches over candidate solutions to minimize total cost.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def renewable_profile(hours: int) -> np.ndarray:
    """
    Produce a smooth day-night shaped renewable availability profile in [0,1].
    Peak around hour 14, width adjustable by Gaussian std.
    """
    t = np.arange(hours)
    peak = 14
    std = 4.0
    prof = np.exp(-0.5 * ((t - peak) / std) ** 2)
    prof = prof / prof.max()
    return prof


def optimize_dispatch_dynamic(demand_forecast: np.ndarray,
                              thermal_cost=1.0,
                              renewable_cost=0.2,
                              shed_penalty=100.0,
                              renew_max_scale=1.0,
                              iters=200,
                              particles=80,
                              inertia=0.7,
                              c1=1.4,
                              c2=1.4,
                              rnd_seed=42):
    """
    PSO-style optimization returning best dispatch plan.

    Parameters:
        demand_forecast: np.ndarray (H,) demand values (absolute units)
        thermal_cost: cost per unit thermal generation
        renewable_cost: cost per unit renewable generation
        shed_penalty: very large penalty per unit of shed energy (use to discourage shedding)
        renew_max_scale: scale factor for renewable availability (set <1 if capacity limited)
        iters, particles: PSO params
        inertia, c1, c2: PSO velocity update params
    Returns:
        DataFrame with hour,demand,renewable,thermal,shed and the objective value
    """
    rng = np.random.default_rng(rnd_seed)
    H = len(demand_forecast)

    # Renewable availability fraction (0..1) per hour (scaled by renew_max_scale)
    R_avail = renewable_profile(H) * renew_max_scale

    # Decision vector per particle: concatenation of re_frac (H) and th_frac (H)
    dim = 2 * H

    # Bounds: re_frac ∈ [0, R_avail[h]], th_frac ∈ [0, 1]
    lb = np.concatenate([np.zeros(H), np.zeros(H)])
    ub = np.concatenate([R_avail, np.ones(H)])

    def clip_pos(x):
        return np.minimum(np.maximum(x, lb), ub)

    def fitness(x):
        """
        Compute scalarized objective (lower is better) for a single concatenated vector x.
        """
        re_frac = x[:H]
        th_frac = x[H:]

        # Actual used renewable limited by availability
        re_used = np.minimum(re_frac, R_avail)
        th_used = np.clip(th_frac, 0.0, 1.0)

        # If re_used + th_used > 1, scale down proportionally to not exceed 100% supply
        total = re_used + th_used
        exceed = total > 1.0
        if np.any(exceed):
            # scale factor per hour to bring total <=1
            scale = np.ones_like(total)
            scale[exceed] = 1.0 / (total[exceed] + 1e-9)
            re_used = re_used * scale
            th_used = th_used * scale

        # Unmet (shed) fraction
        shed_frac = np.maximum(0.0, 1.0 - (re_used + th_used))

        # Compute costs (per hour)
        # supply amounts = fraction * demand
        re_supply = re_used * demand_forecast
        th_supply = th_used * demand_forecast
        shed_amount = shed_frac * demand_forecast

        cost = np.sum(renewable_cost * re_supply + thermal_cost * th_supply + shed_penalty * shed_amount)

        # small regularization to avoid pathological tiny oscillations
        reg = 1e-6 * np.sum(x ** 2)
        return cost + reg

    # Initialize particles uniformly inside bounds
    pos = rng.uniform(lb, ub, size=(particles, dim))
    vel = rng.normal(0, 0.02, size=(particles, dim))
    pbest = pos.copy()
    pbest_val = np.array([fitness(p) for p in pbest])
    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    # Main PSO loop
    for _ in range(iters):
        r1 = rng.random((particles, dim))
        r2 = rng.random((particles, dim))
        vel = inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = clip_pos(pos + vel)

        vals = np.array([fitness(p) for p in pos])

        # Update personal bests
        improved = vals < pbest_val
        if np.any(improved):
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]

        # Update global best
        idx = int(np.argmin(pbest_val))
        if pbest_val[idx] < gbest_val:
            gbest_val = pbest_val[idx]
            gbest = pbest[idx].copy()

    # Decode best solution
    re_frac_best = gbest[:H]
    th_frac_best = gbest[H:]

    # Compute supplies and shed final
    re_used = np.minimum(re_frac_best, R_avail)
    th_used = np.clip(th_frac_best, 0.0, 1.0)
    total = re_used + th_used
    exceed = total > 1.0
    if np.any(exceed):
        scale = np.ones_like(total)
        scale[exceed] = 1.0 / (total[exceed] + 1e-9)
        re_used = re_used * scale
        th_used = th_used * scale
    shed_frac = np.maximum(0.0, 1.0 - (re_used + th_used))

    re_supply = re_used * demand_forecast
    th_supply = th_used * demand_forecast
    shed_amount = shed_frac * demand_forecast

    df = pd.DataFrame({
        "hour": np.arange(H),
        "demand": demand_forecast,
        "renewable_frac": re_used,
        "thermal_frac": th_used,
        "shed_frac": shed_frac,
        "renewable": re_supply,
        "thermal": th_supply,
        "shed": shed_amount
    })

    return df, gbest_val


def run_optimization_from_forecast(forecast_path="artifacts/forecast_lstm.npy",
                                   output_csv="artifacts/dispatch_plan_dynamic.csv"):
    Path("artifacts").mkdir(exist_ok=True)
    # Load forecast (expects 1D numpy array)
    arr = np.load(forecast_path)
    demand = arr.astype(float)

    # Tune penalties: high penalty to strongly discourage shedding
    # You can raise shed_penalty to force near-zero slicing of shed
    df, obj = optimize_dispatch_dynamic(demand_forecast=demand,
                                        thermal_cost=1.0,
                                        renewable_cost=0.2,
                                        shed_penalty=200.0,
                                        renew_max_scale=1.0,
                                        iters=250,
                                        particles=100,
                                        inertia=0.7,
                                        c1=1.4,
                                        c2=1.4,
                                        rnd_seed=123)

    df.to_csv(output_csv, index=False)
    print("Optimization done. Objective value:", obj)
    print(df.head())
    return df


if __name__ == "__main__":
    run_optimization_from_forecast()
