"""
rl_controller.py
Lightweight rule-based RL-like controller for smart grid dispatch.
Instead of full RL training, we use random policy search over a finite horizon
with discrete actions on renewable fraction. Thermal adjusts accordingly.
Reward = -(cost + 20 * shed) with cost model aligned with optimization.
Saves results: artifacts/rl_summary.json
"""

import json
import numpy as np
from pathlib import Path


def simulate_controller(demand: np.ndarray, steps: int = None, episodes: int = 100, rnd_seed=7):
    rng = np.random.default_rng(rnd_seed)
    H = len(demand) if steps is None else min(steps, len(demand))

    # Start fractions (neutral)
    cap = 1.0  # no fixed reserve, allow full dispatch
    re_frac = np.full(H, cap * 0.4)
    th_frac = np.full(H, cap * 0.6)

    # Action space: change renewable fraction, thermal adjusts to keep sum <= cap
    actions = [-0.05, -0.02, 0.0, 0.02, 0.05]  # increments
    best_reward = -1e18
    best_plan = None

    def step_cost(re, th, d):
        shed = max(0.0, 1.0 - (re + th))
        cost = 0.2 * re * d + 1.0 * th * d + 20.0 * shed * d
        return -(cost), shed

    for _ in range(episodes):
        r = re_frac.copy()
        t = th_frac.copy()
        total_reward = 0.0
        for h in range(H):
            delta = actions[rng.integers(0, len(actions))]
            r[h] = np.clip(r[h] + delta, 0, cap)
            t[h] = np.clip(cap - r[h], 0, cap)
            rew, _ = step_cost(r[h], t[h], demand[h])
            total_reward += rew
        if total_reward > best_reward:
            best_reward = total_reward
            best_plan = (r, t)

    r, t = best_plan
    shed = np.maximum(0.0, 1.0 - (r + t))

    Path("artifacts").mkdir(exist_ok=True)
    out = {
        "avg_reward": float(best_reward / H),
        "avg_shed": float(np.mean(shed)),
        "avg_re_frac": float(np.mean(r)),
        "avg_th_frac": float(np.mean(t))
    }
    with open("artifacts/rl_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("RL-like controller done:", out)
    return r, t, shed, out


if __name__ == "__main__":
    arr = np.load("artifacts/forecast_lstm.npy")
    simulate_controller(arr)
