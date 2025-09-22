"""
Enhanced main.py
Pipeline runner: forecasting -> stability -> optimization -> RL controller -> evaluation -> AI Problem Solving
"""

from preprocessing import ensure_dirs, load_spain_energy
from forecasting import lstm_forecast_train_eval
from stability_classifier import train_stability_classifier
from optimization import run_optimization_from_forecast
from rl_controller import simulate_controller
from evaluation import summarize
from ai_problem_solver import SmartGridAIProblemSolver
import numpy as np
import json


def run_enhanced_pipeline():
    """Run the complete enhanced SmartGrid-AI pipeline"""
    print("🚀 Starting Enhanced SmartGrid-AI Pipeline...")
    
    ensure_dirs()

    # 1) Forecasting
    print("\n📊 Step 1: Demand Forecasting...")
    df = load_spain_energy()
    demand_series = df["real_demand"].dropna()
    lstm_out, metrics = lstm_forecast_train_eval(demand_series, look_back=48, epochs=10, steps=24)
    np.save("artifacts/forecast_lstm.npy", lstm_out)
    with open("artifacts/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Forecast complete. MAPE: {metrics.get('mape', 0):.2f}%, RMSE: {metrics.get('rmse', 0):.3f}")

    # 2) Stability classification
    print("\n🔌 Step 2: Grid Stability Analysis...")
    train_stability_classifier()
    print("✅ Stability classifier trained")

    # 3) Optimization (dispatch over the LSTM forecast)
    print("\n⚡ Step 3: Economic Dispatch Optimization...")
    run_optimization_from_forecast("artifacts/forecast_lstm.npy")
    print("✅ Dispatch optimization complete")

    # 4) RL controller (adjusts dispatch dynamically)
    print("\n🤖 Step 4: RL-like Controller Simulation...")
    arr = np.load("artifacts/forecast_lstm.npy")
    simulate_controller(arr)
    print("✅ Controller simulation complete")

    # 5) Evaluation summary
    print("\n📈 Step 5: Performance Evaluation...")
    summary = summarize()
    print("✅ Evaluation complete")

    # 6) AI Problem Solver Demo
    print("\n🧠 Step 6: AI Problem Solver Demo...")
    ai_solver = SmartGridAIProblemSolver()
    
    # Demo problems
    demo_problems = [
        "Our demand forecast accuracy is poor with high MAPE values",
        "We need to optimize renewable energy usage while maintaining grid stability",
        "Operational costs are too high, need economic dispatch optimization"
    ]
    
    for i, problem in enumerate(demo_problems, 1):
        print(f"\n🔍 Demo Problem {i}: {problem}")
        solution = ai_solver.solve_problem(problem)
        print(f"✅ Solution generated with {len(solution['solutions'])} analysis categories")
    
    print("\n🎉 Enhanced Pipeline Complete!")
    print("\n📊 Summary:")
    print(f"• Forecast MAPE: {metrics.get('mape', 0):.2f}%")
    if summary.get('dispatch', {}):
        print(f"• Renewable Share: {summary['dispatch'].get('renewable_share_pct', 0):.1f}%")
        print(f"• Load Shedding: {summary['dispatch'].get('shed_pct', 0):.4f}%")
    
    print(f"\n📁 Artifacts generated in 'artifacts/' directory")
    print("🌐 Launch dashboard: streamlit run src/enhanced_dashboard.py")


if __name__ == "__main__":
    run_enhanced_pipeline()