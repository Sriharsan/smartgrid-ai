from __future__ import annotations

# src/ai_problem_solver.py
"""
AI Problem Solver for SmartGrid-AI
Analyzes user problems and provides intelligent solutions using knowledge from trained models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime



class SmartGridAIProblemSolver:
    def __init__(self, artifacts_path: Path = Path("artifacts")):
        self.artifacts_path = artifacts_path
        self.knowledge_base = self._load_knowledge_base()
        self.problem_patterns = self._init_problem_patterns()
        
    def _load_knowledge_base(self) -> Dict:
        """Load all available knowledge from artifacts"""
        kb = {
            "forecast_metrics": self._safe_load_json("forecast_metrics.json"),
            "rl_summary": self._safe_load_json("rl_summary.json"),
            "summary": self._safe_load_json("summary.json"),
            "stability_report": self._safe_load_text("stability_report.txt"),
            "dispatch_plan": self._safe_load_csv("dispatch_plan_dynamic.csv"),
            "forecast_lstm": self._safe_load_npy("forecast_lstm.npy")
        }
        return kb
    
    def _safe_load_json(self, filename: str) -> Dict:
        try:
            with open(self.artifacts_path / filename) as f:
                return json.load(f)
        except:
            return {}
    
    def _safe_load_text(self, filename: str) -> str:
        try:
            return (self.artifacts_path / filename).read_text()
        except:
            return ""
    
    def _safe_load_csv(self, filename: str) -> pd.DataFrame:
        try:
            return pd.read_csv(self.artifacts_path / filename)
        except:
            return pd.DataFrame()
    
    def _safe_load_npy(self, filename: str) -> Optional[np.ndarray]:
        try:
            return np.load(self.artifacts_path / filename)
        except:
            return None
    
    def _init_problem_patterns(self) -> Dict:
        """Initialize problem pattern recognition"""
        return {
            "forecast_accuracy": [
                r"forecast.*accuracy", r"prediction.*error", r"mape", r"rmse", 
                r"how accurate", r"forecast quality"
            ],
            "demand_spike": [
                r"demand.*spike", r"peak.*demand", r"high.*demand", r"demand.*increase",
                r"load.*spike", r"consumption.*peak"
            ],
            "renewable_optimization": [
                r"renewable.*optim", r"green.*energy", r"solar.*wind", r"clean.*energy",
                r"renewable.*capacity", r"sustainable"
            ],
            "load_shedding": [
                r"load.*shed", r"power.*cut", r"blackout", r"outage", r"shed.*load",
                r"demand.*reduction"
            ],
            "cost_optimization": [
                r"cost.*optim", r"economic.*dispatch", r"minimize.*cost", r"cheaper",
                r"budget", r"financial"
            ],
            "stability_analysis": [
                r"stabil", r"grid.*stabil", r"power.*stabil", r"system.*stabil",
                r"voltage", r"frequency"
            ],
            "capacity_planning": [
                r"capacity.*plan", r"future.*demand", r"expansion", r"upgrade",
                r"infrastructure", r"scaling"
            ]
        }
    
    def classify_problem(self, problem_text: str) -> List[str]:
        """Classify the problem into categories"""
        problem_lower = problem_text.lower()
        categories = []
        
        for category, patterns in self.problem_patterns.items():
            if any(re.search(pattern, problem_lower) for pattern in patterns):
                categories.append(category)
        
        return categories if categories else ["general"]
    
    def solve_forecast_accuracy_problem(self) -> Dict:
        """Provide solutions for forecast accuracy issues"""
        metrics = self.knowledge_base.get("forecast_metrics", {})
        
        solution = {
            "analysis": "Analyzing forecast performance metrics...",
            "current_performance": metrics,
            "recommendations": [],
            "actions": []
        }
        
        mape = metrics.get("mape", 0)
        rmse = metrics.get("rmse", 0)
        
        if mape > 10:
            solution["recommendations"].append(
                f"High MAPE ({mape:.2f}%) indicates significant forecast errors. Consider:"
            )
            solution["actions"].extend([
                "Increase LSTM model complexity (more layers/neurons)",
                "Use longer historical lookback window",
                "Add external weather/economic features",
                "Implement ensemble forecasting (LSTM + ARIMA)"
            ])
        elif mape > 5:
            solution["recommendations"].append(f"Moderate MAPE ({mape:.2f}%). Room for improvement:")
            solution["actions"].extend([
                "Fine-tune hyperparameters",
                "Use more training epochs with early stopping",
                "Consider seasonal decomposition"
            ])
        else:
            solution["recommendations"].append(f"Excellent MAPE ({mape:.2f}%). Maintain current approach.")
        
        return solution
    
    def solve_demand_spike_problem(self) -> Dict:
        """Provide solutions for demand spike scenarios"""
        dispatch = self.knowledge_base.get("dispatch_plan")
        
        solution = {
            "analysis": "Analyzing system response to demand spikes...",
            "current_capacity": {},
            "recommendations": [],
            "actions": []
        }
        
        if not dispatch.empty:
            total_capacity = dispatch["renewable"].max() + dispatch["thermal"].max()
            avg_demand = dispatch["demand"].mean()
            peak_demand = dispatch["demand"].max()
            
            solution["current_capacity"] = {
                "average_demand": float(avg_demand),
                "peak_demand": float(peak_demand),
                "total_capacity": float(total_capacity),
                "capacity_margin": float((total_capacity - peak_demand) / peak_demand * 100)
            }
            
            if total_capacity < peak_demand * 1.2:  # Less than 20% margin
                solution["recommendations"].append("Insufficient capacity margin for demand spikes!")
                solution["actions"].extend([
                    "Install additional thermal generators",
                    "Implement demand response programs",
                    "Add battery energy storage systems",
                    "Develop load shedding protocols"
                ])
            else:
                solution["recommendations"].append("Adequate capacity for moderate spikes")
                solution["actions"].extend([
                    "Optimize dispatch priorities",
                    "Implement predictive load management"
                ])
        
        return solution
    
    def solve_renewable_optimization_problem(self) -> Dict:
        """Provide solutions for renewable energy optimization"""
        dispatch = self.knowledge_base.get("dispatch_plan")
        rl_summary = self.knowledge_base.get("rl_summary", {})
        
        solution = {
            "analysis": "Analyzing renewable energy utilization...",
            "current_renewable_stats": {},
            "recommendations": [],
            "actions": []
        }
        
        if not dispatch.empty:
            total_energy = dispatch["demand"].sum()
            renewable_energy = dispatch["renewable"].sum()
            renewable_share = renewable_energy / total_energy * 100
            
            solution["current_renewable_stats"] = {
                "renewable_share_percent": float(renewable_share),
                "avg_renewable_fraction": rl_summary.get("avg_re_frac", 0),
                "total_renewable_energy": float(renewable_energy)
            }
            
            if renewable_share < 30:
                solution["recommendations"].append(f"Low renewable share ({renewable_share:.1f}%). Significant improvement needed:")
                solution["actions"].extend([
                    "Increase solar/wind capacity installation",
                    "Improve renewable forecasting accuracy",
                    "Add grid-scale energy storage",
                    "Implement smart grid technologies"
                ])
            elif renewable_share < 50:
                solution["recommendations"].append(f"Moderate renewable share ({renewable_share:.1f}%). Good progress:")
                solution["actions"].extend([
                    "Optimize renewable dispatch scheduling",
                    "Reduce curtailment through better forecasting",
                    "Add flexible demand response"
                ])
            else:
                solution["recommendations"].append(f"Excellent renewable share ({renewable_share:.1f}%)!")
                solution["actions"].extend([
                    "Maintain current optimization strategies",
                    "Focus on grid stability with high renewables"
                ])
        
        return solution
    
    def solve_stability_analysis_problem(self) -> Dict:
        """Provide solutions for grid stability issues"""
        stability_report = self.knowledge_base.get("stability_report", "")
        
        solution = {
            "analysis": "Analyzing grid stability characteristics...",
            "stability_metrics": {},
            "recommendations": [],
            "actions": []
        }
        
        # Parse stability report for metrics
        if stability_report:
            lines = stability_report.split('\n')
            for line in lines:
                if "Accuracy:" in line:
                    acc_str = line.split(':')[1].strip()
                    acc = float(acc_str.split('±')[0].strip())
                    solution["stability_metrics"]["classification_accuracy"] = acc
                    
                    if acc < 0.9:
                        solution["recommendations"].append(f"Stability prediction accuracy ({acc:.3f}) needs improvement")
                        solution["actions"].extend([
                            "Collect more diverse training data",
                            "Add real-time monitoring sensors",
                            "Implement advanced ML models",
                            "Increase feature engineering"
                        ])
                    else:
                        solution["recommendations"].append(f"Good stability prediction accuracy ({acc:.3f})")
        
        # General stability recommendations
        solution["actions"].extend([
            "Install power quality monitoring systems",
            "Implement automatic voltage regulators",
            "Add reactive power compensation",
            "Deploy grid stabilizing technologies"
        ])
        
        return solution
    
    def solve_cost_optimization_problem(self) -> Dict:
        """Provide solutions for cost optimization"""
        dispatch = self.knowledge_base.get("dispatch_plan")
        
        solution = {
            "analysis": "Analyzing cost optimization opportunities...",
            "cost_breakdown": {},
            "recommendations": [],
            "actions": []
        }
        
        if not dispatch.empty:
            # Simulate cost calculation
            renewable_cost = 0.2  # $/MWh
            thermal_cost = 1.0    # $/MWh
            shed_penalty = 100.0  # $/MWh
            
            total_renewable_cost = dispatch["renewable"].sum() * renewable_cost
            total_thermal_cost = dispatch["thermal"].sum() * thermal_cost
            total_shed_cost = dispatch["shed"].sum() * shed_penalty
            total_cost = total_renewable_cost + total_thermal_cost + total_shed_cost
            
            solution["cost_breakdown"] = {
                "renewable_cost": float(total_renewable_cost),
                "thermal_cost": float(total_thermal_cost),
                "shedding_cost": float(total_shed_cost),
                "total_cost": float(total_cost),
                "cost_per_mwh": float(total_cost / dispatch["demand"].sum())
            }
            
            renewable_fraction = dispatch["renewable"].sum() / dispatch["demand"].sum()
            
            if renewable_fraction < 0.5:
                solution["recommendations"].append("Increase renewable usage to reduce costs")
                solution["actions"].extend([
                    "Shift more load to renewable sources",
                    "Improve renewable forecasting",
                    "Add energy storage for load shifting"
                ])
            
            if dispatch["shed"].sum() > 0:
                solution["recommendations"].append("Eliminate load shedding to avoid penalties")
                solution["actions"].extend([
                    "Increase total generation capacity",
                    "Implement demand response programs",
                    "Add peak shaving technologies"
                ])
        
        return solution
    
    def solve_problem(self, problem_text: str) -> Dict:
        """Main problem solving function"""
        categories = self.classify_problem(problem_text)
        
        solution = {
            "problem_text": problem_text,
            "detected_categories": categories,
            "timestamp": datetime.now().isoformat(),
            "solutions": {},
            "overall_recommendations": [],
            "priority_actions": []
        }
        
        # Apply appropriate solvers
        for category in categories:
            if category == "forecast_accuracy":
                solution["solutions"]["forecast_accuracy"] = self.solve_forecast_accuracy_problem()
            elif category == "demand_spike":
                solution["solutions"]["demand_spike"] = self.solve_demand_spike_problem()
            elif category == "renewable_optimization":
                solution["solutions"]["renewable_optimization"] = self.solve_renewable_optimization_problem()
            elif category == "stability_analysis":
                solution["solutions"]["stability_analysis"] = self.solve_stability_analysis_problem()
            elif category == "cost_optimization":
                solution["solutions"]["cost_optimization"] = self.solve_cost_optimization_problem()
            elif category == "load_shedding":
                solution["solutions"]["load_shedding"] = self.solve_demand_spike_problem()  # Similar approach
        
        # Generate overall recommendations
        self._generate_overall_recommendations(solution)
        
        # Save solution report
        self._save_solution_report(solution)
        
        return solution
    
    def _generate_overall_recommendations(self, solution: Dict):
        """Generate consolidated recommendations"""
        all_actions = []
        for sol in solution["solutions"].values():
            all_actions.extend(sol.get("actions", []))
        
        # Remove duplicates and prioritize
        unique_actions = list(dict.fromkeys(all_actions))
        
        # Simple prioritization based on frequency and keywords
        high_priority = [a for a in unique_actions if any(kw in a.lower() for kw in ["storage", "forecast", "capacity"])]
        medium_priority = [a for a in unique_actions if a not in high_priority and any(kw in a.lower() for kw in ["optim", "smart", "monitor"])]
        low_priority = [a for a in unique_actions if a not in high_priority + medium_priority]
        
        solution["priority_actions"] = {
            "high_priority": high_priority[:3],  # Top 3
            "medium_priority": medium_priority[:3],
            "low_priority": low_priority[:3]
        }
        
        solution["overall_recommendations"] = [
            "Implement high-priority actions first for maximum impact",
            "Monitor system performance after each change",
            "Consider integrated solutions that address multiple issues"
        ]
    
    def _save_solution_report(self, solution: Dict):
        """Save detailed solution report"""
        report_path = self.artifacts_path / "ai_solution_report.json"
        with open(report_path, "w") as f:
            json.dump(solution, f, indent=2)
        
        # Also save a human-readable report
        readable_path = self.artifacts_path / "ai_solution_report.txt"
        with open(readable_path, "w") as f:
            f.write("SmartGrid-AI Problem Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Problem: {solution['problem_text']}\n")
            f.write(f"Analysis Date: {solution['timestamp']}\n")
            f.write(f"Detected Categories: {', '.join(solution['detected_categories'])}\n\n")
            
            for category, sol in solution["solutions"].items():
                f.write(f"\n{category.upper().replace('_', ' ')}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Analysis: {sol.get('analysis', 'N/A')}\n\n")
                if sol.get("recommendations"):
                    f.write("Recommendations:\n")
                    for rec in sol["recommendations"]:
                        f.write(f"• {rec}\n")
                if sol.get("actions"):
                    f.write("\nSuggested Actions:\n")
                    for action in sol["actions"]:
                        f.write(f"  - {action}\n")
                f.write("\n")
            
            f.write("\nPRIORITY ACTIONS\n")
            f.write("=" * 20 + "\n")
            for priority, actions in solution["priority_actions"].items():
                f.write(f"\n{priority.upper()}:\n")
                for action in actions:
                    f.write(f"  • {action}\n")


# Enhanced dashboard.py with AI Problem Solver integration
"""
Enhanced Streamlit Dashboard — SmartGrid-AI with AI Problem Solver
"""


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

from ai_problem_solver import SmartGridAIProblemSolver

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SmartGrid-AI Enhanced Dashboard",
    page_icon="⚡",
    layout="wide",
)

ART = Path("artifacts")
DATA = Path("data")

# ---------------------------------------------------------------------
# Initialize AI Problem Solver
# ---------------------------------------------------------------------
@st.cache_resource
def get_ai_solver():
    return SmartGridAIProblemSolver(ART)

ai_solver = get_ai_solver()

# ---------------------------------------------------------------------
# Helpers (same as before)
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
# Header
# ---------------------------------------------------------------------
st.title("⚡ SmartGrid-AI Enhanced Dashboard")
st.caption("AI-Powered Forecast → Stability → Dispatch → Control → Problem Solving")

# KPIs
col1, col2, col3, col4 = st.columns(4)
forecast_metrics_path = ART / "forecast_metrics.json"
rl_summary_path = ART / "rl_summary.json"

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
TAB_OVERVIEW, TAB_AI_SOLVER, TAB_FORECAST, TAB_DISPATCH, TAB_STABILITY, TAB_CONTROLLER, TAB_DATA = st.tabs([
    "Overview",
    "🧠 AI Problem Solver",
    "Forecast",
    "Dispatch",
    "Stability", 
    "Controller",
    "Data Explorer"
])

# ---------------------------------------------------------------------
# Overview (same as before)
# ---------------------------------------------------------------------
with TAB_OVERVIEW:
    st.subheader("Operational Snapshot")
    forecast_lstm_path = ART / "forecast_lstm.npy"
    dispatch_path = ART / "dispatch_plan_dynamic.csv"
    if not dispatch_path.exists():
        dispatch_path = ART / "dispatch_plan.csv"
    
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

# ---------------------------------------------------------------------
# AI Problem Solver Tab
# ---------------------------------------------------------------------
with TAB_AI_SOLVER:
    st.subheader("🧠 AI-Powered Problem Solver")
    st.markdown("*Describe your SmartGrid challenge and get intelligent solutions powered by trained models and domain expertise.*")
    
    # Problem input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Describe Your Problem**")
        problem_input = st.text_area(
            "What challenge are you facing with your smart grid system?",
            placeholder="Examples:\n• Our demand forecast accuracy is poor\n• Need to optimize renewable energy usage\n• Experiencing frequent load shedding\n• Cost optimization for next month\n• Grid stability issues during peak hours",
            height=150
        )
        
        # Quick problem templates
        st.markdown("**Quick Problem Templates**")
        template_cols = st.columns(3)
        
        with template_cols[0]:
            if st.button("📈 Forecast Issues"):
                problem_input = "Our demand forecasting accuracy is poor with high MAPE values. How can we improve our LSTM model performance?"
                st.experimental_rerun()
        
        with template_cols[1]:
            if st.button("🌱 Renewable Optimization"):
                problem_input = "We want to maximize renewable energy utilization while maintaining grid stability. What's the optimal renewable capacity?"
                st.experimental_rerun()
                
        with template_cols[2]:
            if st.button("💰 Cost Reduction"):
                problem_input = "Our operational costs are too high. How can we optimize economic dispatch to reduce costs while meeting demand?"
                st.experimental_rerun()
    
    with col2:
        st.markdown("**AI Solution Engine**")
        st.info("💡 The AI analyzes your problem using:\n• Forecast accuracy metrics\n• Dispatch optimization data\n• Grid stability models\n• Historical performance\n• Domain expertise")
        
        if st.button("🚀 Generate AI Solution", type="primary", use_container_width=True):
            if problem_input.strip():
                with st.spinner("🧠 AI analyzing your problem..."):
                    solution = ai_solver.solve_problem(problem_input)
                    st.session_state.current_solution = solution
                    st.success("✅ Solution generated!")
            else:
                st.warning("Please describe your problem first!")
    
    # Display solution
    if 'current_solution' in st.session_state:
        solution = st.session_state.current_solution
        
        st.markdown("---")
        st.subheader("📋 AI Solution Report")
        
        # Problem summary
        st.markdown(f"**Problem:** {solution['problem_text']}")
        st.markdown(f"**Detected Categories:** {', '.join(solution['detected_categories'])}")
        st.markdown(f"**Analysis Date:** {solution['timestamp']}")
        
        # Priority actions
        st.markdown("### 🎯 Priority Actions")
        priority_actions = solution.get('priority_actions', {})
        
        if priority_actions.get('high_priority'):
            st.markdown("**🔴 High Priority:**")
            for action in priority_actions['high_priority']:
                st.markdown(f"• {action}")
        
        if priority_actions.get('medium_priority'):
            st.markdown("**🟡 Medium Priority:**")
            for action in priority_actions['medium_priority']:
                st.markdown(f"• {action}")
        
        if priority_actions.get('low_priority'):
            st.markdown("**🟢 Low Priority:**")
            for action in priority_actions['low_priority']:
                st.markdown(f"• {action}")
        
        # Detailed solutions by category
        st.markdown("### 📊 Detailed Analysis by Category")
        
        for category, sol in solution.get('solutions', {}).items():
            with st.expander(f"📈 {category.replace('_', ' ').title()}", expanded=False):
                st.markdown(f"**Analysis:** {sol.get('analysis', 'N/A')}")
                
                # Display metrics if available
                if 'current_performance' in sol and sol['current_performance']:
                    st.markdown("**Current Performance:**")
                    for key, value in sol['current_performance'].items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
                
                # Display any other metrics
                for metrics_key in ['current_capacity', 'current_renewable_stats', 'stability_metrics', 'cost_breakdown']:
                    if metrics_key in sol and sol[metrics_key]:
                        st.markdown(f"**{metrics_key.replace('_', ' ').title()}:**")
                        for key, value in sol[metrics_key].items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else str(value))
                            else:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                
                # Recommendations
                if sol.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in sol['recommendations']:
                        st.markdown(f"• {rec}")
                
                # Actions
                if sol.get('actions'):
                    st.markdown("**Suggested Actions:**")
                    for action in sol['actions'][:5]:  # Show top 5 actions
                        st.markdown(f"  - {action}")
        
        # Download reports
        st.markdown("### 📥 Download Reports")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON report
            json_report = json.dumps(solution, indent=2)
            st.download_button(
                "📄 Download JSON Report",
                data=json_report.encode('utf-8'),
                file_name=f"ai_solution_{solution['timestamp'][:10]}.json",
                mime="application/json"
            )
        
        with col2:
            # Text report (check if it exists)
            txt_report_path = ART / "ai_solution_report.txt"
            if txt_report_path.exists():
                txt_report = txt_report_path.read_text()
                st.download_button(
                    "📝 Download Text Report", 
                    data=txt_report.encode('utf-8'),
                    file_name=f"ai_solution_{solution['timestamp'][:10]}.txt",
                    mime="text/plain"
                )

# ---------------------------------------------------------------------
# Other tabs remain the same as original
# ---------------------------------------------------------------------
with TAB_FORECAST:
    st.subheader("Forecast Quality & Trajectories")
    forecast_lstm_path = ART / "forecast_lstm.npy"
    forecast_arima_path = ART / "forecast_arima.csv"
    
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

with TAB_DISPATCH:
    st.subheader("Economic Dispatch")
    dispatch_path = ART / "dispatch_plan_dynamic.csv"
    if not dispatch_path.exists():
        dispatch_path = ART / "dispatch_plan.csv"
    
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

with TAB_STABILITY:
    st.subheader("Grid Stability Classifier")
    stability_report_path = ART / "stability_report.txt"
    rep = load_text(stability_report_path)
    if not rep:
        st.info("No stability report available.")
    else:
        st.code(rep, language="text")

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

with TAB_DATA:
    st.subheader("Data Explorer")
    forecast_lstm_path = ART / "forecast_lstm.npy"
    dispatch_path = ART / "dispatch_plan_dynamic.csv"
    if not dispatch_path.exists():
        dispatch_path = ART / "dispatch_plan.csv"
        
    if forecast_lstm_path.exists():
        st.write("**LSTM forecast (first 10)**")
        st.write(load_npy(forecast_lstm_path)[:10])
    if dispatch_path.exists():
        st.write("**Dispatch plan**")
        st.dataframe(load_csv(dispatch_path), use_container_width=True)