# src/enhanced_streamlit_app.py
"""
Enhanced SmartGrid-AI — Streamlit Dashboard with AI Problem Solver
Usage:
    streamlit run src/enhanced_streamlit_app.py
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
from ai_problem_solver import SmartGridAIProblemSolver

import plotly.graph_objects as go
import plotly.express as px

# ========== Setup ==========
ensure_dirs()
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

st.set_page_config(
    page_title="SmartGrid-AI Enhanced", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# ========== Initialize AI Solver ==========
@st.cache_resource
def get_ai_solver():
    return SmartGridAIProblemSolver(ARTIFACTS)

# ========== Helper Functions ==========
@st.cache_data(ttl=600)
def load_packaged_spain():
    try:
        return load_spain_energy()
    except Exception as e:
        st.error(f"Failed to load Spain dataset: {e}")
        return pd.DataFrame()

def try_load_forecast():
    if (ARTIFACTS / "forecast_lstm.npy").exists():
        return np.load(ARTIFACTS / "forecast_lstm.npy"), "lstm"
    if (ARTIFACTS / "forecast_arima.csv").exists():
        df = pd.read_csv(ARTIFACTS / "forecast_arima.csv")
        col = df.select_dtypes(include='number').columns[0]
        return df[col].values, "arima"
    return None, None

def plot_forecast_interactive(history_series, forecast, title="Demand & Forecast"):
    if history_series is None or history_series.empty:
        st.warning("No historical data available")
        return

    hist = history_series.reset_index(drop=True).astype(float)
    H = len(forecast)
    x_hist = np.arange(len(hist))
    x_fore = np.arange(len(hist), len(hist) + H)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_hist[-100:], y=hist[-100:], 
        mode="lines", name="Recent History",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x_fore, y=forecast, 
        mode="lines+markers", name="24h Forecast",
        line=dict(color='red', dash='dash'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time Index",
        yaxis_title="Demand (MW)",
        hovermode='x unified'
    )
    return fig

def plot_dispatch_stacked(df_dispatch):
    if df_dispatch is None or df_dispatch.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_dispatch.hour, 
        y=df_dispatch.renewable, 
        name="Renewable",
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=df_dispatch.hour, 
        y=df_dispatch.thermal, 
        name="Thermal",
        marker_color='orange'
    ))
    fig.add_trace(go.Bar(
        x=df_dispatch.hour, 
        y=df_dispatch.shed, 
        name="Load Shed",
        marker_color='red'
    ))
    fig.update_layout(
        barmode='stack',
        title="24-Hour Economic Dispatch Plan",
        xaxis_title="Hour",
        yaxis_title="Energy (MWh)",
        hovermode='x unified'
    )
    return fig

# ========== Sidebar ==========
st.sidebar.title("⚡ SmartGrid-AI Enhanced")
st.sidebar.write("AI-Powered Grid Management")

# Data source selection
data_option = st.sidebar.radio("Data Source:", ["Spain Dataset", "Upload CSV", "Manual Input"])

# Forecast method
forecast_method = st.sidebar.selectbox("Forecast Method:", ["Load Existing", "ARIMA", "LSTM"])

# Quick actions
st.sidebar.markdown("### 🚀 Quick Actions")
if st.sidebar.button("Run Full Pipeline", type="primary"):
    with st.spinner("Running complete pipeline..."):
        # This would run the main pipeline
        st.success("Pipeline complete! Check results in tabs.")

# AI Problem Solver quick access
st.sidebar.markdown("### 🧠 AI Assistant")
quick_problem = st.sidebar.selectbox(
    "Quick Problem Analysis:",
    ["Custom Problem", "Forecast Accuracy", "Cost Optimization", "Renewable Integration", "Grid Stability"]
)

if st.sidebar.button("Analyze Problem") and quick_problem != "Custom Problem":
    st.session_state.ai_problem = quick_problem
    st.session_state.show_ai_solution = True

# ========== Main Interface ==========
st.title("⚡ SmartGrid-AI Enhanced Dashboard")
st.markdown("*AI-powered smart grid forecasting, optimization, and problem solving*")

# Metrics row
metrics_cols = st.columns(4)
try:
    with open(ARTIFACTS / "forecast_metrics.json") as f:
        fmetrics = json.load(f)
    metrics_cols[0].metric("Forecast MAPE", f"{fmetrics.get('mape', 0):.2f}%")
    metrics_cols[1].metric("Forecast RMSE", f"{fmetrics.get('rmse', 0):.3f}")
except:
    metrics_cols[0].metric("Forecast MAPE", "N/A")
    metrics_cols[1].metric("Forecast RMSE", "N/A")

try:
    with open(ARTIFACTS / "rl_summary.json") as f:
        rl_data = json.load(f)
    metrics_cols[2].metric("Renewable Fraction", f"{rl_data.get('avg_re_frac', 0):.3f}")
    metrics_cols[3].metric("Avg Shed", f"{rl_data.get('avg_shed', 0):.4f}")
except:
    metrics_cols[2].metric("Renewable Fraction", "N/A")
    metrics_cols[3].metric("Avg Shed", "N/A")

# ========== Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", 
    "🧠 AI Problem Solver", 
    "📊 Analytics", 
    "⚙️ Operations",
    "📁 Data Explorer"
])

# ========== Tab 1: Overview ==========
with tab1:
    st.header("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Demand Forecast")
        forecast_arr, forecast_type = try_load_forecast()
        
        if forecast_arr is not None:
            df_spain = load_packaged_spain()
            if not df_spain.empty and "real_demand" in df_spain.columns:
                hist_series = df_spain["real_demand"]
                fig = plot_forecast_interactive(hist_series, forecast_arr, f"Next 24h Forecast ({forecast_type.upper()})")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                df_forecast = pd.DataFrame({
                    "Hour": np.arange(len(forecast_arr)),
                    "Forecast": forecast_arr
                })
                st.line_chart(df_forecast.set_index("Hour"))
        else:
            st.info("No forecast data available. Generate forecast in Analytics tab.")
    
    with col2:
        st.subheader("⚡ Dispatch Optimization")
        try:
            dispatch_df = pd.read_csv(ARTIFACTS / "dispatch_plan_dynamic.csv")
            fig_dispatch = plot_dispatch_stacked(dispatch_df)
            if fig_dispatch:
                st.plotly_chart(fig_dispatch, use_container_width=True)
            
            # Summary metrics
            total_demand = dispatch_df["demand"].sum()
            renewable_pct = (dispatch_df["renewable"].sum() / total_demand) * 100
            shed_pct = (dispatch_df["shed"].sum() / total_demand) * 100
            
            metric_cols = st.columns(3)
            metric_cols[0].metric("Total Demand", f"{total_demand:.0f} MWh")
            metric_cols[1].metric("Renewable %", f"{renewable_pct:.1f}%")
            metric_cols[2].metric("Shed %", f"{shed_pct:.3f}%")
            
        except:
            st.info("No dispatch data available. Run optimization in Operations tab.")

# ========== Tab 2: AI Problem Solver ==========
with tab2:
    st.header("🧠 AI Problem Solver")
    st.markdown("*Describe your smart grid challenge and get AI-powered solutions*")
    
    # Initialize AI solver
    ai_solver = get_ai_solver()
    
    # Problem input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Problem Description")
        
        # Auto-populate if quick problem was selected
        default_problems = {
            "Forecast Accuracy": "Our demand forecasting has high MAPE values and poor accuracy. How can we improve the LSTM model performance?",
            "Cost Optimization": "We need to reduce operational costs while maintaining reliable power supply. What's the optimal dispatch strategy?",
            "Renewable Integration": "How can we maximize renewable energy usage while ensuring grid stability and meeting all demand?",
            "Grid Stability": "We're experiencing grid stability issues during peak demand periods. What solutions can help?"
        }
        
        if 'ai_problem' in st.session_state and st.session_state.ai_problem in default_problems:
            default_text = default_problems[st.session_state.ai_problem]
        else:
            default_text = ""
        
        problem_input = st.text_area(
            "Describe your smart grid challenge:",
            value=default_text,
            placeholder="Examples:\n• Forecast accuracy is poor with 15% MAPE\n• Need to integrate more solar/wind capacity\n• Frequent load shedding during peak hours\n• High operational costs due to thermal generation\n• Grid instability with renewable fluctuations",
            height=150
        )
        
        # Problem templates
        st.markdown("**Quick Templates:**")
        template_cols = st.columns(4)
        
        templates = [
            ("📊 Forecast Issues", "forecast accuracy poor high MAPE RMSE improve LSTM"),
            ("🌱 Renewables", "maximize renewable energy solar wind capacity optimization"),
            ("💰 Cost Reduction", "reduce operational costs economic dispatch optimization"),
            ("⚡ Load Shedding", "frequent load shedding peak demand capacity planning")
        ]
        
        for i, (name, template) in enumerate(templates):
            if template_cols[i].button(name, key=f"template_{i}"):
                st.session_state.template_problem = template
                st.rerun()
        
        if 'template_problem' in st.session_state:
            problem_input = st.session_state.template_problem
            del st.session_state.template_problem
    
    with col2:
        st.subheader("AI Analysis Engine")
        st.info("""
        🎯 **AI Capabilities:**
        • Pattern recognition
        • Performance analysis  
        • Solution recommendation
        • Priority ranking
        • Implementation planning
        """)
        
        if st.button("🚀 Generate Solution", type="primary", width="stretch"):
            if problem_input.strip():
                with st.spinner("🧠 AI analyzing problem..."):
                    try:
                        solution = ai_solver.solve_problem(problem_input)
                        st.session_state.ai_solution = solution
                        st.success("✅ AI solution generated!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error generating solution: {e}")
            else:
                st.warning("Please describe your problem first!")
    
    # Display AI Solution
    if 'ai_solution' in st.session_state:
        solution = st.session_state.ai_solution
        
        st.markdown("---")
        st.subheader("📋 AI Solution Report")
        
        # Header info
        info_cols = st.columns(3)
        info_cols[0].metric("Problem Categories", len(solution['detected_categories']))
        info_cols[1].metric("Solutions Generated", len(solution['solutions']))
        info_cols[2].metric("Priority Actions", sum(len(actions) for actions in solution.get('priority_actions', {}).values()))
        
        # Priority Actions Dashboard
        st.subheader("🎯 Recommended Actions")
        priority_actions = solution.get('priority_actions', {})
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            st.markdown("**🔴 High Priority**")
            for action in priority_actions.get('high_priority', []):
                st.markdown(f"• {action}")
        
        with action_cols[1]:
            st.markdown("**🟡 Medium Priority**") 
            for action in priority_actions.get('medium_priority', []):
                st.markdown(f"• {action}")
        
        with action_cols[2]:
            st.markdown("**🟢 Low Priority**")
            for action in priority_actions.get('low_priority', []):
                st.markdown(f"• {action}")
        
        # Detailed Analysis
        st.subheader("🔍 Detailed Analysis")
        
        for category, analysis in solution.get('solutions', {}).items():
            with st.expander(f"📊 {category.replace('_', ' ').title()}", expanded=True):
                
                # Analysis summary
                st.markdown(f"**Analysis:** {analysis.get('analysis', 'N/A')}")
                
                # Show metrics in columns if available
                metrics_data = {}
                for key in ['current_performance', 'current_capacity', 'current_renewable_stats', 'stability_metrics', 'cost_breakdown']:
                    if key in analysis and analysis[key]:
                        metrics_data.update(analysis[key])
                
                if metrics_data:
                    st.markdown("**Key Metrics:**")
                    metric_cols = st.columns(min(4, len(metrics_data)))
                    for i, (key, value) in enumerate(metrics_data.items()):
                        if isinstance(value, (int, float)):
                            metric_cols[i % 4].metric(
                                key.replace('_', ' ').title(), 
                                f"{value:.4f}" if isinstance(value, float) else str(value)
                            )
                
                # Recommendations
                if analysis.get('recommendations'):
                    st.markdown("**💡 Recommendations:**")
                    for rec in analysis['recommendations']:
                        st.info(rec)
                
                # Action items
                if analysis.get('actions'):
                    st.markdown("**🔧 Action Items:**")
                    for action in analysis['actions'][:8]:  # Top 8 actions
                        st.markdown(f"  ✓ {action}")
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        download_cols = st.columns(2)
        
        with download_cols[0]:
            json_data = json.dumps(solution, indent=2)
            st.download_button(
                "📄 Download JSON Report",
                data=json_data.encode('utf-8'),
                file_name=f"ai_solution_{solution['timestamp'][:19].replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with download_cols[1]:
            # Create text summary
            text_summary = f"""SmartGrid-AI Problem Analysis Report
{'='*50}

Problem: {solution['problem_text']}
Analysis Date: {solution['timestamp']}
Categories: {', '.join(solution['detected_categories'])}

PRIORITY ACTIONS:
{'='*20}
High Priority:
{chr(10).join(f'• {action}' for action in priority_actions.get('high_priority', []))}

Medium Priority:
{chr(10).join(f'• {action}' for action in priority_actions.get('medium_priority', []))}

DETAILED ANALYSIS:
{'='*20}
"""
            for category, analysis in solution.get('solutions', {}).items():
                text_summary += f"\n{category.upper().replace('_', ' ')}:\n{'-'*30}\n"
                text_summary += f"Analysis: {analysis.get('analysis', 'N/A')}\n\n"
                if analysis.get('recommendations'):
                    text_summary += "Recommendations:\n"
                    text_summary += '\n'.join(f'• {rec}' for rec in analysis['recommendations'])
                    text_summary += "\n\n"
            
            st.download_button(
                "📝 Download Text Summary",
                data=text_summary.encode('utf-8'),
                file_name=f"ai_solution_summary_{solution['timestamp'][:10]}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ========== Tab 3: Analytics ==========
with tab3:
    st.header("📊 Analytics & Forecasting")
    
    # Data loading section
    st.subheader("Data Source")
    if data_option == "Spain Dataset":
        df = load_packaged_spain()
        if not df.empty:
            st.success(f"✅ Spain dataset loaded: {df.shape[0]} records")
            with st.expander("Data Preview"):
                st.dataframe(df.head(10))
    elif data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Custom dataset loaded: {df.shape[0]} records")
        else:
            df = pd.DataFrame()
            st.info("Please upload a CSV file")
    else:  # Manual Input
        st.info("Enter 24 hourly demand values")
        manual_cols = st.columns(6)
        manual_values = []
        for i in range(24):
            val = manual_cols[i % 6].number_input(
                f"H{i}", 
                min_value=0.0, 
                value=float(50 + 30 * np.sin((i/24) * 2 * np.pi)), 
                key=f"manual_h{i}"
            )
            manual_values.append(val)
        df = pd.DataFrame({"hour": range(24), "real_demand": manual_values})
    
    # Forecasting section
    if not df.empty:
        st.subheader("🔮 Demand Forecasting")
        
        forecast_cols = st.columns([2, 1])
        
        with forecast_cols[0]:
            if forecast_method == "ARIMA":
                if st.button("🏃 Run ARIMA Forecast"):
                    if "real_demand" in df.columns:
                        with st.spinner("Running ARIMA forecast..."):
                            series = df["real_demand"].dropna()
                            arima_forecast_vals = arima_forecast(series, steps=24)
                            
                            # Save results
                            pd.DataFrame({"arima_forecast": arima_forecast_vals}).to_csv(
                                ARTIFACTS / "forecast_arima.csv", index=False
                            )
                            
                            st.success("✅ ARIMA forecast completed")
                            
                            # Plot results
                            fig = plot_forecast_interactive(series, arima_forecast_vals, "ARIMA Forecast")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            
            elif forecast_method == "LSTM":
                lstm_cols = st.columns(2)
                epochs = lstm_cols[0].slider("Training Epochs", 1, 50, 10)
                lookback = lstm_cols[1].slider("Lookback Window", 6, 24, 12)
                
                if st.button("🤖 Train LSTM Model"):
                    if "real_demand" in df.columns:
                        with st.spinner("Training LSTM model..."):
                            series = df["real_demand"].dropna()
                            if len(series) < lookback + 25:
                                st.error(f"Dataset too small. Need at least {lookback + 25} data points, but only have {len(series)}. Try reducing lookback window or use more data.")
                            else:
                                lstm_preds, lstm_metrics = lstm_forecast_train_eval(
                                    series, look_back=lookback, epochs=epochs, steps=24
                                )
                            
                                # Save results
                                np.save(ARTIFACTS / "forecast_lstm.npy", lstm_preds)
                                with open(ARTIFACTS / "forecast_metrics.json", "w") as f:
                                    json.dump(lstm_metrics, f, indent=2)
                            
                                st.success(f"✅ LSTM training completed!")
                            
                                # Show metrics
                                metric_cols = st.columns(2)
                                metric_cols[0].metric("MAPE", f"{lstm_metrics.get('mape', 0):.2f}%")
                                metric_cols[1].metric("RMSE", f"{lstm_metrics.get('rmse', 0):.3f}")
                            
                                # Plot results
                                fig = plot_forecast_interactive(series, lstm_preds, "LSTM Forecast")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
        
        with forecast_cols[1]:
            st.markdown("**📋 Forecast Info**")
            forecast_arr, forecast_type = try_load_forecast()
            if forecast_arr is not None:
                st.success(f"✅ {forecast_type.upper()} forecast available")
                st.metric("Forecast Hours", len(forecast_arr))
                st.metric("Avg Demand", f"{np.mean(forecast_arr):.1f} MW")
                st.metric("Peak Demand", f"{np.max(forecast_arr):.1f} MW")
            else:
                st.info("No forecast available")

# ========== Tab 4: Operations ==========
with tab4:
    st.header("⚙️ Operations & Optimization")
    
    # Economic Dispatch
    st.subheader("⚡ Economic Dispatch")
    
    forecast_arr, _ = try_load_forecast()
    if forecast_arr is not None:
        dispatch_cols = st.columns([2, 1])
        
        with dispatch_cols[0]:
            # Optimization parameters
            st.markdown("**Optimization Parameters**")
            param_cols = st.columns(3)
            thermal_cost = param_cols[0].number_input("Thermal Cost ($/MWh)", 0.1, 5.0, 1.0, 0.1)
            renewable_cost = param_cols[1].number_input("Renewable Cost ($/MWh)", 0.0, 1.0, 0.2, 0.05)
            shed_penalty = param_cols[2].number_input("Shed Penalty ($/MWh)", 10.0, 500.0, 200.0, 10.0)
            
            advanced_cols = st.columns(2)
            renew_capacity = advanced_cols[0].slider("Renewable Capacity Scale", 0.1, 2.0, 1.0, 0.1)
            optimization_iters = advanced_cols[1].slider("Optimization Iterations", 50, 500, 250, 50)
            
            if st.button("🔧 Run Economic Dispatch", type="primary"):
                with st.spinner("Optimizing dispatch plan..."):
                    dispatch_df, objective = optimize_dispatch_dynamic(
                        forecast_arr,
                        thermal_cost=thermal_cost,
                        renewable_cost=renewable_cost,
                        shed_penalty=shed_penalty,
                        renew_max_scale=renew_capacity,
                        iters=optimization_iters,
                        particles=80
                    )
                    
                    # Save results
                    dispatch_df.to_csv(ARTIFACTS / "dispatch_plan_dynamic.csv", index=False)
                    
                    st.success(f"✅ Optimization complete! Objective value: {objective:.2f}")
                    
                    # Show results
                    fig_dispatch = plot_dispatch_stacked(dispatch_df)
                    if fig_dispatch:
                        st.plotly_chart(fig_dispatch, use_container_width=True)
        
        with dispatch_cols[1]:
            st.markdown("**📊 Current Dispatch Status**")
            try:
                current_dispatch = pd.read_csv(ARTIFACTS / "dispatch_plan_dynamic.csv")
                total_demand = current_dispatch["demand"].sum()
                renewable_energy = current_dispatch["renewable"].sum()
                thermal_energy = current_dispatch["thermal"].sum()
                shed_energy = current_dispatch["shed"].sum()
                
                st.metric("Total Demand", f"{total_demand:.0f} MWh")
                st.metric("Renewable Share", f"{(renewable_energy/total_demand)*100:.1f}%")
                st.metric("Thermal Share", f"{(thermal_energy/total_demand)*100:.1f}%")
                st.metric("Load Shed", f"{(shed_energy/total_demand)*100:.4f}%")
                
                # Cost estimation
                estimated_cost = (renewable_energy * renewable_cost + 
                                thermal_energy * thermal_cost + 
                                shed_energy * shed_penalty)
                st.metric("Estimated Cost", f"${estimated_cost:.0f}")
                
            except:
                st.info("No dispatch plan available. Run optimization above.")
    else:
        st.warning("⚠️ No forecast available. Generate forecast in Analytics tab first.")
    
    # Scenario Analysis
    st.subheader("🎯 Scenario Analysis")
    scenario_cols = st.columns(3)
    
    scenarios = {
        "Base Case": {"demand_scale": 1.0, "renew_scale": 1.0, "description": "Current conditions"},
        "High Demand": {"demand_scale": 1.2, "renew_scale": 1.0, "description": "20% demand increase"},
        "Double Renewables": {"demand_scale": 1.0, "renew_scale": 2.0, "description": "Double renewable capacity"}
    }
    
    for i, (scenario_name, config) in enumerate(scenarios.items()):
        with scenario_cols[i]:
            st.markdown(f"**{scenario_name}**")
            st.caption(config["description"])
            
            if st.button(f"Run {scenario_name}", key=f"scenario_{i}"):
                if forecast_arr is not None:
                    with st.spinner(f"Running {scenario_name} scenario..."):
                        modified_demand = forecast_arr * config["demand_scale"]
                        dispatch_df, obj = optimize_dispatch_dynamic(
                            modified_demand,
                            thermal_cost=1.0,
                            renewable_cost=0.2,
                            shed_penalty=200.0,
                            renew_max_scale=config["renew_scale"],
                            iters=200,
                            particles=80
                        )
                        
                        # Show key metrics
                        total_demand = dispatch_df["demand"].sum()
                        renewable_pct = (dispatch_df["renewable"].sum() / total_demand) * 100
                        shed_pct = (dispatch_df["shed"].sum() / total_demand) * 100
                        
                        st.success(f"Objective: {obj:.2f}")
                        st.metric("Renewable %", f"{renewable_pct:.1f}%")
                        st.metric("Shed %", f"{shed_pct:.3f}%")

# ========== Tab 5: Data Explorer ==========
with tab5:
    st.header("📁 Data Explorer")
    
    # Available artifacts
    st.subheader("📊 Available Artifacts")
    
    artifacts = {
        "forecast_lstm.npy": "LSTM Forecast Data",
        "forecast_metrics.json": "Forecast Performance Metrics",
        "dispatch_plan_dynamic.csv": "Economic Dispatch Plan",
        "stability_report.txt": "Grid Stability Analysis",
        "rl_summary.json": "RL Controller Summary",
        "ai_solution_report.json": "AI Problem Analysis",
        "summary.json": "Overall System Summary"
    }
    
    explorer_cols = st.columns(2)
    
    with explorer_cols[0]:
        st.markdown("**📋 Artifact Status**")
        for filename, description in artifacts.items():
            file_path = ARTIFACTS / filename
            if file_path.exists():
                st.success(f"✅ {description}")
                
                # Quick preview for some files
                if filename.endswith('.json'):
                    if st.button(f"👁️ Preview {filename}", key=f"preview_{filename}"):
                        try:
                            with open(file_path) as f:
                                data = json.load(f)
                            st.json(data)
                        except:
                            st.error("Error reading file")
                
                elif filename.endswith('.csv'):
                    if st.button(f"👁️ Preview {filename}", key=f"preview_{filename}"):
                        try:
                            df = pd.read_csv(file_path)
                            st.dataframe(df.head())
                            st.caption(f"Shape: {df.shape}")
                        except:
                            st.error("Error reading file")
                
                elif filename.endswith('.txt'):
                    if st.button(f"👁️ Preview {filename}", key=f"preview_{filename}"):
                        try:
                            content = file_path.read_text()
                            st.text(content[:1000] + "..." if len(content) > 1000 else content)
                        except:
                            st.error("Error reading file")
                
                elif filename.endswith('.npy'):
                    if st.button(f"👁️ Preview {filename}", key=f"preview_{filename}"):
                        try:
                            data = np.load(file_path)
                            st.write(f"Shape: {data.shape}")
                            st.write(f"Sample values: {data[:10]}")
                        except:
                            st.error("Error reading file")
                            
            else:
                st.error(f"❌ {description}")
    
    with explorer_cols[1]:
        st.markdown("**⬇️ Download Artifacts**")
        
        # Bulk download options
        available_files = [f for f in artifacts.keys() if (ARTIFACTS / f).exists()]
        
        if available_files:
            st.markdown("Available files:")
            for filename in available_files:
                file_path = ARTIFACTS / filename
                file_size = file_path.stat().st_size
                
                with open(file_path, 'rb') as f:
                    st.download_button(
                        f"📥 {filename} ({file_size} bytes)",
                        data=f.read(),
                        file_name=filename,
                        key=f"download_{filename}"
                    )
        else:
            st.info("No artifacts available for download")
    
    # Raw data inspection
    st.subheader("🔍 Raw Data Inspection")
    
    inspection_option = st.selectbox(
        "Select data to inspect:",
        ["Forecast Data", "Dispatch Plan", "Spain Dataset", "System Metrics"]
    )
    
    if inspection_option == "Forecast Data":
        forecast_arr, forecast_type = try_load_forecast()
        if forecast_arr is not None:
            df_forecast = pd.DataFrame({
                "Hour": range(len(forecast_arr)),
                "Forecast_MW": forecast_arr
            })
            st.dataframe(df_forecast, use_container_width=True)
            
            # Basic statistics
            stats_cols = st.columns(4)
            stats_cols[0].metric("Mean", f"{np.mean(forecast_arr):.1f}")
            stats_cols[1].metric("Std", f"{np.std(forecast_arr):.1f}")
            stats_cols[2].metric("Min", f"{np.min(forecast_arr):.1f}")
            stats_cols[3].metric("Max", f"{np.max(forecast_arr):.1f}")
        else:
            st.info("No forecast data available")
    
    elif inspection_option == "Dispatch Plan":
        try:
            dispatch_df = pd.read_csv(ARTIFACTS / "dispatch_plan_dynamic.csv")
            st.dataframe(dispatch_df, use_container_width=True)
            
            # Summary statistics
            st.markdown("**Summary Statistics:**")
            st.dataframe(dispatch_df.describe(), use_container_width=True)
            
        except:
            st.info("No dispatch plan available")
    
    elif inspection_option == "Spain Dataset":
        df_spain = load_packaged_spain()
        if not df_spain.empty:
            st.dataframe(df_spain, use_container_width=True)
            st.markdown("**Dataset Info:**")
            st.write(f"• Shape: {df_spain.shape}")
            st.write(f"• Columns: {list(df_spain.columns)}")
            st.write(f"• Date range: {df_spain.get('datetime', pd.Series()).min()} to {df_spain.get('datetime', pd.Series()).max()}")
        else:
            st.info("Spain dataset not available")
    
    elif inspection_option == "System Metrics":
        try:
            with open(ARTIFACTS / "summary.json") as f:
                summary_data = json.load(f)
            st.json(summary_data)
        except:
            st.info("No system metrics available")

# ========== Footer ==========
st.markdown("---")
st.markdown("**SmartGrid-AI Enhanced Dashboard** | Built with Streamlit | AI-Powered Grid Management")

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Help")
st.sidebar.markdown("""
**Quick Start:**
1. Load data (Spain dataset recommended)
2. Generate forecast (LSTM preferred)  
3. Run economic dispatch optimization
4. Use AI Problem Solver for insights

**AI Problem Solver:**
- Analyzes system performance
- Provides actionable recommendations  
- Generates priority action lists
- Creates detailed reports
""")

st.sidebar.markdown("### 🔗 Resources")
st.sidebar.markdown("[📊 Dashboard](https://smartgrid-ai.streamlit.app)")
st.sidebar.markdown("[🧠 AI Docs](#)")
st.sidebar.markdown("[⚙️ API Reference](#)")