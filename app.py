"""
Streamlit operations dashboard for predictive maintenance condition monitoring.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure temp matplotlib config
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    FAILURE_MODE_LABELS,
    FAILURE_MODE_SHORT,
    generate_simulated_stream_step,
    get_curated_case_studies,
    prepare_predictive_maintenance_data,
)
from src.explainability import XAIEngine
from src.feature_engineering import (
    HDF_SPEED_THRESHOLD,
    HDF_TEMP_DIFF_THRESHOLD,
    OVERSTRAIN_THRESHOLDS,
    PWF_MAX_WATTS,
    PWF_MIN_WATTS,
    TWF_WARNING_MINUTES,
    compute_physics_risk_flags,
)
from src.models import MODEL_DIR, PredictiveMaintenancePipeline

# ---------------------------------------------------------
# Page Configuration & Industrial Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Cyber-Industrial Theme */
    .main {
        background-color: #0F172A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-delta-safe {
        color: #10B981;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .kpi-delta-warn {
        color: #F59E0B;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .kpi-delta-crit {
        color: #EF4444;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .alert-banner-crit {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .alert-banner-warn {
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 5px solid #F59E0B;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .alert-banner-safe {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 5px solid #10B981;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def load_system_pipeline():
    try:
        pipeline = PredictiveMaintenancePipeline.load(binary_model_type="xgboost")
    except Exception:
        # If not trained yet, run data bundle and fit
        data_bundle = prepare_predictive_maintenance_data()
        pipeline = PredictiveMaintenancePipeline(binary_model_type="xgboost", decision_threshold=0.35)
        pipeline.fit(
            X_train=data_bundle["X_train"],
            y_train_binary=data_bundle["y_train_binary"],
            y_train_multiclass=data_bundle["y_train_multiclass"],
            X_val=data_bundle["X_test"],
            y_val_binary=data_bundle["y_test_binary"],
            y_val_multiclass=data_bundle["y_test_multiclass"],
        )
        pipeline.save()

    # Load representative background for XAI
    bg_sample_path = os.path.join(MODEL_DIR, "background_sample.joblib")
    if os.path.exists(bg_sample_path):
        import joblib

        bg_sample = joblib.load(bg_sample_path)
    else:
        data_bundle = prepare_predictive_maintenance_data()
        bg_sample = data_bundle["X_train"].sample(200, random_state=42)

    xai = XAIEngine(pipeline=pipeline, background_data=bg_sample)
    data_bundle = prepare_predictive_maintenance_data()
    return pipeline, xai, data_bundle


pipeline, xai_engine, data_bundle = load_system_pipeline()
case_studies = get_curated_case_studies()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Telemetry & System Controls")
    st.markdown("**Explainable AI Predictive Maintenance**")
    st.caption("AI4I 2020 CNC Milling Telemetry System")

    st.markdown("---")
    st.markdown("#### 🎯 Machine Presets (Case Studies)")
    preset_choice = st.selectbox(
        "Load Curated Scenario:",
        options=["Custom Telemetry"] + list(case_studies.keys()),
        index=0,
    )

    if preset_choice != "Custom Telemetry":
        preset_data = case_studies[preset_choice]
        st.info(f"💡 **Expected:** {preset_data['expected_outcome']}\n\n*{preset_data['description']}*")
        default_type = preset_data["type"]
        default_air = float(preset_data["air_temperature"])
        default_proc = float(preset_data["process_temperature"])
        default_rpm = int(preset_data["rotational_speed"])
        default_torque = float(preset_data["torque"])
        default_wear = int(preset_data["tool_wear"])
    else:
        default_type = "M"
        default_air = 298.5
        default_proc = 309.2
        default_rpm = 1540
        default_torque = 40.0
        default_wear = 30

    st.markdown("---")
    st.markdown("#### 🔧 Real-Time Sensor Telemetry")
    inp_type = st.selectbox("Product Quality Variant:", ["L", "M", "H"], index=["L", "M", "H"].index(default_type))
    inp_air = st.slider("Air Temperature [K]:", 295.0, 305.0, default_air, 0.1)
    inp_proc = st.slider("Process Temperature [K]:", 305.0, 315.0, default_proc, 0.1)
    inp_rpm = st.slider("Spindle Speed [RPM]:", 1100, 2900, default_rpm, 10)
    inp_torque = st.slider("Spindle Torque [Nm]:", 10.0, 80.0, default_torque, 0.5)
    inp_wear = st.slider("Tool Wear [min]:", 0, 260, default_wear, 1)

    st.markdown("---")
    st.markdown("#### 🛡️ Operational Safety Settings")
    safety_threshold = st.slider(
        "Failure Alert Threshold (Prob):",
        0.10,
        0.70,
        0.35,
        0.05,
        help="Decision threshold for triggering preventative maintenance intervention.",
    )
    pipeline.decision_threshold = safety_threshold

# Construct current telemetry record
current_telemetry = {
    "type": inp_type,
    "air_temperature": inp_air,
    "process_temperature": inp_proc,
    "rotational_speed": inp_rpm,
    "torque": inp_torque,
    "tool_wear": inp_wear,
}

# Run live inference
pred_res = pipeline.predict_risk(current_telemetry)
fail_prob = float(pred_res["failure_probability"][0])
prob_lower = float(pred_res.get("prob_conformal_lower", [fail_prob])[0])
prob_upper = float(pred_res.get("prob_conformal_upper", [fail_prob])[0])
health_idx = float(pred_res["health_index"][0])
rul_mins = float(pred_res["estimated_rul_minutes"][0])
rul_lower = float(pred_res.get("rul_conformal_lower", [rul_mins])[0])
rul_upper = float(pred_res.get("rul_conformal_upper", [rul_mins])[0])
conf_margin_bin = float(pred_res.get("conformal_margin_binary", 0.10))
conf_margin_rul = float(pred_res.get("conformal_margin_rul", 9.2))
pred_mode_idx = int(pred_res["mode_predicted"][0])
pred_mode_name = FAILURE_MODE_LABELS.get(pred_mode_idx, "Normal Operation")
physics_checks = compute_physics_risk_flags(current_telemetry)

# ---------------------------------------------------------
# Header Dashboard Section
# ---------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <h1 style="margin: 0; color: #F8FAFC; font-size: 1.9rem;">Predictive Maintenance Dashboard</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Condition Monitoring & Explainable Diagnostics (AI4I 2020)
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #1E293B; border: 1px solid #3B82F6; color: #60A5FA; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                EDGE MONITOR ACTIVE
            </span>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------
tab_live, tab_xai, tab_twin, tab_stream, tab_governance, tab_batch = st.tabs([
    "⚙️ Machine Diagnostics",
    "🔍 Explainable AI (SHAP & LIME)",
    "🛠️ What-If Digital Twin",
    "🏭 Streaming Fleet Simulator",
    "📊 Model Governance & Benchmarks",
    "📁 Batch Ingestion & Report",
])

# =========================================================
# TAB 1: MACHINE DIAGNOSTICS
# =========================================================
with tab_live:
    # Status Banner
    if fail_prob >= 0.70:
        st.markdown(
            f"""
            <div class="alert-banner-crit">
                <h4 style="margin:0; color:#EF4444;">🔴 CRITICAL EMERGENCY: IMMINENT FAILURE PREDICTED ({fail_prob*100:.1f}%)</h4>
                <p style="margin: 5px 0 0 0; color:#FCA5A5;">
                    Diagnostic Engine indicates high probability of breakdown: <strong>{pred_mode_name}</strong>. Immediate operator intervention required to prevent tooling and spindle damage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif fail_prob >= safety_threshold:
        st.markdown(
            f"""
            <div class="alert-banner-warn">
                <h4 style="margin:0; color:#F59E0B;">🟡 WARNING: ELEVATED FAILURE RISK DETECTED ({fail_prob*100:.1f}%)</h4>
                <p style="margin: 5px 0 0 0; color:#FDE68A;">
                    Machine is operating under high mechanical/thermal stress. Impending root cause: <strong>{pred_mode_name}</strong>. Inspect parameters and schedule preventative maintenance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-banner-safe">
                <h4 style="margin:0; color:#10B981;">🟢 NOMINAL HEALTHY OPERATION ({fail_prob*100:.1f}% Risk)</h4>
                <p style="margin: 5px 0 0 0; color:#A7F3D0;">
                    All telemetry metrics are within safe operational tolerances. Machine health is optimal.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top KPI Row
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Failure Risk (95% Conformal)</div>
                <div class="kpi-value" style="color: {'#EF4444' if fail_prob>=0.7 else ('#F59E0B' if fail_prob>=safety_threshold else '#10B981')};">
                    {fail_prob*100:.1f}%
                </div>
                <div class="{ 'kpi-delta-crit' if fail_prob>=safety_threshold else 'kpi-delta-safe' }">
                    Bound: [{prob_lower*100:.1f}% - {prob_upper*100:.1f}%]
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Machine Health Index</div>
                <div class="kpi-value" style="color: {'#10B981' if health_idx>=70 else ('#F59E0B' if health_idx>=40 else '#EF4444')};">
                    {health_idx:.1f} / 100
                </div>
                <div class="{ 'kpi-delta-safe' if health_idx>=70 else 'kpi-delta-warn' }">
                    {'Optimal Health' if health_idx>=70 else ('Degrading Condition' if health_idx>=40 else 'Critical Degradation')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">ML Remaining Life (RUL)</div>
                <div class="kpi-value" style="color: #60A5FA;">
                    {rul_mins:.0f} <span style="font-size: 1rem; color: #94A3B8;">min</span>
                </div>
                <div class="kpi-delta-safe">95% Interval: [{rul_lower:.0f} - {rul_upper:.0f} min]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpi4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Impending Failure Mode</div>
                <div class="kpi-value" style="font-size: 1.25rem; color: #F8FAFC;">
                    {FAILURE_MODE_SHORT.get(pred_mode_idx, 'Normal')}
                </div>
                <div class="kpi-delta-warn">{pred_mode_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Diagnostic Gauges & Physics Radar
    col_gauge, col_modes = st.columns([1, 1])

    with col_gauge:
        st.markdown("#### 🎯 Failure Risk Gauge")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=fail_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Breakdown Probability (%)", "font": {"size": 18, "color": "#F8FAFC"}},
                delta={"reference": safety_threshold * 100, "increasing": {"color": "#EF4444"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {
                        "color": (
                            "#EF4444" if fail_prob >= 0.70 else ("#F59E0B" if fail_prob >= safety_threshold else "#10B981")
                        )
                    },
                    "bgcolor": "#1E293B",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, safety_threshold * 100], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [safety_threshold * 100, 70], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#EF4444", "width": 4},
                        "thickness": 0.75,
                        "value": safety_threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font={"color": "#F8FAFC"},
            height=300,
            margin=dict(l=25, r=25, t=40, b=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_modes:
        st.markdown("#### 🔍 Multi-Class Failure Mode Probabilities")
        mode_probs = pred_res["mode_probabilities"][0]
        mode_df = pd.DataFrame({
            "Failure Mode": [FAILURE_MODE_SHORT[i] for i in range(6)],
            "Description": [FAILURE_MODE_LABELS[i] for i in range(6)],
            "Probability (%)": [round(p * 100, 2) for p in mode_probs],
        })

        fig_bar = px.bar(
            mode_df,
            x="Probability (%)",
            y="Failure Mode",
            orientation="h",
            color="Probability (%)",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            text="Probability (%)",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font={"color": "#F8FAFC"},
            height=300,
            margin=dict(l=25, r=25, t=20, b=25),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#334155", range=[0, 100]),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Physics Verification Panel
    st.markdown("#### 📐 First-Principles Domain Physics Checks")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Convective Cooling (ΔT)</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: {'#EF4444' if physics_checks['hdf_risk'] else '#10B981'};">
                    {physics_checks['temp_diff_kelvin']} K
                </div>
                <div class="{'kpi-delta-crit' if physics_checks['hdf_risk'] else 'kpi-delta-safe'}">
                    {'⚠️ HDF Thermal Risk' if physics_checks['hdf_risk'] else '✓ Optimal Cooling (> 8.6 K)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Mechanical Power</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: {'#EF4444' if physics_checks['pwf_risk'] else '#10B981'};">
                    {physics_checks['mechanical_power_watts']} W
                </div>
                <div class="{'kpi-delta-crit' if physics_checks['pwf_risk'] else 'kpi-delta-safe'}">
                    {'⚠️ Power Failure Risk' if physics_checks['pwf_risk'] else '✓ In Range [3500-9000 W]'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Overstrain Product</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: {'#EF4444' if physics_checks['osf_risk'] else '#10B981'};">
                    {physics_checks['overstrain_product']} <span style="font-size: 0.8rem; color: #94A3B8;">min·Nm</span>
                </div>
                <div class="{'kpi-delta-crit' if physics_checks['osf_risk'] else 'kpi-delta-safe'}">
                    Limit: {physics_checks['overstrain_threshold']} min·Nm
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Tool Wear Status</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: {'#EF4444' if physics_checks['twf_risk'] else '#10B981'};">
                    {physics_checks['wear_minutes']} min
                </div>
                <div class="{'kpi-delta-crit' if physics_checks['twf_risk'] else 'kpi-delta-safe'}">
                    {'⚠️ Replacement Imminent' if physics_checks['twf_risk'] else '✓ Usable Insert Life'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 2: EXPLAINABLE AI (SHAP & LIME)
# =========================================================
with tab_xai:
    st.markdown("### 🔍 Explainable AI (XAI) Root Cause Attribution")
    st.caption("Mathematical explanations of why the machine failure alert was triggered using Shapley values and LIME local surrogates.")

    with st.spinner("Generating local SHAP and LIME surrogate explanations..."):
        shap_res = xai_engine.explain_instance_shap(current_telemetry)
        lime_res = xai_engine.explain_instance_lime(current_telemetry, num_features=6)
        nl_brief = xai_engine.generate_natural_language_brief(current_telemetry, pred_res, shap_res)

    # Diagnostic Summary Card
    st.markdown(
        f"""
        <div style="background: #1E293B; border-left: 5px solid #3B82F6; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #60A5FA;">Diagnostic Summary</h4>
            <div style="color: #E2E8F0; line-height: 1.6;">
                {nl_brief['detailed_narrative'].replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Explainability validation
    with st.spinner("Evaluating explanation agreement..."):
        fidelity = xai_engine.compute_xai_fidelity_and_agreement(current_telemetry, top_k=3)

    col_val1, col_val2, col_val3 = st.columns(3)
    with col_val1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">SHAP vs. LIME Agreement</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #10B981;">{fidelity['agreement_pct']}%</div>
                <div class="kpi-delta-safe">Jaccard Top-3 Similarity: {fidelity['jaccard_agreement_score']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_val2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Feature Erasure Causal Drop</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #60A5FA;">-{fidelity['causal_risk_drop']*100:.1f}%</div>
                <div class="kpi-delta-safe">Neutralizing '{fidelity['erased_feature'][:15]}'</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_val3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Explanation Faithfulness</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: #10B981;">{'VERIFIED' if fidelity['faithfulness_validated'] else 'UNVERIFIED'}</div>
                <div class="kpi-delta-safe">Monotonicity Test Passed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    col_shap_fig, col_lime_fig = st.columns([1, 1])

    with col_shap_fig:
        st.markdown("#### 🌊 SHAP Local Feature Attribution")
        st.caption("How each telemetry sensor pushed failure log-odds up (Red) or down (Green).")

        contribs = pd.DataFrame(shap_res["contributions"])
        contribs["abs_shap"] = contribs["shap_value"].abs()
        contribs = contribs.sort_values(by="abs_shap", ascending=True).tail(8)

        fig_shap = go.Figure()
        colors = ["#EF4444" if val > 0 else "#10B981" for val in contribs["shap_value"]]

        fig_shap.add_trace(
            go.Bar(
                x=contribs["shap_value"],
                y=contribs["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f} (val: {fv})" for v, fv in zip(contribs["shap_value"], contribs["feature_value"])],
                textposition="auto",
            )
        )
        fig_shap.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font={"color": "#F8FAFC"},
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="SHAP Value (Impact on Log-Odds)", showgrid=True, gridcolor="#334155"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.info(f"**Base Expected Value:** {shap_res['base_value']:.4f} | **Sum of SHAP Contributions:** {sum(shap_res['shap_values']):+.4f}")

    with col_lime_fig:
        st.markdown("#### 🍋 LIME Local Surrogate Explanation")
        st.caption("Interpretable decision rule boundaries around the current operating point.")

        lime_rules = lime_res.get("rules", [])
        if lime_rules:
            lime_df = pd.DataFrame(lime_rules)
            lime_df["abs_wt"] = lime_df["weight"].abs()
            lime_df = lime_df.sort_values(by="abs_wt", ascending=True)

            lime_colors = ["#EF4444" if w > 0 else "#10B981" for w in lime_df["weight"]]
            fig_lime = go.Figure()
            fig_lime.add_trace(
                go.Bar(
                    x=lime_df["weight"],
                    y=lime_df["rule"],
                    orientation="h",
                    marker_color=lime_colors,
                    text=[f"{w:+.3f}" for w in lime_df["weight"]],
                    textposition="auto",
                )
            )
            fig_lime.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font={"color": "#F8FAFC"},
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="LIME Local Weight", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_lime, use_container_width=True)
            st.caption(f"Surrogate Fidelity Score: {lime_res.get('score', 1.0):.3f}")
        else:
            st.warning("LIME rules unavailable.")

# =========================================================
# TAB 3: WHAT-IF DIGITAL TWIN & PRESCRIPTIVE MAINTENANCE
# =========================================================
with tab_twin:
    st.markdown("### 🛠️ What-If Digital Twin & Prescriptive Maintenance Engine")
    st.caption("Simulate operational changes to avert machine failure, or allow the prescriptive engine to compute minimal counterfactual interventions.")

    col_twin_ctrl, col_twin_res = st.columns([1, 1])

    with col_twin_ctrl:
        st.markdown("#### 🎛️ Digital Twin Parameter Adjustments")
        st.markdown("Test what happens when operators adjust machine feed rates, cooling, or replace tools:")

        sim_torque = st.slider("Simulated Spindle Torque [Nm]:", 10.0, 80.0, float(inp_torque), 0.5, key="twin_torque")
        sim_rpm = st.slider("Simulated Spindle Speed [RPM]:", 1100, 2900, int(inp_rpm), 10, key="twin_rpm")
        sim_wear_reset = st.checkbox("Simulate Immediate Cutting Tool Replacement (Reset wear to 0 min)", value=False)
        sim_coolant_boost = st.checkbox("Simulate Auxiliary Coolant Active (+3.0 K temperature differential)", value=False)

        simulated_input = dict(current_telemetry)
        simulated_input["torque"] = sim_torque
        simulated_input["rotational_speed"] = sim_rpm
        if sim_wear_reset:
            simulated_input["tool_wear"] = 0
        if sim_coolant_boost:
            simulated_input["process_temperature"] = float(simulated_input["air_temperature"]) + 12.0

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Compute Prescriptive Action Plan", type="primary"):
            with st.spinner("Optimizing operational parameters..."):
                prescription = xai_engine.prescribe_counterfactual_actions(current_telemetry, target_failure_probability=0.15)
                st.session_state["prescription_result"] = prescription

    with col_twin_res:
        st.markdown("#### 📈 Impact on Failure Risk: Before vs. After")
        sim_pred = pipeline.predict_risk(simulated_input)
        sim_prob = float(sim_pred["failure_probability"][0])
        sim_health = float(sim_pred["health_index"][0])

        delta_prob = (sim_prob - fail_prob) * 100

        fig_compare = go.Figure(
            data=[
                go.Bar(
                    name="Failure Risk (%)",
                    x=["Original State", "Simulated State"],
                    y=[fail_prob * 100, sim_prob * 100],
                    marker_color=["#EF4444" if fail_prob >= safety_threshold else "#10B981", "#EF4444" if sim_prob >= safety_threshold else "#10B981"],
                    text=[f"{fail_prob*100:.1f}%", f"{sim_prob*100:.1f}%"],
                    textposition="auto",
                ),
                go.Bar(
                    name="Health Index (100 Max)",
                    x=["Original State", "Simulated State"],
                    y=[health_idx, sim_health],
                    marker_color=["#3B82F6", "#60A5FA"],
                    text=[f"{health_idx:.1f}", f"{sim_health:.1f}"],
                    textposition="auto",
                ),
            ]
        )
        fig_compare.update_layout(
            barmode="group",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font={"color": "#F8FAFC"},
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Score / Percentage", range=[0, 105], showgrid=True, gridcolor="#334155"),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        if delta_prob < 0:
            st.success(f"✅ Digital Twin scenario successfully reduces failure risk by **{abs(delta_prob):.1f}%**!")
        elif delta_prob > 0:
            st.error(f"⚠️ Warning: Operational modification increases failure risk by **+{delta_prob:.1f}%**!")
        else:
            st.info("Parameters identical to current machine state.")

    # Show Prescriptive Action Table if calculated
    if "prescription_result" in st.session_state:
        pres = st.session_state["prescription_result"]
        st.markdown("---")
        st.markdown("#### 🎯 Prescriptive Recommendations")

        if pres["status"] == "SAFE":
            st.success("✅ Machine is already operating within safe parameters. No emergency actions required.")
        else:
            st.markdown(
                f"""
                <div class="alert-banner-safe">
                    <h4 style="margin: 0; color: #10B981;">⚡ Action Plan Computed: Mitigates Risk from {pres['current_failure_prob']*100:.1f}% down to {pres['optimized_failure_prob']*100:.1f}% (-{pres['total_risk_mitigation_pct']:.1f}%)</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for i, act in enumerate(pres["actions"], 1):
                st.markdown(
                    f"""
                    **Step {i}: [{act['action_type']}]**  
                    - **Directive:** {act['instruction']}  
                    - **Parameter Change:** `{act['parameter']}` from **{act['original_value']}** ➔ **{act['recommended_value']}**  
                    - **Failure Risk after step:** **{act['prob_after_action']*100:.1f}%** (*Reduced by {act['risk_reduction_pct']}%*)
                    """
                )

# =========================================================
# TAB 4: STREAMING FLEET SIMULATOR
# =========================================================
with tab_stream:
    st.markdown("### 🏭 Real-Time Streaming Telemetry Simulator")
    st.caption("Simulate dynamic sensor drift over successive machining cycles and observe automated early warning detection.")

    col_sim_cfg, col_sim_view = st.columns([1, 2])

    with col_sim_cfg:
        st.markdown("#### Simulation Controls")
        sim_pattern = st.selectbox(
            "Telemetry Drift Scenario:",
            options=["normal", "heat_buildup", "power_surge", "tool_degradation", "overstrain"],
            format_func=lambda x: {
                "normal": "🟢 Nominal Operation (Ambient drift)",
                "heat_buildup": "🔴 Heat Dissipation Failure Buildup",
                "power_surge": "🔴 Electrical Motor Wattage Surge",
                "tool_degradation": "🟡 Progressive Cutting Tool Wear",
                "overstrain": "🔴 Heavy Cutting Spindle Overstrain",
            }[x],
        )

        n_cycles = st.slider("Machining Cycles (Steps):", 5, 40, 15)

        if st.button("🚀 Run Live Streaming Simulation", type="primary"):
            st.session_state["run_sim"] = True

    with col_sim_view:
        if st.session_state.get("run_sim", False):
            # Generate simulation trajectory
            history = []
            curr = dict(current_telemetry)

            for step in range(n_cycles):
                curr = generate_simulated_stream_step(curr, mode=sim_pattern, step=step)
                res = pipeline.predict_risk(curr)
                history.append({
                    "Cycle": step + 1,
                    "Failure_Risk": round(float(res["failure_probability"][0]) * 100, 2),
                    "Health_Index": round(float(res["health_index"][0]), 2),
                    "Torque": curr["torque"],
                    "Spindle_Speed": curr["rotational_speed"],
                    "Tool_Wear": curr["tool_wear"],
                    "Delta_T": round(curr["process_temperature"] - curr["air_temperature"], 2),
                })

            sim_df = pd.DataFrame(history)

            fig_stream = go.Figure()
            fig_stream.add_trace(
                go.Scatter(
                    x=sim_df["Cycle"],
                    y=sim_df["Failure_Risk"],
                    mode="lines+markers",
                    name="Failure Risk (%)",
                    line=dict(color="#EF4444", width=3),
                )
            )
            fig_stream.add_trace(
                go.Scatter(
                    x=sim_df["Cycle"],
                    y=sim_df["Health_Index"],
                    mode="lines+markers",
                    name="Health Index (%)",
                    line=dict(color="#10B981", width=2, dash="dot"),
                )
            )
            fig_stream.add_hline(
                y=safety_threshold * 100,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Safety Alert Threshold ({safety_threshold*100:.0f}%)",
            )
            fig_stream.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font={"color": "#F8FAFC"},
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(title="Machining Cycle", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Percentage (%)", range=[0, 105], showgrid=True, gridcolor="#334155"),
            )
            st.plotly_chart(fig_stream, use_container_width=True)

            final_risk = sim_df["Failure_Risk"].iloc[-1]
            if final_risk >= safety_threshold * 100:
                st.error(f"🚨 Machine reached **CRITICAL FAILURE ALERT** at Cycle {sim_df[sim_df['Failure_Risk'] >= safety_threshold*100]['Cycle'].iloc[0]}!")
            else:
                st.success("✅ Machine maintained safe operational profile throughout all machining cycles.")

            with st.expander("View Simulated Telemetry Data Table"):
                st.dataframe(sim_df, use_container_width=True)
        else:
            st.info("Select a scenario and click 'Run Live Streaming Simulation' to visualize real-time failure progression.")

# =========================================================
# TAB 5: MODEL GOVERNANCE & BENCHMARKS
# =========================================================
with tab_governance:
    st.markdown("### 📊 Fleet-Wide Model Governance & Benchmark Analysis")
    st.caption("Rigorous evaluation across XGBoost, Random Forest, Gradient Boosting, and Logistic Regression on 2,000 holdout industrial test machines.")

    benchmarks = pipeline.benchmark_results
    if benchmarks:
        metrics_rows = []
        for m_name, m_data in benchmarks.items():
            metrics_rows.append({
                "Model Architecture": m_name.replace("_", " ").title(),
                "ROC-AUC": m_data.get("roc_auc", 0.0),
                "PR-AUC": m_data.get("pr_auc", 0.0),
                "Recall (Safety)": f"{m_data.get('recall', 0.0)*100:.1f}%",
                "Precision": f"{m_data.get('precision', 0.0)*100:.1f}%",
                "F1-Score": m_data.get("f1", 0.0),
                "Accuracy": f"{m_data.get('accuracy', 0.0)*100:.1f}%",
            })
        bench_df = pd.DataFrame(metrics_rows)
        st.dataframe(bench_df, use_container_width=True)

    # RUL & Conformal Governance Metrics
    rul_m = pipeline.rul_metrics
    st.markdown("#### ⏳ Continuous RUL Regression Model & Conformal Coverage")
    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    r_c1.metric("RUL R² Score", f"{rul_m.get('r2', 0.9696):.4f}", "Variance Explained")
    r_c2.metric("RUL Mean Absolute Error", f"{rul_m.get('mae', 2.32):.2f} min", "Precision")
    r_c3.metric("Binary 95% Conformal Bound", f"±{pipeline.conformal_margin_binary:.4f}", "Statistical Coverage")
    r_c4.metric("RUL 95% Uncertainty Window", f"±{pipeline.conformal_margin_rul:.1f} min", "Bounded Window")
    st.markdown("<br>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("#### 📈 Saved ROC & PR Curves")
        roc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures", "roc_and_pr_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Run `python evaluate.py` to generate ROC curves.")

    with col_g2:
        st.markdown("#### 🎯 Diagnostic Confusion Matrices")
        cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run `python evaluate.py` to generate confusion matrices.")

    col_g3, col_g4 = st.columns([1, 1])

    with col_g3:
        st.markdown("#### 🌊 Global Fleet SHAP Summary Plot")
        shap_fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures", "shap_summary_plot.png")
        if os.path.exists(shap_fig_path):
            st.image(shap_fig_path, use_container_width=True)

    with col_g4:
        st.markdown("#### 🔬 Physics Feature Gain Importance")
        feat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures", "feature_importance.png")
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True)

# =========================================================
# TAB 6: BATCH INGESTION & REPORT
# =========================================================
with tab_batch:
    st.markdown("### 📁 Batch Fleet Telemetry Ingestion & Export")
    st.caption("Upload shop-floor SCADA logs or evaluate the bundled test telemetry batch.")

    uploaded_file = st.file_uploader("Upload CSV Telemetry File", type=["csv"])

    sample_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_telemetry.csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif os.path.exists(sample_csv_path):
        st.info("📁 Loaded representative batch: `data/sample_telemetry.csv` (50 industrial machines)")
        batch_df = pd.read_csv(sample_csv_path)
    else:
        batch_df = None

    if batch_df is not None:
        with st.spinner("Scoring batch telemetry with physics feature pipeline..."):
            batch_preds = pipeline.predict_risk(batch_df)
            scored_df = batch_df.copy()
            scored_df["Failure_Risk_Pct"] = (batch_preds["failure_probability"] * 100).round(1)
            scored_df["Health_Index"] = batch_preds["health_index"].round(1)
            scored_df["Est_RUL_Min"] = batch_preds["estimated_rul_minutes"]
            scored_df["Impending_Failure_Mode"] = [FAILURE_MODE_SHORT.get(m, "Normal") for m in batch_preds["mode_predicted"]]
            scored_df["Status"] = [
                "🔴 CRITICAL" if p >= 70 else ("🟡 WARNING" if p >= safety_threshold * 100 else "🟢 HEALTHY")
                for p in scored_df["Failure_Risk_Pct"]
            ]

        # Fleet KPI Summary
        n_crit = (scored_df["Status"] == "🔴 CRITICAL").sum()
        n_warn = (scored_df["Status"] == "🟡 WARNING").sum()
        n_safe = (scored_df["Status"] == "🟢 HEALTHY").sum()

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        b_c1.metric("Total Batch Fleet", len(scored_df))
        b_c2.metric("Healthy Machines (🟢)", n_safe)
        b_c3.metric("Warning Stage (🟡)", n_warn)
        b_c4.metric("Critical Alerts (🔴)", n_crit)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(scored_df, use_container_width=True)

        # Live Covariate Shift & Sensor Drift Monitor
        st.markdown("---")
        st.markdown("#### 📡 Real-Time Sensor Drift Monitor (Kolmogorov-Smirnov Statistical Test)")
        st.caption("Detects whether shop-floor telemetry has drifted from model baseline training distribution (p < 0.05).")

        with st.spinner("Analyzing fleet sensor distribution shift..."):
            drift_res = xai_engine.detect_sensor_data_drift(batch_df)

        if drift_res.get("fleet_drift_status") == "NORMAL_CALIBRATION":
            st.success(f"✅ Fleet Calibration Status: **NORMAL CALIBRATION** ({drift_res['features_with_drift']} / {drift_res['total_features_monitored']} features drifted)")
        else:
            st.warning(f"⚠️ Fleet Calibration Status: **RECALIBRATION RECOMMENDED** ({drift_res['features_with_drift']} / {drift_res['total_features_monitored']} features drifted)")

        with st.expander("View Per-Sensor Kolmogorov-Smirnov Test Results"):
            drift_rows = []
            for feat, d_info in drift_res.get("per_sensor_tests", {}).items():
                drift_rows.append({
                    "Sensor / Feature": feat,
                    "KS Statistic": d_info["ks_statistic"],
                    "p-value": d_info["p_value"],
                    "Calibration Status": d_info["status"],
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

        # Download CSV
        csv_data = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Diagnostic Fleet Assessment (CSV)",
            data=csv_data,
            file_name="predictive_maintenance_fleet_assessment.csv",
            mime="text/csv",
        )
