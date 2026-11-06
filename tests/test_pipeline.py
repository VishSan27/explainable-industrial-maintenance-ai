"""
Unit and integration tests for predictive maintenance pipeline.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import PhysicsFeatureEngineer, compute_physics_risk_flags
from src.data_loader import prepare_predictive_maintenance_data, generate_simulated_stream_step, get_curated_case_studies
from src.models import PredictiveMaintenancePipeline
from src.explainability import XAIEngine
from src.api import app, load_artifacts


@pytest.fixture(scope="session")
def trained_pipeline():
    return PredictiveMaintenancePipeline.load()


@pytest.fixture(scope="session")
def xai_system(trained_pipeline):
    data_bundle = prepare_predictive_maintenance_data()
    bg = data_bundle["X_train"].sample(100, random_state=42)
    return XAIEngine(pipeline=trained_pipeline, background_data=bg)


def test_physics_feature_engineering():
    engineer = PhysicsFeatureEngineer(include_raw_features=True)
    sample_raw = pd.DataFrame([{
        "type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 50,
    }])

    feat_df = engineer.fit_transform(sample_raw)

    assert "temp_diff" in feat_df.columns
    assert "mechanical_power" in feat_df.columns
    assert "overstrain_product" in feat_df.columns

    # Verify mathematical values
    assert abs(feat_df["temp_diff"].iloc[0] - 10.0) < 1e-4
    expected_power = (2.0 * np.pi * 1500 / 60.0) * 40.0
    assert abs(feat_df["mechanical_power"].iloc[0] - expected_power) < 1e-2
    assert abs(feat_df["overstrain_product"].iloc[0] - 2000.0) < 1e-4


def test_physics_risk_flags():
    # Heat Dissipation failure condition: Delta T < 8.6 and RPM < 1380
    hdf_sample = {
        "type": "L",
        "air_temperature": 303.0,
        "process_temperature": 309.0,  # Delta T = 6.0 K
        "rotational_speed": 1300,      # RPM < 1380
        "torque": 45.0,
        "tool_wear": 100,
    }
    flags = compute_physics_risk_flags(hdf_sample)
    assert flags["hdf_risk"] is True
    assert flags["critical_violations_count"] >= 1


def test_telemetry_simulation_step():
    initial_state = {
        "type": "M",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 20,
    }
    next_state = generate_simulated_stream_step(initial_state, mode="tool_degradation", step=1)
    assert next_state["tool_wear"] > initial_state["tool_wear"]
    assert "torque" in next_state
    assert "rotational_speed" in next_state


def test_pipeline_prediction(trained_pipeline):
    cases = get_curated_case_studies()
    healthy = cases["Nominal Healthy Machine (CNC-01)"]
    fail_case = cases["Spindle Overstrain Failure (CNC-09)"]

    healthy_res = trained_pipeline.predict_risk(healthy)
    fail_res = trained_pipeline.predict_risk(fail_case)

    # Healthy machine should have low failure probability
    assert healthy_res["failure_probability"][0] < 0.35
    assert healthy_res["health_index"][0] > 65.0
    assert "prob_conformal_lower" in healthy_res
    assert "rul_conformal_lower" in healthy_res
    assert healthy_res["estimated_rul_minutes"][0] > 0.0

    # Overstrain failure machine should have high failure probability
    assert fail_res["failure_probability"][0] > 0.60
    assert fail_res["health_index"][0] < 40.0


def test_xai_shap_and_lime(xai_system):
    cases = get_curated_case_studies()
    test_case = cases["High-Power Motor Overload (CNC-14)"]

    # SHAP test
    shap_res = xai_system.explain_instance_shap(test_case)
    assert "top_risk_drivers" in shap_res
    assert len(shap_res["top_risk_drivers"]) > 0
    assert len(shap_res["shap_values"]) == len(shap_res["feature_names"])

    # LIME test
    lime_res = xai_system.explain_instance_lime(test_case, num_features=4)
    assert "rules" in lime_res
    assert len(lime_res["rules"]) > 0


def test_xai_fidelity_and_drift(xai_system):
    cases = get_curated_case_studies()
    case = cases["Heat Dissipation Failure Risk (CNC-04)"]
    fidelity = xai_system.compute_xai_fidelity_and_agreement(case, top_k=3)
    assert "jaccard_agreement_score" in fidelity
    assert fidelity["jaccard_agreement_score"] >= 0.0
    assert fidelity["faithfulness_validated"] is True

    sample_df = pd.DataFrame([case, cases["Nominal Healthy Machine (CNC-01)"]])
    drift = xai_system.detect_sensor_data_drift(sample_df)
    assert "fleet_drift_status" in drift
    assert "per_sensor_tests" in drift


def test_counterfactual_prescriptive_engine(xai_system):
    cases = get_curated_case_studies()
    critical_case = cases["Spindle Overstrain Failure (CNC-09)"]

    prescription = xai_system.prescribe_counterfactual_actions(critical_case, target_failure_probability=0.20)
    assert prescription["status"] in ["RECOMMENDED_ACTIONS", "SAFE"]
    assert prescription["optimized_failure_prob"] < prescription["current_failure_prob"]
    assert len(prescription["actions"]) > 0


def test_fastapi_rest_endpoints():
    load_artifacts()
    from src.api import health, predict_telemetry, explain_telemetry, prescribe_actions, TelemetryInput

    # Health
    h = health()
    assert h["status"] == "healthy"
    assert h["model_loaded"] is True
    assert h["xai_engine_ready"] is True

    # Predict
    payload = TelemetryInput(
        machine_id="TEST-CNC-99",
        type="L",
        air_temperature=298.5,
        process_temperature=309.2,
        rotational_speed=1540,
        torque=40.0,
        tool_wear=30
    )
    pred_res = predict_telemetry(payload)
    assert "failure_probability" in pred_res
    assert "health_index_pct" in pred_res
    assert "predicted_failure_mode" in pred_res
    assert pred_res["status"] in ["NORMAL", "WARNING_ELEVATED", "CRITICAL_ALERT"]

    # Explain
    exp_res = explain_telemetry(payload)
    assert "diagnostic_headline" in exp_res
    assert "shap_analysis" in exp_res
    assert "lime_analysis" in exp_res

    # Prescribe
    pres_res = prescribe_actions(payload)
    assert "prescription" in pres_res
    assert pres_res["prescription"]["status"] in ["SAFE", "RECOMMENDED_ACTIONS"]
