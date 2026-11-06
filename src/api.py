"""
FastAPI service for telemetry ingestion and inference.
"""

import os
import sys
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure root directory on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PredictiveMaintenancePipeline, MODEL_DIR
from src.explainability import XAIEngine
from src.data_loader import prepare_predictive_maintenance_data, FAILURE_MODE_LABELS
from src.feature_engineering import compute_physics_risk_flags

# Global pipeline and explainability engine instances
pipeline: Optional[PredictiveMaintenancePipeline] = None
xai_engine: Optional[XAIEngine] = None


def load_artifacts():
    global pipeline, xai_engine
    if pipeline is not None and xai_engine is not None:
        return

    try:
        pipeline = PredictiveMaintenancePipeline.load(binary_model_type="xgboost")
        bg_sample_path = os.path.join(MODEL_DIR, "background_sample.joblib")
        if os.path.exists(bg_sample_path):
            bg_sample = joblib.load(bg_sample_path)
        else:
            data_bundle = prepare_predictive_maintenance_data()
            bg_sample = data_bundle["X_train"].sample(200, random_state=42)
            joblib.dump(bg_sample, bg_sample_path)

        xai_engine = XAIEngine(pipeline=pipeline, background_data=bg_sample)
        print("API: Pipeline and XAI Engine loaded.")
    except Exception as e:
        print(f"Warning on startup load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    description="REST API for machine failure early detection, diagnosis, and prescriptive recommendations.",
    lifespan=lifespan,
)


class TelemetryInput(BaseModel):
    machine_id: Optional[str] = Field(default="CNC-MILL-01", description="Identifier of the industrial machine")
    type: str = Field(default="M", description="Product variant quality: 'L', 'M', or 'H'")
    air_temperature: float = Field(default=298.5, description="Ambient air temperature in Kelvin [K]")
    process_temperature: float = Field(default=309.2, description="Process temperature in Kelvin [K]")
    rotational_speed: int = Field(default=1540, description="Spindle rotational speed [RPM]")
    torque: float = Field(default=40.0, description="Spindle resistance torque in Newton-meters [Nm]")
    tool_wear: int = Field(default=25, description="Cumulative cutting tool wear in minutes [min]")


class TelemetryBatchInput(BaseModel):
    telemetry_records: List[TelemetryInput]


@app.get("/")
def root():
    return {
        "service": "Explainable AI Predictive Maintenance API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "model_loaded": pipeline is not None,
        "xai_engine_ready": xai_engine is not None,
    }


@app.post("/predict")
def predict_telemetry(telemetry: TelemetryInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")

    raw_dict = telemetry.model_dump()
    pred_res = pipeline.predict_risk(raw_dict)
    prob = float(pred_res["failure_probability"][0])
    mode_idx = int(pred_res["mode_predicted"][0])
    health_score = float(pred_res["health_index"][0])
    rul_mins = float(pred_res["estimated_rul_minutes"][0])

    physics = compute_physics_risk_flags(raw_dict)

    if prob >= 0.70:
        status = "CRITICAL_ALERT"
    elif prob >= 0.35:
        status = "WARNING_ELEVATED"
    else:
        status = "NORMAL"

    return {
        "machine_id": telemetry.machine_id,
        "status": status,
        "failure_probability": round(prob, 4),
        "prob_conformal_95_ci": [
            round(float(pred_res.get("prob_conformal_lower", [prob])[0]), 4),
            round(float(pred_res.get("prob_conformal_upper", [prob])[0]), 4),
        ],
        "health_index_pct": round(health_score, 1),
        "estimated_rul_minutes": round(rul_mins, 1),
        "rul_conformal_95_ci": [
            round(float(pred_res.get("rul_conformal_lower", [rul_mins])[0]), 1),
            round(float(pred_res.get("rul_conformal_upper", [rul_mins])[0]), 1),
        ],
        "predicted_failure_mode": FAILURE_MODE_LABELS.get(mode_idx, "Unknown"),
        "failure_mode_id": mode_idx,
        "physics_checks": physics,
    }


@app.post("/explain")
def explain_telemetry(telemetry: TelemetryInput):
    if pipeline is None or xai_engine is None:
        raise HTTPException(status_code=503, detail="Explainability engine not ready.")

    raw_dict = telemetry.model_dump()
    pred_res = pipeline.predict_risk(raw_dict)
    shap_exp = xai_engine.explain_instance_shap(raw_dict)
    lime_exp = xai_engine.explain_instance_lime(raw_dict, num_features=5)
    brief = xai_engine.generate_natural_language_brief(raw_dict, pred_res, shap_exp)

    return {
        "machine_id": telemetry.machine_id,
        "diagnostic_headline": brief["headline"],
        "severity": brief["severity"],
        "detailed_narrative": brief["detailed_narrative"],
        "shap_analysis": {
            "base_value": round(shap_exp["base_value"], 4),
            "top_risk_drivers": shap_exp["top_risk_drivers"],
            "top_safety_drivers": shap_exp["top_safety_drivers"],
        },
        "lime_analysis": lime_exp,
    }


@app.post("/prescribe")
def prescribe_actions(telemetry: TelemetryInput, target_risk: float = 0.15):
    if xai_engine is None:
        raise HTTPException(status_code=503, detail="XAI engine not available.")

    raw_dict = telemetry.model_dump()
    prescription = xai_engine.prescribe_counterfactual_actions(raw_dict, target_failure_probability=target_risk)
    return {
        "machine_id": telemetry.machine_id,
        "prescription": prescription,
    }


@app.post("/batch-predict")
def batch_predict(batch: TelemetryBatchInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    results = []
    total_critical = 0
    total_warning = 0
    total_healthy = 0

    for rec in batch.telemetry_records:
        raw_dict = rec.model_dump()
        pred_res = pipeline.predict_risk(raw_dict)
        prob = float(pred_res["failure_probability"][0])
        mode_idx = int(pred_res["mode_predicted"][0])

        if prob >= 0.70:
            status = "CRITICAL"
            total_critical += 1
        elif prob >= 0.35:
            status = "WARNING"
            total_warning += 1
        else:
            status = "HEALTHY"
            total_healthy += 1

        results.append({
            "machine_id": rec.machine_id,
            "status": status,
            "failure_probability": round(prob, 4),
            "predicted_mode": FAILURE_MODE_LABELS.get(mode_idx, "Normal"),
            "health_index": round(float(pred_res["health_index"][0]), 1),
            "rul_minutes": round(float(pred_res["estimated_rul_minutes"][0]), 1),
        })

    return {
        "fleet_summary": {
            "total_machines": len(batch.telemetry_records),
            "healthy": total_healthy,
            "warning": total_warning,
            "critical": total_critical,
        },
        "records": results,
    }
