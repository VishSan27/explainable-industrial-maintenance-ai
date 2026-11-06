"""
Training and benchmarking pipeline for predictive maintenance.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import prepare_predictive_maintenance_data, get_curated_case_studies
from src.models import PredictiveMaintenancePipeline, MODEL_DIR
from src.explainability import XAIEngine


def run_training_pipeline():
    print("Loading AI4I 2020 dataset...")
    data_bundle = prepare_predictive_maintenance_data()
    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_train_bin = data_bundle["y_train_binary"]
    y_test_bin = data_bundle["y_test_binary"]
    y_train_multi = data_bundle["y_train_multiclass"]
    y_test_multi = data_bundle["y_test_multiclass"]
    y_train_rul = data_bundle["y_train_rul"]
    y_test_rul = data_bundle["y_test_rul"]

    print(f"Train samples: {len(X_train)} (failures: {y_train_bin.sum()})")
    print(f"Test samples:  {len(X_test)} (failures: {y_test_bin.sum()})")

    print("\nTraining binary, multi-class, and RUL models...")
    pipeline = PredictiveMaintenancePipeline(binary_model_type="xgboost", decision_threshold=0.35)
    benchmarks = pipeline.fit(
        X_train=X_train,
        y_train_binary=y_train_bin,
        y_train_multiclass=y_train_multi,
        y_train_rul=y_train_rul,
        X_val=X_test,
        y_val_binary=y_test_bin,
        y_val_multiclass=y_test_multi,
        y_val_rul=y_test_rul,
    )

    print("\n" + "-" * 75)
    print(f"{'Model':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Recall':<8} | {'Precision':<10} | {'F1':<8}")
    print("-" * 75)
    for model_name, metrics in benchmarks.items():
        print(
            f"{model_name:<22} | "
            f"{metrics['roc_auc']:<8.4f} | "
            f"{metrics['pr_auc']:<8.4f} | "
            f"{metrics['recall']:<8.4f} | "
            f"{metrics['precision']:<10.4f} | "
            f"{metrics['f1']:<8.4f}"
        )
    print("-" * 75)

    print("\nRUL Regression Metrics:")
    print(f"  R2:   {pipeline.rul_metrics.get('r2', 'N/A')}")
    print(f"  MAE:  {pipeline.rul_metrics.get('mae', 'N/A')} min")
    print(f"  RMSE: {pipeline.rul_metrics.get('rmse', 'N/A')} min")
    print(f"Conformal bounds (95%): binary +/-{pipeline.conformal_margin_binary:.4f}, RUL +/-{pipeline.conformal_margin_rul:.1f} min")

    print("\nSaving models and artifacts to", MODEL_DIR)
    pipeline.save(MODEL_DIR)
    bg_sample = X_train.sample(250, random_state=42)
    joblib.dump(bg_sample, os.path.join(MODEL_DIR, "background_sample.joblib"))

    # Verify explainability module
    print("\nInitializing XAI engine...")
    xai = XAIEngine(pipeline=pipeline, background_data=X_train.sample(250, random_state=42))

    case_studies = get_curated_case_studies()
    test_case = case_studies["Heat Dissipation Failure Risk (CNC-04)"]

    pred = pipeline.predict_risk(test_case)
    prob = pred["failure_probability"][0]
    print(f"Validation case test: prob = {prob:.4f}, health = {pred['health_index'][0]:.1f}%, RUL = {pred['estimated_rul_minutes'][0]} min")

    shap_exp = xai.explain_instance_shap(test_case)
    print("Top SHAP drivers:")
    for driver in shap_exp["top_risk_drivers"][:3]:
        print(f"  {driver['feature']}: SHAP={driver['shap_value']:.4f} (val: {driver['feature_value']})")

    prescr = xai.prescribe_counterfactual_actions(test_case)
    print(f"Prescriptive recommendation: risk {prescr['current_failure_prob']*100:.1f}% -> {prescr['optimized_failure_prob']*100:.1f}%")

    fidelity = xai.compute_xai_fidelity_and_agreement(test_case, top_k=3)
    print(f"XAI agreement: {fidelity['agreement_pct']}% | Erasure risk drop: {fidelity['causal_risk_drop']*100:.1f}%")

    # Save sample batch for SCADA simulation
    sample_batch = X_test.head(50).copy()
    sample_batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_telemetry.csv")
    sample_batch.to_csv(sample_batch_path, index=False)
    print(f"Saved sample batch to {sample_batch_path}")
    print("\nPipeline training complete.")


if __name__ == "__main__":
    run_training_pipeline()
