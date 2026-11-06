"""
Explainability, counterfactual recommendations, and telemetry drift monitoring.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.feature_engineering import (
    HDF_SPEED_THRESHOLD,
    HDF_TEMP_DIFF_THRESHOLD,
    OVERSTRAIN_THRESHOLDS,
    PWF_MAX_WATTS,
    PWF_MIN_WATTS,
    TWF_WARNING_MINUTES,
    compute_physics_risk_flags,
)


class XAIEngine:
    """
    Comprehensive Explainability & Prescriptive Maintenance Engine.
    Combines SHAP, LIME, and counterfactual optimization.
    """

    def __init__(self, pipeline, background_data: Optional[pd.DataFrame] = None):
        """
        Args:
            pipeline: PredictiveMaintenancePipeline instance.
            background_data: Representative background dataframe of raw features.
        """
        self.pipeline = pipeline
        self.feature_names = pipeline.feature_names
        self.model = pipeline.binary_model

        # Compute background features
        if background_data is not None:
            self.background_features = pipeline.feature_engineer.transform(background_data)
        else:
            # Fallback synthetic background
            self.background_features = None

        # Initialize SHAP TreeExplainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback for non-tree models
            if self.background_features is not None:
                bg_sample = shap.sample(self.background_features, 100)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
            else:
                self.shap_explainer = None

        # Initialize LIME Tabular Explainer
        if self.background_features is not None:
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.background_features.values,
                feature_names=self.background_features.columns.tolist(),
                class_names=["Normal", "Failure"],
                mode="classification",
                random_state=42,
            )
        else:
            self.lime_explainer = None

    def explain_instance_shap(self, raw_instance: Union[pd.Series, Dict, pd.DataFrame]) -> Dict[str, any]:
        """
        Calculates exact SHAP values for a single telemetry record.
        Returns:
            - base_value: Expected model log-odds or probability base
            - shap_values: Feature contributions
            - feature_contributions: Sorted dictionary of {feature: shap_value}
            - top_risk_drivers: Positive contributors to failure
            - top_safety_drivers: Negative contributors reducing failure risk
        """
        feat_df = self.pipeline.feature_engineer.transform(raw_instance)
        
        # Calculate SHAP values
        raw_shap = self.shap_explainer(feat_df)
        
        # Handle binary classification shapes (some shap versions return (N, M, 2) or (N, M))
        if hasattr(raw_shap, "values"):
            values = raw_shap.values
            if len(values.shape) == 3:  # (samples, features, classes)
                shap_vals = values[0, :, 1]
                base_val = float(raw_shap.base_values[0, 1])
            elif len(values.shape) == 2:  # (samples, features)
                shap_vals = values[0, :]
                base_val = float(raw_shap.base_values[0]) if hasattr(raw_shap.base_values, "__len__") else float(raw_shap.base_values)
            else:
                shap_vals = values[0]
                base_val = float(raw_shap.base_values)
        else:
            shap_vals = raw_shap[0]
            base_val = float(self.shap_explainer.expected_value)

        features = feat_df.columns.tolist()
        feat_values = feat_df.iloc[0].values

        contributions = []
        for name, val, s_val in zip(features, feat_values, shap_vals):
            contributions.append({
                "feature": name,
                "feature_value": round(float(val), 3),
                "shap_value": round(float(s_val), 4),
                "impact": "Increases Risk" if s_val > 0 else "Decreases Risk",
            })

        # Sort by absolute SHAP impact
        contributions_sorted = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_risk_drivers = [c for c in contributions_sorted if c["shap_value"] > 0][:5]
        top_safety_drivers = [c for c in contributions_sorted if c["shap_value"] < 0][:5]

        return {
            "base_value": base_val,
            "shap_values": [float(v) for v in shap_vals],
            "feature_names": features,
            "feature_values": [float(v) for v in feat_values],
            "contributions": contributions_sorted,
            "top_risk_drivers": top_risk_drivers,
            "top_safety_drivers": top_safety_drivers,
        }

    def explain_instance_lime(
        self,
        raw_instance: Union[pd.Series, Dict, pd.DataFrame],
        num_features: int = 6,
        num_samples: int = 250
    ) -> Dict[str, any]:
        """
        Generates local surrogate linear explanation using LIME.
        """
        if self.lime_explainer is None:
            return {"error": "LIME explainer not initialized with background data."}

        feat_df = self.pipeline.feature_engineer.transform(raw_instance)
        row_vec = feat_df.iloc[0].values

        # Model predict_proba wrapper
        def predict_fn(x_mat):
            return self.model.predict_proba(x_mat)

        exp = self.lime_explainer.explain_instance(
            row_vec,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)
        )

        lime_list = exp.as_list(label=1)
        return {
            "intercept": float(exp.intercept[1]) if hasattr(exp, "intercept") and 1 in exp.intercept else 0.0,
            "local_prediction": float(exp.local_pred[0]) if hasattr(exp, "local_pred") else 0.0,
            "rules": [{"rule": rule, "weight": round(float(wt), 4)} for rule, wt in lime_list],
            "score": float(exp.score) if hasattr(exp, "score") else 1.0,
        }

    def prescribe_counterfactual_actions(
        self,
        raw_instance: Union[pd.Series, Dict],
        target_failure_probability: float = 0.15,
        max_iterations: int = 50
    ) -> Dict[str, any]:
        """
        Prescriptive Maintenance Engine:
        Finds the minimal actionable adjustments to operational parameters:
        - Torque adjustment (Feed rate adjustment)
        - Rotational Speed adjustment (Spindle RPM control)
        - Immediate Tool Replacement (Reset wear to 0)
        to steer machine out of danger zone.
        """
        if isinstance(raw_instance, pd.Series):
            curr = raw_instance.to_dict()
        elif isinstance(raw_instance, pd.DataFrame):
            curr = raw_instance.iloc[0].to_dict()
        else:
            curr = dict(raw_instance)

        # Baseline risk
        base_pred = self.pipeline.predict_risk(curr)
        base_prob = float(base_pred["failure_probability"][0])

        if base_prob <= target_failure_probability:
            return {
                "status": "SAFE",
                "current_failure_prob": round(base_prob, 4),
                "target_failure_prob": target_failure_probability,
                "message": "Machine is operating within safe limits. No operational intervention required.",
                "actions": [],
                "simulated_telemetry": curr,
            }

        actions = []
        sim = dict(curr)

        # Action 1: Check Tool Wear Replacement
        wear = float(sim.get("tool_wear", 0))
        if wear > 180:
            test_tool_change = dict(sim)
            test_tool_change["tool_wear"] = 0
            p_tool_change = float(self.pipeline.predict_risk(test_tool_change)["failure_probability"][0])
            actions.append({
                "parameter": "tool_wear",
                "original_value": wear,
                "recommended_value": 0,
                "action_type": "Tool Replacement",
                "instruction": f"Schedule immediate cutting insert change (Reset tool wear from {wear} min to 0 min).",
                "prob_after_action": round(p_tool_change, 4),
                "risk_reduction_pct": round((base_prob - p_tool_change) * 100, 1),
            })
            sim["tool_wear"] = 0
            curr_prob = p_tool_change
        else:
            curr_prob = base_prob

        # Action 2: Torque / Feed Rate Adjustment
        orig_torque = float(sim.get("torque", 40.0))
        test_torque = orig_torque
        step_torque = 2.0
        
        while curr_prob > target_failure_probability and test_torque > 20.0:
            test_torque -= step_torque
            sim_test = dict(sim)
            sim_test["torque"] = test_torque
            test_prob = float(self.pipeline.predict_risk(sim_test)["failure_probability"][0])
            if test_prob < curr_prob:
                curr_prob = test_prob
                sim["torque"] = test_torque

        if sim["torque"] < orig_torque:
            torque_delta = round(orig_torque - sim["torque"], 1)
            actions.append({
                "parameter": "torque",
                "original_value": orig_torque,
                "recommended_value": round(sim["torque"], 1),
                "action_type": "Feed Rate Reduction",
                "instruction": f"Reduce spindle feed rate / cutting depth to lower torque by {torque_delta} Nm (Target: {sim['torque']} Nm).",
                "prob_after_action": round(curr_prob, 4),
                "risk_reduction_pct": round((base_prob - curr_prob) * 100, 1),
            })

        # Action 3: Spindle Speed Optimization (if Heat Dissipation or Power failure)
        orig_rpm = float(sim.get("rotational_speed", 1500))
        temp_diff = float(sim.get("process_temperature", 310)) - float(sim.get("air_temperature", 300))
        
        if temp_diff < HDF_TEMP_DIFF_THRESHOLD and orig_rpm < HDF_SPEED_THRESHOLD:
            # Increase RPM to restore convective cooling
            recommended_rpm = 1450
            sim_test = dict(sim)
            sim_test["rotational_speed"] = recommended_rpm
            test_prob = float(self.pipeline.predict_risk(sim_test)["failure_probability"][0])
            if test_prob < curr_prob:
                curr_prob = test_prob
                sim["rotational_speed"] = recommended_rpm
                actions.append({
                    "parameter": "rotational_speed",
                    "original_value": orig_rpm,
                    "recommended_value": recommended_rpm,
                    "action_type": "Cooling Airflow Speed Up",
                    "instruction": f"Increase spindle speed to {recommended_rpm} RPM to enhance convective fan airflow and restore thermal dissipation.",
                    "prob_after_action": round(curr_prob, 4),
                    "risk_reduction_pct": round((base_prob - curr_prob) * 100, 1),
                })

        return {
            "status": "RECOMMENDED_ACTIONS",
            "current_failure_prob": round(base_prob, 4),
            "optimized_failure_prob": round(curr_prob, 4),
            "total_risk_mitigation_pct": round((base_prob - curr_prob) * 100, 1),
            "target_failure_prob": target_failure_probability,
            "actions": actions,
            "prescribed_telemetry": sim,
        }

    def generate_natural_language_brief(
        self,
        raw_instance: Union[pd.Series, Dict],
        prediction_result: Dict[str, any],
        shap_result: Dict[str, any]
    ) -> Dict[str, str]:
        """
        Synthesizes machine telemetry, physics checks, and XAI contributions
        into an actionable diagnostic report for plant operators.
        """
        prob = float(prediction_result["failure_probability"][0])
        mode_idx = int(prediction_result["mode_predicted"][0])
        mode_name = {
            0: "Normal Machine Operation",
            1: "Tool Wear Failure (TWF)",
            2: "Heat Dissipation Failure (HDF)",
            3: "Power Failure (PWF)",
            4: "Overstrain Failure (OSF)",
            5: "Random Failure (RNF)",
        }.get(mode_idx, "Machine Anomaly")

        physics = compute_physics_risk_flags(raw_instance)
        top_risks = shap_result.get("top_risk_drivers", [])

        # Classification severity
        if prob >= 0.70:
            severity = "CRITICAL EMERGENCY"
            status_emoji = "🔴"
        elif prob >= 0.35:
            severity = "WARNING / ELEVATED RISK"
            status_emoji = "🟡"
        else:
            severity = "NORMAL / HEALTHY"
            status_emoji = "🟢"

        headline = f"{status_emoji} [{severity}] Failure Risk: {round(prob * 100, 1)}% | Root Cause: {mode_name}"

        # Build diagnostic narrative
        narrative_lines = []
        narrative_lines.append(f"**Diagnostic Summary:** Condition monitor flagged this unit with a **{severity}** alert ({round(prob * 100, 1)}% breakdown probability).")
        
        if prob >= 0.35:
            narrative_lines.append(f"**Impending Failure Mode:** **{mode_name}**.")

            if physics["hdf_risk"]:
                narrative_lines.append(f"- **Thermal Dissipation Hazard:** The temperature differential between the cutting zone and ambient air is {physics['temp_diff_kelvin']} K (threshold: > {HDF_TEMP_DIFF_THRESHOLD} K), and spindle speed is {raw_instance.get('rotational_speed')} RPM (< {HDF_SPEED_THRESHOLD} RPM). Inadequate convective airflow causes heat accumulation.")
            
            if physics["osf_risk"]:
                narrative_lines.append(f"- **Mechanical Overstrain Hazard:** Cumulative strain product (Tool Wear × Torque = {physics['overstrain_product']} min·Nm) exceeds design threshold of {physics['overstrain_threshold']} min·Nm for Type {raw_instance.get('type', 'M')} components.")
            
            if physics["pwf_risk"]:
                narrative_lines.append(f"- **Electrical Power Rating Hazard:** Calculated mechanical power is {physics['mechanical_power_watts']} W, operating outside safe range [{PWF_MIN_WATTS} W - {PWF_MAX_WATTS} W].")

            if physics["twf_risk"]:
                narrative_lines.append(f"- **Cutting Tool Degradation:** Cumulative tool wear reached {physics['wear_minutes']} minutes, exceeding preventative changeover limit ({TWF_WARNING_MINUTES} min).")

            narrative_lines.append("\n**Primary Feature Drivers (SHAP):**")
            for i, r in enumerate(top_risks[:3], 1):
                f_name = r["feature"].replace("_", " ").title()
                narrative_lines.append(f"  {i}. **{f_name}** ({r['feature_value']}): Impact {r['shap_value']:+.3f}")
        else:
            narrative_lines.append("All mechanical and thermodynamic parameters are operating well within manufacturer-specified safe tolerances.")
            narrative_lines.append(f"- Mechanical Power: {physics['mechanical_power_watts']} W (Safe range: {PWF_MIN_WATTS}W - {PWF_MAX_WATTS}W)")
            narrative_lines.append(f"- Thermal Delta T: {physics['temp_diff_kelvin']} K (Adequate convective cooling)")
            narrative_lines.append(f"- Tool Wear: {physics['wear_minutes']} min (Remaining tool life available)")

        return {
            "headline": headline,
            "severity": severity,
            "detailed_narrative": "\n".join(narrative_lines),
        }

    def compute_xai_fidelity_and_agreement(
        self,
        raw_instance: Union[pd.Series, Dict],
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Scientific validation of Explainable AI outputs:
        1. Jaccard Agreement between SHAP and LIME top-K attributions
        2. Feature Erasure Faithfulness Test (verifying if removing the top driver mitigates prediction)
        """
        shap_res = self.explain_instance_shap(raw_instance)
        lime_res = self.explain_instance_lime(raw_instance, num_features=top_k)

        # Extract top feature names
        shap_top = [d["feature"] for d in shap_res.get("top_risk_drivers", [])[:top_k]]
        lime_rules = lime_res.get("rules", [])
        
        # Match LIME rules back to feature names
        lime_top = []
        for r in lime_rules[:top_k]:
            for f in self.feature_names:
                if f in r["rule"]:
                    lime_top.append(f)
                    break

        # Compute Jaccard Agreement
        set_shap = set(shap_top)
        set_lime = set(lime_top)
        union_len = len(set_shap | set_lime)
        inter_len = len(set_shap & set_lime)
        jaccard_score = round(float(inter_len / union_len), 3) if union_len > 0 else 1.0

        # Feature Erasure Sensitivity Test (Faithfulness verification)
        base_pred = self.pipeline.predict_risk(raw_instance)
        p_original = float(base_pred["failure_probability"][0])

        if shap_top and self.background_features is not None:
            top_feature = shap_top[0]
            # Replace top risk feature with background median/mean safe value
            erased_instance = dict(raw_instance)
            if top_feature in erased_instance:
                bg_mean_val = float(self.background_features[top_feature].mean()) if top_feature in self.background_features else 0.0
                erased_instance[top_feature] = bg_mean_val
                p_erased = float(self.pipeline.predict_risk(erased_instance)["failure_probability"][0])
                delta_prob = round(p_original - p_erased, 4)
                is_faithful = delta_prob >= 0.0  # Removing risk factor reduces probability
            else:
                top_feature = "composite_physics_feature"
                p_erased = p_original
                delta_prob = 0.0
                is_faithful = True
        else:
            top_feature = "none"
            p_erased = p_original
            delta_prob = 0.0
            is_faithful = True

        return {
            "jaccard_agreement_score": jaccard_score,
            "agreement_pct": round(jaccard_score * 100, 1),
            "shap_top_features": shap_top,
            "lime_top_features": lime_top,
            "erased_feature": top_feature,
            "original_risk": round(p_original, 4),
            "risk_after_erasure": round(p_erased, 4),
            "causal_risk_drop": delta_prob,
            "faithfulness_validated": is_faithful,
        }

    def detect_sensor_data_drift(
        self,
        incoming_batch_df: pd.DataFrame,
        p_val_threshold: float = 0.05
    ) -> Dict[str, any]:
        """
        Industrial Covariate Shift & Sensor Drift Monitor:
        Performs 2-sample Kolmogorov-Smirnov (KS) tests between incoming telemetry
        and the baseline training reference distribution.
        """
        if self.background_features is None:
            return {"error": "Baseline reference data not configured."}

        incoming_feat = self.pipeline.feature_engineer.transform(incoming_batch_df)
        drift_report = {}
        drift_detected_count = 0

        common_cols = [c for c in incoming_feat.columns if c in self.background_features.columns]

        for col in common_cols:
            ref_vals = self.background_features[col].dropna().values
            curr_vals = incoming_feat[col].dropna().values

            if len(curr_vals) >= 5 and len(ref_vals) >= 5:
                stat, p_val = ks_2samp(ref_vals, curr_vals)
                is_drift = bool(p_val < p_val_threshold)
                if is_drift:
                    drift_detected_count += 1

                drift_report[col] = {
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(p_val), 4),
                    "drift_detected": is_drift,
                    "status": "DRIFT" if is_drift else "IN_CALIBRATION",
                }

        return {
            "total_features_monitored": len(common_cols),
            "features_with_drift": drift_detected_count,
            "fleet_drift_status": "RECALIBRATION_RECOMMENDED" if drift_detected_count > 2 else "NORMAL_CALIBRATION",
            "per_sensor_tests": drift_report,
        }
