"""
Pipeline definition for binary failure detection, multi-class diagnosis, and RUL regression.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature_engineering import PhysicsFeatureEngineer

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


class PredictiveMaintenancePipeline:
    """
    End-to-end industrial predictive maintenance pipeline combining:
    1. Physics feature engineering
    2. Binary failure detection (Champion: XGBoost)
    3. Multi-class failure mode diagnosis (Random Forest)
    4. Continuous Remaining Useful Life (RUL) regression (Random Forest Regressor)
    5. Conformal Prediction & Uncertainty Quantification (95% coverage bounds)
    """

    def __init__(
        self,
        binary_model_type: str = "xgboost",
        decision_threshold: float = 0.40,  # Optimized for early warning
        random_state: int = 42
    ):
        self.binary_model_type = binary_model_type
        self.decision_threshold = decision_threshold
        self.random_state = random_state

        self.feature_engineer = PhysicsFeatureEngineer(include_raw_features=True)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

        self.binary_model = None
        self.multiclass_model = None
        self.rul_model = None

        # Conformal prediction calibration parameters
        self.conformal_margin_binary: float = 0.05
        self.conformal_margin_rul: float = 6.0
        self.rul_metrics: Dict[str, float] = {}
        self.benchmark_results: Dict[str, Dict] = {}

    def _init_binary_model(self, model_type: str, scale_pos_weight: float = 1.0):
        if model_type == "xgboost":
            return XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
                use_label_encoder=False if hasattr(XGBClassifier, "use_label_encoder") else None,
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.random_state,
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_binary: pd.Series,
        y_train_multiclass: Optional[pd.Series] = None,
        y_train_rul: Optional[pd.Series] = None,
        benchmark_all: bool = True,
        X_val: Optional[pd.DataFrame] = None,
        y_val_binary: Optional[pd.Series] = None,
        y_val_multiclass: Optional[pd.Series] = None,
        y_val_rul: Optional[pd.Series] = None,
    ) -> Dict[str, any]:
        """
        Fits physics feature engineer, trains candidate binary models,
        evaluates metrics, and trains the multi-class root-cause classifier.
        """
        # Feature transformations
        X_train_feat = self.feature_engineer.fit_transform(X_train)
        self.feature_names = self.feature_engineer.get_feature_names_out()

        # Class imbalance weighting
        neg_count = (y_train_binary == 0).sum()
        pos_count = max((y_train_binary == 1).sum(), 1)
        scale_pos_weight = float(neg_count / pos_count)

        self.scaler.fit(X_train_feat)

        # Train and evaluate candidate binary models
        candidate_models = ["xgboost", "random_forest", "gradient_boosting", "logistic_regression"]
        self.benchmark_results = {}

        if X_val is not None and y_val_binary is not None:
            X_val_feat = self.feature_engineer.transform(X_val)
        else:
            X_val_feat = X_train_feat
            y_val_binary = y_train_binary

        for m_name in candidate_models:
            m = self._init_binary_model(m_name, scale_pos_weight=scale_pos_weight if m_name == "xgboost" else 1.0)
            if m_name == "logistic_regression":
                m.fit(self.scaler.transform(X_train_feat), y_train_binary)
                val_probs = m.predict_proba(self.scaler.transform(X_val_feat))[:, 1]
            else:
                m.fit(X_train_feat, y_train_binary)
                val_probs = m.predict_proba(X_val_feat)[:, 1]

            val_preds = (val_probs >= self.decision_threshold).astype(int)

            self.benchmark_results[m_name] = {
                "model": m,
                "accuracy": round(float(accuracy_score(y_val_binary, val_preds)), 4),
                "precision": round(float(precision_score(y_val_binary, val_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_val_binary, val_preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_val_binary, val_preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_val_binary, val_probs)), 4),
                "pr_auc": round(float(average_precision_score(y_val_binary, val_probs)), 4),
                "brier_score": round(float(brier_score_loss(y_val_binary, val_probs)), 4),
            }

        # Set champion model
        if self.binary_model_type in self.benchmark_results:
            self.binary_model = self.benchmark_results[self.binary_model_type]["model"]
        else:
            self.binary_model = self.benchmark_results["xgboost"]["model"]

        # Failure mode classifier
        if y_train_multiclass is not None:
            self.multiclass_model = RandomForestClassifier(
                n_estimators=180,
                max_depth=14,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.multiclass_model.fit(X_train_feat, y_train_multiclass)

        # RUL regressor
        if y_train_rul is not None:
            self.rul_model = RandomForestRegressor(
                n_estimators=160,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.rul_model.fit(X_train_feat, y_train_rul)

            # Evaluate RUL Regressor & Conformal Calibration Quantile
            if y_val_rul is not None and X_val is not None:
                val_rul_preds = self.rul_model.predict(X_val_feat)
                self.rul_metrics = {
                    "r2": round(float(r2_score(y_val_rul, val_rul_preds)), 4),
                    "mae": round(float(mean_absolute_error(y_val_rul, val_rul_preds)), 2),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_val_rul, val_rul_preds))), 2),
                }
                # Conformal residual margin at 95% coverage
                rul_residuals = np.abs(y_val_rul.values - val_rul_preds)
                self.conformal_margin_rul = round(float(np.percentile(rul_residuals, 95)), 2)

        # Compute Conformal Prediction Margin for Binary Classifier (95% coverage)
        if X_val is not None and y_val_binary is not None and self.binary_model is not None:
            val_probs_champ = self.binary_model.predict_proba(X_val_feat)[:, 1]
            prob_residuals = np.abs(y_val_binary.values - val_probs_champ)
            self.conformal_margin_binary = round(float(np.percentile(prob_residuals, 95)), 4)

        return self.benchmark_results

    def predict_risk(
        self,
        X: Union[pd.DataFrame, Dict, List]
    ) -> Dict[str, any]:
        """
        Calculates failure probability, decision, predicted failure mode,
        machine health index, ML-regressed RUL, and 95% conformal prediction intervals.
        """
        X_feat = self.feature_engineer.transform(X)

        if self.binary_model_type == "logistic_regression":
            probs = self.binary_model.predict_proba(self.scaler.transform(X_feat))[:, 1]
        else:
            probs = self.binary_model.predict_proba(X_feat)[:, 1]

        preds = (probs >= self.decision_threshold).astype(int)

        # Multi-class prediction
        if self.multiclass_model is not None:
            mode_preds = self.multiclass_model.predict(X_feat)
            mode_probs = self.multiclass_model.predict_proba(X_feat)
        else:
            mode_preds = np.zeros(len(probs), dtype=int)
            mode_probs = np.zeros((len(probs), 6))

        # Health degradation index (100% = pristine, 0% = failed)
        health_scores = np.clip(100.0 * (1.0 - probs), 0.0, 100.0)

        # Remaining Useful Life (RUL in minutes)
        if self.rul_model is not None:
            rul_mins = np.round(self.rul_model.predict(X_feat), 1)
        else:
            wear_val = X_feat["tool_wear"].values
            wear_remaining = np.maximum(0.0, 240.0 - wear_val)
            stress_factor = np.clip(1.0 + (probs * 2.5), 1.0, 4.0)
            rul_mins = np.round(wear_remaining / stress_factor, 1)

        # Conformal Prediction Intervals (95% statistical coverage)
        prob_lower = np.clip(np.round(probs - self.conformal_margin_binary, 4), 0.0, 1.0)
        prob_upper = np.clip(np.round(probs + self.conformal_margin_binary, 4), 0.0, 1.0)
        rul_lower = np.maximum(0.0, np.round(rul_mins - self.conformal_margin_rul, 1))
        rul_upper = np.round(rul_mins + self.conformal_margin_rul, 1)

        return {
            "failure_probability": probs,
            "failure_predicted": preds,
            "prob_conformal_lower": prob_lower,
            "prob_conformal_upper": prob_upper,
            "conformal_margin_binary": self.conformal_margin_binary,
            "health_index": health_scores,
            "estimated_rul_minutes": rul_mins,
            "rul_conformal_lower": rul_lower,
            "rul_conformal_upper": rul_upper,
            "conformal_margin_rul": self.conformal_margin_rul,
            "mode_predicted": mode_preds,
            "mode_probabilities": mode_probs,
            "features_df": X_feat,
        }

    def save(self, directory: Optional[str] = None):
        """Persists trained pipeline artifacts to disk."""
        save_dir = directory or MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.feature_engineer, os.path.join(save_dir, "feature_engineer.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        joblib.dump(self.binary_model, os.path.join(save_dir, f"binary_{self.binary_model_type}.joblib"))
        if self.multiclass_model is not None:
            joblib.dump(self.multiclass_model, os.path.join(save_dir, "multiclass_model.joblib"))
        if self.rul_model is not None:
            joblib.dump(self.rul_model, os.path.join(save_dir, "rul_model.joblib"))

        conformal_bundle = {
            "conformal_margin_binary": self.conformal_margin_binary,
            "conformal_margin_rul": self.conformal_margin_rul,
            "rul_metrics": self.rul_metrics,
        }
        joblib.dump(conformal_bundle, os.path.join(save_dir, "conformal_bundle.joblib"))

        # Save benchmark summary
        summary = {}
        for k, v in self.benchmark_results.items():
            summary[k] = {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]}
        joblib.dump(summary, os.path.join(save_dir, "benchmark_metrics.joblib"))

    @classmethod
    def load(cls, directory: Optional[str] = None, binary_model_type: str = "xgboost") -> "PredictiveMaintenancePipeline":
        """Loads trained pipeline from disk."""
        load_dir = directory or MODEL_DIR
        pipeline = cls(binary_model_type=binary_model_type)

        pipeline.feature_engineer = joblib.load(os.path.join(load_dir, "feature_engineer.joblib"))
        pipeline.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        pipeline.binary_model = joblib.load(os.path.join(load_dir, f"binary_{binary_model_type}.joblib"))

        multi_path = os.path.join(load_dir, "multiclass_model.joblib")
        if os.path.exists(multi_path):
            pipeline.multiclass_model = joblib.load(multi_path)

        rul_path = os.path.join(load_dir, "rul_model.joblib")
        if os.path.exists(rul_path):
            pipeline.rul_model = joblib.load(rul_path)

        conformal_path = os.path.join(load_dir, "conformal_bundle.joblib")
        if os.path.exists(conformal_path):
            conf = joblib.load(conformal_path)
            pipeline.conformal_margin_binary = conf.get("conformal_margin_binary", 0.05)
            pipeline.conformal_margin_rul = conf.get("conformal_margin_rul", 6.0)
            pipeline.rul_metrics = conf.get("rul_metrics", {})

        metrics_path = os.path.join(load_dir, "benchmark_metrics.joblib")
        if os.path.exists(metrics_path):
            pipeline.benchmark_results = joblib.load(metrics_path)

        pipeline.feature_names = pipeline.feature_engineer.get_feature_names_out()
        return pipeline
