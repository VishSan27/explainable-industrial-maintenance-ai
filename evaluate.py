"""
Evaluation and visualization script for predictive maintenance pipeline.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import prepare_predictive_maintenance_data, FAILURE_MODE_LABELS, FAILURE_MODE_SHORT
from src.models import PredictiveMaintenancePipeline

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def evaluate_system():
    print("Evaluating pipeline on test split...")

    data_bundle = prepare_predictive_maintenance_data()
    X_test = data_bundle["X_test"]
    y_test_bin = data_bundle["y_test_binary"]
    y_test_multi = data_bundle["y_test_multiclass"]
    y_test_rul = data_bundle["y_test_rul"]

    pipeline = PredictiveMaintenancePipeline.load(binary_model_type="xgboost")
    X_test_feat = pipeline.feature_engineer.transform(X_test)

    # Inference
    binary_probs = pipeline.binary_model.predict_proba(X_test_feat)[:, 1]
    binary_preds = (binary_probs >= pipeline.decision_threshold).astype(int)

    multi_preds = pipeline.multiclass_model.predict(X_test_feat)

    # RUL regression
    rul_preds = pipeline.rul_model.predict(X_test_feat)
    r2_rul = r2_score(y_test_rul, rul_preds)
    mae_rul = mean_absolute_error(y_test_rul, rul_preds)
    rmse_rul = np.sqrt(mean_squared_error(y_test_rul, rul_preds))

    # Binary metrics
    acc = accuracy_score(y_test_bin, binary_preds)
    prec = precision_score(y_test_bin, binary_preds, zero_division=0)
    rec = recall_score(y_test_bin, binary_preds, zero_division=0)
    f1 = f1_score(y_test_bin, binary_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, binary_probs)
    pr_auc = average_precision_score(y_test_bin, binary_probs)

    print(f"\nBinary Classification (Threshold = {pipeline.decision_threshold:.2f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    print(f"\nRUL Regression:")
    print(f"  R2:   {r2_rul:.4f}")
    print(f"  MAE:  {mae_rul:.2f} min")
    print(f"  RMSE: {rmse_rul:.2f} min")

    # Plot ROC and PR curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test_bin, binary_probs)
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test_bin, binary_probs)
    plt.plot(rec_vals, prec_vals, color="#10B981", lw=2, label=f"XGBoost (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=y_test_bin.mean(), color="#94A3B8", linestyle="--", label=f"Baseline ({y_test_bin.mean():.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")

    plt.tight_layout()
    roc_pr_path = os.path.join(FIG_DIR, "roc_and_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_pr_path}")

    # Plot Confusion Matrices
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    cm_bin = confusion_matrix(y_test_bin, binary_preds)
    sns.heatmap(
        cm_bin,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Failure (1)"],
        yticklabels=["Normal (0)", "Failure (1)"],
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Binary Confusion Matrix")

    plt.subplot(1, 2, 2)
    classes_present = sorted(list(set(y_test_multi) | set(multi_preds)))
    class_names = [FAILURE_MODE_SHORT[c] for c in classes_present]
    cm_multi = confusion_matrix(y_test_multi, multi_preds, labels=classes_present)
    sns.heatmap(
        cm_multi,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted Mode")
    plt.ylabel("True Mode")
    plt.title("Multi-Class Confusion Matrix")

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # SHAP Summary Plot
    explainer = shap.TreeExplainer(pipeline.binary_model)
    sample_feat = X_test_feat.sample(250, random_state=42)
    shap_vals = explainer(sample_feat)

    plt.figure(figsize=(10, 6))
    if hasattr(shap_vals, "values") and len(shap_vals.values.shape) == 3:
        shap.summary_plot(shap_vals.values[:, :, 1], sample_feat, show=False)
    else:
        shap.summary_plot(shap_vals, sample_feat, show=False)
    plt.title("SHAP Feature Summary", pad=15)
    plt.tight_layout()
    shap_path = os.path.join(FIG_DIR, "shap_summary_plot.png")
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"Saved: {shap_path}")

    # Feature Importance Plot
    plt.figure(figsize=(10, 5))
    importances = pipeline.binary_model.feature_importances_
    feat_series = pd.Series(importances, index=pipeline.feature_names).sort_values(ascending=True)
    colors = ["#3B82F6" if "temp" in f or "power" in f or "overstrain" in f or "wear" in f else "#94A3B8" for f in feat_series.index]
    feat_series.plot(kind="barh", color=colors, edgecolor="black", linewidth=0.5)
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Model Feature Importance")
    plt.tight_layout()
    feat_imp_path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_imp_path}")


if __name__ == "__main__":
    evaluate_system()
