# Explainable AI-Based Predictive Maintenance

Predictive maintenance system for CNC milling machines and rotary equipment using machine telemetry (AI4I 2020 dataset). Combines physics-guided feature engineering with gradient-boosted decision trees, continuous Remaining Useful Life (RUL) regression, conformal uncertainty bounds, and local/global explainability via SHAP and LIME.

---

## Architecture Overview

The system addresses three operational tasks from machine sensor telemetry:

1. **Early Failure Detection (Binary):** Flags whether a machine is at imminent risk of breakdown using a cost-sensitive XGBoost classifier calibrated to prioritize recall.
2. **Failure Mode Diagnosis (Multi-Class):** Categorizes active degradation into one of five physical failure mechanisms (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, or Random Failure) using a Random Forest classifier.
3. **Remaining Useful Life (RUL) Prognosis:** Predicts remaining operational cutting life in minutes using a Random Forest regressor with 95% split-conformal confidence intervals.
4. **Model Explainability & Prescriptive Guidance:** Computes local feature attributions (SHAP waterfall & LIME surrogates), validates explanation faithfulness via feature erasure, monitors live sensor drift using Kolmogorov-Smirnov tests, and suggests minimal counterfactual parameter adjustments to avert failure.

```
Sensor Telemetry (Torque, Speed, Temperatures, Tool Wear)
               │
               ▼
  Physics Feature Engineering (ΔT, Power = ω·τ, Overstrain Product)
               │
       ┌───────┴───────┬──────────────────────┐
       ▼               ▼                      ▼
Binary Detection   Mode Diagnosis       RUL Regression
   (XGBoost)       (Random Forest)     (Random Forest)
       │               │                      │
       └───────┬───────┴──────────────────────┘
               ▼
   Conformal Uncertainty (95% Coverage)
               │
               ▼
   Explainability (SHAP / LIME) + Drift Monitor (KS-Test)
               │
               ▼
   Prescriptive Counterfactual Engine -> Web Dashboard / REST API
```

---

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository). It reflects real milling operations across 10,000 records with a realistic 3.39% failure class imbalance.

Key sensors:
* `Type`: Quality variant (`L`, `M`, `H`)
* `Air temperature [K]`: Ambient room temperature
* `Process temperature [K]`: Cutting zone friction temperature
* `Rotational speed [rpm]`: Spindle velocity
* `Torque [Nm]`: Cutting resistance torque
* `Tool wear [min]`: Cumulative cutting minutes on current insert

---

## Benchmark Results (2,000 Holdout Test Samples)

### Binary Failure Detection

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Selected)** | **0.9801** | **0.8869** | **85.29%** | 78.38% | **0.8169** | **98.70%** |
| Random Forest | 0.9716 | 0.8933 | 83.82% | 89.06% | 0.8636 | 99.10% |
| Gradient Boosting | 0.9773 | 0.8381 | 82.35% | 94.92% | 0.8819 | 99.25% |
| Logistic Regression | 0.9400 | 0.4726 | 92.65% | 13.29% | 0.2325 | 80.60% |

### Remaining Useful Life (RUL) Regression

* **$R^2$ Score:** 0.9696
* **Mean Absolute Error (MAE):** 2.32 minutes
* **Root Mean Squared Error (RMSE):** 11.63 minutes

### Uncertainty Quantification (Conformal Prediction)

* **Binary Failure Probability Bound (95% coverage):** $\pm 0.1070$
* **RUL Bound (95% coverage):** $\pm 9.2$ minutes

---

## Project Structure

```
.
├── app.py                     # Streamlit operations dashboard
├── train.py                   # Model training and artifact serialization
├── evaluate.py                # Model evaluation and plot generation
├── requirements.txt           # Python dependencies
├── data/
│   ├── ai4i2020.csv           # Telemetry dataset
│   └── sample_telemetry.csv   # Sample batch for SCADA simulation
├── models/
│   └── saved_models/          # Serialized models, scalers, and conformal quantiles
├── reports/
│   └── figures/               # Generated evaluation plots (ROC, PR, confusion matrix, SHAP)
├── src/
│   ├── api.py                 # FastAPI REST service
│   ├── data_loader.py         # Dataset loading and streaming generator
│   ├── feature_engineering.py # Physics-guided domain feature transformations
│   ├── models.py              # Multi-task training, RUL regression, and conformal intervals
│   └── explainability.py      # SHAP, LIME, counterfactuals, and drift monitoring
└── tests/
    └── test_pipeline.py       # Automated unit and integration tests
```

---

## Setup and Installation

### 1. Prerequisites
* Python 3.10+
* Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
Trains the feature transformer, binary detector, multi-class classifier, and RUL regressor, and computes conformal calibration residuals:
```bash
python train.py
```

### 4. Evaluate Models
Computes performance metrics and saves evaluation plots to `reports/figures/`:
```bash
python evaluate.py
```

### 5. Launch the Dashboard
Starts the Streamlit application for real-time monitoring and what-if simulation:
```bash
streamlit run app.py
```

### 6. Run the REST API
Starts the FastAPI service:
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive API docs available at `http://localhost:8000/docs`.

### 7. Run Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## References

* Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I).
* Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. ACM SIGKDD.
* Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
