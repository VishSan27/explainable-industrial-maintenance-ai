"""
Feature engineering module for domain-specific thermal and mechanical metrics.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Standardized feature name mapping for flexibility
COLUMN_ALIASES = {
    "type": "type",
    "Type": "type",
    "product_type": "type",
    "air_temperature": "air_temperature",
    "Air temperature": "air_temperature",
    "Air temperature [K]": "air_temperature",
    "air_temp": "air_temperature",
    "process_temperature": "process_temperature",
    "Process temperature": "process_temperature",
    "Process temperature [K]": "process_temperature",
    "process_temp": "process_temperature",
    "rotational_speed": "rotational_speed",
    "Rotational speed": "rotational_speed",
    "Rotational speed [rpm]": "rotational_speed",
    "rpm": "rotational_speed",
    "torque": "torque",
    "Torque": "torque",
    "Torque [Nm]": "torque",
    "tool_wear": "tool_wear",
    "Tool wear": "tool_wear",
    "Tool wear [min]": "tool_wear",
    "wear": "tool_wear",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
TYPE_MAP_REV = {0: "L", 1: "M", 2: "H"}

# Physical Overstrain thresholds (min * Nm) per product quality variant
OVERSTRAIN_THRESHOLDS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
HDF_TEMP_DIFF_THRESHOLD = 8.6  # Kelvin
HDF_SPEED_THRESHOLD = 1380.0    # RPM
PWF_MIN_WATTS = 3500.0          # Watts
PWF_MAX_WATTS = 9000.0          # Watts
TWF_WARNING_MINUTES = 200.0     # Minutes of wear
TWF_CRITICAL_MINUTES = 240.0    # Tool life failure cutoff


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names in dataframe according to known aliases."""
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_dict[col] = COLUMN_ALIASES[col]
        else:
            col_clean = col.strip().lower().replace(" [k]", "").replace(" [rpm]", "").replace(" [nm]", "").replace(" [min]", "")
            col_clean = col_clean.replace(" ", "_")
            if col_clean in COLUMN_ALIASES:
                rename_dict[col] = COLUMN_ALIASES[col_clean]
    return df.rename(columns=rename_dict)


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that derives physics-based features
    for CNC milling / rotary machine failure diagnosis.
    """

    def __init__(self, include_raw_features: bool = True):
        self.include_raw_features = include_raw_features
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        df = standardize_columns(df)

        # Ensure required raw features exist
        required_raw = [
            "type",
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ]
        for col in required_raw:
            if col not in df.columns:
                raise ValueError(f"Missing required raw feature: '{col}'. Present: {df.columns.tolist()}")

        out = pd.DataFrame(index=df.index)

        # Encode product variant
        if df["type"].dtype == object or isinstance(df["type"].iloc[0], str):
            out["type_encoded"] = df["type"].map(TYPE_MAP).fillna(0).astype(int)
        else:
            out["type_encoded"] = df["type"].astype(int)

        air_temp = df["air_temperature"].astype(float)
        proc_temp = df["process_temperature"].astype(float)
        rpm = df["rotational_speed"].astype(float)
        torque = df["torque"].astype(float)
        wear = df["tool_wear"].astype(float)

        if self.include_raw_features:
            out["air_temperature"] = air_temp
            out["process_temperature"] = proc_temp
            out["rotational_speed"] = rpm
            out["torque"] = torque
            out["tool_wear"] = wear

        # Thermal features
        out["temp_diff"] = proc_temp - air_temp
        out["temp_ratio"] = proc_temp / np.maximum(air_temp, 1.0)

        # Mechanical power: P = omega * torque (Watts)
        out["mechanical_power"] = (2.0 * math.pi * rpm / 60.0) * torque

        # Overstrain product (tool wear * torque)
        out["overstrain_product"] = wear * torque

        # Power dissipation and load ratios
        out["power_dissipation_ratio"] = out["mechanical_power"] / np.maximum(out["temp_diff"], 0.1)
        out["torque_speed_ratio"] = torque / np.maximum(rpm, 1.0)

        # Tool wear progression
        out["wear_progression_ratio"] = wear / TWF_CRITICAL_MINUTES

        # Variant-specific strain limit (L: 11000, M: 12000, H: 13000)
        variant_limit = out["type_encoded"].map({0: 11000.0, 1: 12000.0, 2: 13000.0}).fillna(11000.0)
        out["overstrain_safety_margin"] = out["overstrain_product"] / variant_limit

        self.feature_names_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_


def compute_physics_risk_flags(row: Union[pd.Series, Dict]) -> Dict[str, Union[bool, float, str]]:
    """
    Evaluates exact domain physical boundary violation checks
    for immediate rule-based safety alerts in industrial operations.
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    # Standardize names
    row_std = row.copy()
    for k, v in row.items():
        if k in COLUMN_ALIASES:
            row_std[COLUMN_ALIASES[k]] = v

    type_val = str(row_std.get("type", "L"))
    air_t = float(row_std.get("air_temperature", 300.0))
    proc_t = float(row_std.get("process_temperature", 310.0))
    rpm = float(row_std.get("rotational_speed", 1500.0))
    torque = float(row_std.get("torque", 40.0))
    wear = float(row_std.get("tool_wear", 0.0))

    temp_diff = proc_t - air_t
    power = (2.0 * math.pi * rpm / 60.0) * torque
    overstrain = wear * torque
    overstrain_limit = OVERSTRAIN_THRESHOLDS.get(type_val, 11000.0)

    # Physical checks
    hdf_triggered = (temp_diff < HDF_TEMP_DIFF_THRESHOLD) and (rpm < HDF_SPEED_THRESHOLD)
    pwf_triggered = (power < PWF_MIN_WATTS) or (power > PWF_MAX_WATTS)
    osf_triggered = overstrain > overstrain_limit
    twf_triggered = wear >= TWF_WARNING_MINUTES

    return {
        "hdf_risk": bool(hdf_triggered),
        "pwf_risk": bool(pwf_triggered),
        "osf_risk": bool(osf_triggered),
        "twf_risk": bool(twf_triggered),
        "temp_diff_kelvin": round(temp_diff, 2),
        "mechanical_power_watts": round(power, 1),
        "overstrain_product": round(overstrain, 1),
        "overstrain_threshold": overstrain_limit,
        "wear_minutes": round(wear, 1),
        "critical_violations_count": int(sum([hdf_triggered, pwf_triggered, osf_triggered, twf_triggered])),
    }
