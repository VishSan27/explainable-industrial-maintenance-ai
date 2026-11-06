"""
Data loading, preprocessing, and telemetry stream simulation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ai4i2020.csv")

FAILURE_MODE_LABELS = {
    0: "Normal Operation",
    1: "Tool Wear Failure (TWF)",
    2: "Heat Dissipation Failure (HDF)",
    3: "Power Failure (PWF)",
    4: "Overstrain Failure (OSF)",
    5: "Random Failure (RNF)",
}

FAILURE_MODE_SHORT = {
    0: "Normal",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
    5: "RNF",
}


def load_raw_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads raw AI4I 2020 dataset from CSV."""
    path = filepath or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please run data download first.")
    
    df = pd.read_csv(path)
    return df


def prepare_predictive_maintenance_data(
    filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Cleans raw AI4I dataset and prepares:
    - X_raw: raw features
    - y_binary: Machine failure indicator (0 / 1)
    - y_multiclass: Consolidated failure mode category (0: Normal, 1: TWF, 2: HDF, 3: PWF, 4: OSF, 5: RNF)
    - Train / test stratified splits
    """
    df = load_raw_dataset(filepath)

    # Standardize column names
    rename_cols = {
        "Type": "type",
        "Air temperature": "air_temperature",
        "Air temperature [K]": "air_temperature",
        "Process temperature": "process_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed": "rotational_speed",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque": "torque",
        "Torque [Nm]": "torque",
        "Tool wear": "tool_wear",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    }
    df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

    # Drop non-predictive identifiers if present
    drop_candidates = ["UDI", "Product ID", "udi", "product_id"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

    # Extract target modes
    # If machine_failure is 1, determine primary failure mode priority: HDF > PWF > OSF > TWF > RNF
    def assign_failure_class(row):
        if row.get("hdf", 0) == 1:
            return 2
        elif row.get("pwf", 0) == 1:
            return 3
        elif row.get("osf", 0) == 1:
            return 4
        elif row.get("twf", 0) == 1:
            return 1
        elif row.get("rnf", 0) == 1:
            return 5
        elif row.get("machine_failure", 0) == 1:
            return 4  # Default unassigned failure to mechanical overstrain
        return 0  # Normal

    df["failure_mode"] = df.apply(assign_failure_class, axis=1)

    # Calculate Continuous Remaining Useful Life (RUL in minutes)
    # RUL is defined by remaining cutting tool capacity (max 240 min)
    # penalized by thermal-mechanical operating stress
    def calculate_ground_truth_rul(row):
        wear = float(row.get("tool_wear", 0.0))
        is_failed = int(row.get("machine_failure", 0))
        if is_failed == 1:
            return 0.0
        remaining_wear = max(0.0, 240.0 - wear)
        # Moderate operational degradation factor
        return round(remaining_wear, 1)

    df["rul_minutes"] = df.apply(calculate_ground_truth_rul, axis=1)

    feature_cols = ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    X = df[feature_cols].copy()
    y_binary = df["machine_failure"].copy()
    y_multiclass = df["failure_mode"].copy()
    y_rul = df["rul_minutes"].copy()

    # Stratified split using binary target
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, y_train_rul, y_test_rul = train_test_split(
        X, y_binary, y_multiclass, y_rul, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    return {
        "df": df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_binary": y_train_bin,
        "y_test_binary": y_test_bin,
        "y_train_multiclass": y_train_multi,
        "y_test_multiclass": y_test_multi,
        "y_train_rul": y_train_rul,
        "y_test_rul": y_test_rul,
    }


def generate_simulated_stream_step(
    current_state: Dict[str, float],
    mode: str = "normal",
    step: int = 0
) -> Dict[str, any]:
    """
    Generates the next simulated industrial telemetry reading.
    Supported modes:
    - 'normal': steady state with ambient fluctuations
    - 'heat_buildup': process temperature rises, cooling narrows Delta T
    - 'power_surge': rapid torque or RPM surge exceeding motor ratings
    - 'tool_degradation': progressive tool wear accumulation reaching critical wear
    - 'overstrain': high torque cutting operation leading to spindle stress
    """
    np.random.seed(int(current_state.get("seed", 42)) + step)

    type_val = current_state.get("type", "M")
    air_t = current_state.get("air_temperature", 300.0)
    proc_t = current_state.get("process_temperature", 310.0)
    rpm = current_state.get("rotational_speed", 1530.0)
    torque = current_state.get("torque", 40.0)
    wear = current_state.get("tool_wear", 10.0)

    # Ambient drift
    air_t += np.random.normal(0, 0.05)
    
    if mode == "normal":
        proc_t = air_t + 10.0 + np.random.normal(0, 0.2)
        rpm = np.clip(rpm + np.random.normal(0, 15), 1400, 1750)
        torque = np.clip(torque + np.random.normal(0, 1.5), 30, 50)
        wear = np.clip(wear + 0.15, 0, 190)

    elif mode == "heat_buildup":
        # Narrow temperature difference (< 8.6 K) with lower cooling airflow (< 1380 RPM)
        rpm = np.clip(rpm - 25.0, 1150, 1370)
        proc_t = air_t + max(5.0, 10.0 - 0.4 * step)
        torque = np.clip(torque + np.random.normal(0, 1.0), 38, 55)
        wear = np.clip(wear + 0.2, 0, 200)

    elif mode == "power_surge":
        # Over-torque or extreme power (Power = (2*pi/60)*RPM*Torque > 9000W or < 3500W)
        rpm = np.clip(rpm + 40.0, 1800, 2600)
        torque = np.clip(torque + 4.0, 55.0, 75.0)
        proc_t += 0.3
        wear = np.clip(wear + 0.25, 0, 210)

    elif mode == "tool_degradation":
        # Tool wear increases progressively towards 220-250 min
        wear = min(250.0, wear + 3.5)
        torque = np.clip(torque + 0.5, 35.0, 65.0)
        proc_t += 0.1

    elif mode == "overstrain":
        # High torque cutting with high tool wear, exceeding variant limit
        wear = np.clip(wear + 1.0, 175.0, 240.0)
        torque = np.clip(torque + 1.5, 58.0, 75.0)
        rpm = np.clip(rpm - 15.0, 1250, 1450)

    return {
        "type": type_val,
        "air_temperature": round(float(air_t), 2),
        "process_temperature": round(float(proc_t), 2),
        "rotational_speed": int(rpm),
        "torque": round(float(torque), 2),
        "tool_wear": int(wear),
    }


def get_curated_case_studies() -> Dict[str, Dict]:
    """Returns curated representative machines illustrating key industrial failure modes."""
    return {
        "Nominal Healthy Machine (CNC-01)": {
            "type": "M",
            "air_temperature": 298.5,
            "process_temperature": 309.2,
            "rotational_speed": 1540,
            "torque": 38.5,
            "tool_wear": 25,
            "expected_outcome": "Healthy - Normal Operation",
            "description": "Standard steady machining conditions with optimal cooling and low tool wear."
        },
        "Heat Dissipation Failure Risk (CNC-04)": {
            "type": "L",
            "air_temperature": 302.8,
            "process_temperature": 309.4,  # Delta T = 6.6 K (< 8.6 K)
            "rotational_speed": 1310,      # RPM < 1380
            "torque": 48.0,
            "tool_wear": 120,
            "expected_outcome": "Failure Risk: Heat Dissipation (HDF)",
            "description": "Insufficient convective cooling with low spindle speed and small thermal differential."
        },
        "Spindle Overstrain Failure (CNC-09)": {
            "type": "L",
            "air_temperature": 299.0,
            "process_temperature": 310.5,
            "rotational_speed": 1360,
            "torque": 66.5,                # High torque
            "tool_wear": 205,              # Overstrain Product = 66.5 * 205 = 13,632 (> 11,000 for L)
            "expected_outcome": "Failure Risk: Overstrain Breakdown (OSF)",
            "description": "Excessive cutting resistance on a heavily worn tool exceeding mechanical strain limits."
        },
        "High-Power Motor Overload (CNC-14)": {
            "type": "M",
            "air_temperature": 298.0,
            "process_temperature": 308.5,
            "rotational_speed": 2450,      # High RPM
            "torque": 52.0,                # High torque -> Mechanical Power ~ 13,340 W (> 9000 W limit)
            "tool_wear": 85,
            "expected_outcome": "Failure Risk: Motor Overload Power Failure (PWF)",
            "description": "Extreme mechanical wattage demand exceeding spindle drive inverter capacity."
        },
        "Tool Wear Failure Imminent (CNC-22)": {
            "type": "H",
            "air_temperature": 300.2,
            "process_temperature": 311.0,
            "rotational_speed": 1490,
            "torque": 44.0,
            "tool_wear": 235,              # Wear >= 200 min warning cutoff
            "expected_outcome": "Failure Risk: Tool Wear Replacement Needed (TWF)",
            "description": "Cutting edge flank wear near end-of-life; risk of workpiece damage and tool rupture."
        }
    }
