from pathlib import Path

import pandas as pd


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Data Files
# ============================================================

PREDICTION_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_dataset.csv"
)

TRAINING_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset.csv"
)


# ============================================================
# Model File
# ============================================================

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transfer_fee_model_v1_2.joblib"
)


# ============================================================
# Prediction Settings
# ============================================================

PREDICTION_DATE = pd.Timestamp(
    "2026-07-01"
)

TOP5_LEAGUE_IDS = {
    "GB1",
    "ES1",
    "IT1",
    "FR1",
    "L1",
}


# ============================================================
# Model Features - v1.2 (No Market Value)
# ============================================================

MODEL_FEATURES = [
    "age_at_transfer",
    "height",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "is_same_league",
    "is_top5_destination",
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "starts_ratio",
    "minutes_per_match",
    "age_squared",
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]