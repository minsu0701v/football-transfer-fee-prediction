from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Nullable Value
# ============================================================

def nullable_value(
    value: Any,
) -> Any:
    """
    NaN → None
    NumPy 자료형 → Python 자료형
    """

    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================
# Calculate Age
# ============================================================

def calculate_age(
    date_of_birth: Any,
    prediction_date: pd.Timestamp,
) -> float:
    """
    생년월일 기준 나이 계산
    """

    birth_date = pd.to_datetime(
        date_of_birth,
        errors="coerce",
    )

    if pd.isna(birth_date):
        return np.nan

    return (
        prediction_date - birth_date
    ).days / 365.25