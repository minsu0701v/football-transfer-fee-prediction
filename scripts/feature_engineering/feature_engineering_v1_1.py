from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v1_1.csv"
)


# ============================================================
# Feature Engineering
# ============================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    0으로 나누는 경우 NaN 처리.
    이후 모델의 SimpleImputer에서 처리한다.
    """
    denominator = denominator.replace(0, np.nan)

    return numerator / denominator


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # 숫자형 변환
    numeric_columns = [
        "matches",
        "started",
        "goals",
        "assists",
        "minutes",
        "rating",
        "age_at_transfer",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 1. 90분당 공격 생산성
    # --------------------------------------------------------

    data["goals_per90"] = (
        safe_divide(
            data["goals"],
            data["minutes"],
        )
        * 90
    )

    data["assists_per90"] = (
        safe_divide(
            data["assists"],
            data["minutes"],
        )
        * 90
    )

    data["goal_contributions_per90"] = (
        safe_divide(
            data["goals"] + data["assists"],
            data["minutes"],
        )
        * 90
    )

    # --------------------------------------------------------
    # 2. 출전 비율
    # --------------------------------------------------------

    data["starts_ratio"] = safe_divide(
        data["started"],
        data["matches"],
    )

    data["minutes_per_match"] = safe_divide(
        data["minutes"],
        data["matches"],
    )

    # --------------------------------------------------------
    # 3. 나이 비선형 Feature
    # --------------------------------------------------------

    data["age_squared"] = (
        data["age_at_transfer"] ** 2
    )

    return data


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {INPUT_FILE}"
        )

    print("=" * 60)
    print("Feature Engineering V1.1")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(f"원본 행 수   : {len(df):,}")
    print(f"원본 컬럼 수 : {len(df.columns)}")

    result = add_features(df)

    new_features = [
        "goals_per90",
        "assists_per90",
        "goal_contributions_per90",
        "starts_ratio",
        "minutes_per_match",
        "age_squared",
    ]

    print("\n[추가 Feature]")

    for feature in new_features:
        print(
            f"{feature:<30}"
            f"결측 {result[feature].isna().sum():>5,}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"최종 행 수   : {len(result):,}")
    print(f"최종 컬럼 수 : {len(result.columns)}")
    print(f"저장 완료    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()