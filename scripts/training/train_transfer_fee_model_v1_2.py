import copy
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer


# ============================================================
# 설정
# ============================================================

DATA_FILE = Path("data/processed/training_dataset.csv")
BASE_MODEL_FILE = Path("models/transfer_fee_model_v1_1.joblib")

MODEL_C_FILE = Path("models/transfer_fee_model_v1_2_c.joblib")
MODEL_D_FILE = Path("models/transfer_fee_model_v1_2_d.joblib")
ENSEMBLE_FILE = Path("models/transfer_fee_model_v1_2.joblib")
METADATA_FILE = Path("models/transfer_fee_model_v1_2_metadata.json")

TARGET = "transfer_fee"

ALPHA_C = 0.4
ALPHA_D = 0.6

# Weight C
WEIGHT_UNDER_30M = 1.0
WEIGHT_30_TO_50M = 2.5
WEIGHT_50M_PLUS = 4.0

# Model C:
# - is_same_league 제거
# - to_league_id 유지
C_REMOVE_FEATURES = [
    "is_same_league",
]

# Model D:
# - is_same_league 제거
# - to_league_id 제거
D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# ============================================================
# v1.1 파생변수
# ============================================================

def add_v11_features(df):
    df = df.copy()

    df["goals_per90"] = np.where(
        df["minutes"] > 0,
        df["goals"] / df["minutes"] * 90,
        0,
    )

    df["assists_per90"] = np.where(
        df["minutes"] > 0,
        df["assists"] / df["minutes"] * 90,
        0,
    )

    df["goal_contributions_per90"] = np.where(
        df["minutes"] > 0,
        (df["goals"] + df["assists"]) / df["minutes"] * 90,
        0,
    )

    df["starts_ratio"] = np.where(
        df["matches"] > 0,
        df["started"] / df["matches"],
        0,
    )

    df["minutes_per_match"] = np.where(
        df["matches"] > 0,
        df["minutes"] / df["matches"],
        0,
    )

    df["age_squared"] = df["age_at_transfer"] ** 2

    return df


# ============================================================
# Weight C
# ============================================================

def make_weight_c(y):
    y = np.asarray(y)

    weights = np.ones(
        len(y),
        dtype=float,
    )

    weights[
        (y >= 30_000_000)
        & (y < 50_000_000)
    ] = WEIGHT_30_TO_50M

    weights[
        y >= 50_000_000
    ] = WEIGHT_50M_PLUS

    return weights


# ============================================================
# sklearn Pipeline 유틸
# ============================================================

def clone_model(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def find_column_transformer(model):
    if not hasattr(model, "steps"):
        raise ValueError(
            "불러온 v1.1 모델이 sklearn Pipeline 형태가 아닙니다."
        )

    for step_name, step in model.steps:
        if isinstance(step, ColumnTransformer):
            return step_name, step

    raise ValueError(
        "Pipeline 안에서 ColumnTransformer를 찾지 못했습니다."
    )


def get_estimator_step_name(model):
    if not hasattr(model, "steps") or len(model.steps) == 0:
        raise ValueError(
            "불러온 모델에서 estimator step을 찾을 수 없습니다."
        )

    return model.steps[-1][0]


def remove_features_from_model(
    base_model,
    features_to_remove,
):
    model = clone_model(base_model)

    remove_set = set(features_to_remove)

    _, preprocessor = find_column_transformer(
        model
    )

    new_transformers = []

    for (
        name,
        transformer,
        columns,
    ) in preprocessor.transformers:

        if isinstance(
            columns,
            (
                list,
                tuple,
                np.ndarray,
                pd.Index,
            ),
        ):
            new_columns = [
                column
                for column in columns
                if column not in remove_set
            ]

            new_transformers.append(
                (
                    name,
                    transformer,
                    new_columns,
                )
            )
        else:
            new_transformers.append(
                (
                    name,
                    transformer,
                    columns,
                )
            )

    preprocessor.transformers = (
        new_transformers
    )

    return model


# ============================================================
# 최종 모델 학습
# ============================================================

def train_variant(
    base_model,
    df,
    base_features,
    removed_features,
):
    features = [
        feature
        for feature in base_features
        if feature not in removed_features
    ]

    model = remove_features_from_model(
        base_model,
        removed_features,
    )

    X = df[features]
    y = df[TARGET]

    sample_weight = make_weight_c(y)

    estimator_step = get_estimator_step_name(
        model
    )

    fit_params = {
        f"{estimator_step}__sample_weight":
            sample_weight
    }

    model.fit(
        X,
        np.log1p(y),
        **fit_params,
    )

    return (
        model,
        features,
        sample_weight,
    )


# ============================================================
# 간단 검증
# ============================================================

def verify_saved_model(
    ensemble_path,
):
    bundle = joblib.load(
        ensemble_path
    )

    required_keys = {
        "version",
        "model_c",
        "model_d",
        "alpha_c",
        "alpha_d",
        "features_c",
        "features_d",
    }

    missing_keys = (
        required_keys
        - set(bundle.keys())
    )

    if missing_keys:
        raise ValueError(
            "저장된 v1.2 bundle에 다음 값이 없습니다: "
            f"{sorted(missing_keys)}"
        )

    if not np.isclose(
        bundle["alpha_c"]
        + bundle["alpha_d"],
        1.0,
    ):
        raise ValueError(
            "앙상블 가중치 합이 1이 아닙니다."
        )

    if (
        "is_same_league"
        in bundle["features_c"]
    ):
        raise ValueError(
            "Model C에 is_same_league가 남아 있습니다."
        )

    if (
        "is_same_league"
        in bundle["features_d"]
    ):
        raise ValueError(
            "Model D에 is_same_league가 남아 있습니다."
        )

    if (
        "to_league_id"
        not in bundle["features_c"]
    ):
        raise ValueError(
            "Model C에서 to_league_id가 사라졌습니다."
        )

    if (
        "to_league_id"
        in bundle["features_d"]
    ):
        raise ValueError(
            "Model D에 to_league_id가 남아 있습니다."
        )

    return bundle


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print("Transfer Fee Prediction Model v1.2 - Final Training")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}"
        )

    if not BASE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"v1.1 모델을 찾을 수 없습니다: {BASE_MODEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. 데이터 / v1.1 모델 로드
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    base_model = joblib.load(
        BASE_MODEL_FILE
    )

    base_features = list(
        base_model.feature_names_in_
    )

    if "value_at_transfer" in base_features:
        raise ValueError(
            "현재 base model은 market value를 사용합니다. "
            "no-market v1.1 모델이 아닙니다."
        )

    print()
    print("✓ no-market v1.1 모델 확인")
    print(f"원본 데이터 rows: {len(df)}")

    # --------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df = add_v11_features(
        df
    )

    missing_columns = [
        feature
        for feature in base_features
        if feature not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 다음 v1.1 feature가 없습니다:\n"
            f"{missing_columns}"
        )

    # target 결측 제거
    before_rows = len(df)

    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()

    after_rows = len(df)

    if before_rows != after_rows:
        print(
            f"Target 결측/비정상 행 제거: "
            f"{before_rows - after_rows}개"
        )

    years = (
        df["transfer_date"]
        .dt.year
        .dropna()
    )

    if len(years) > 0:
        print(
            "학습 연도:",
            int(years.min()),
            "~",
            int(years.max()),
        )

    print(
        f"최종 학습 rows: {len(df)}"
    )

    # --------------------------------------------------------
    # 4. Model C
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("1/2 Model C 학습")
    print("=" * 72)

    model_c, features_c, weights_c = (
        train_variant(
            base_model,
            df,
            base_features,
            C_REMOVE_FEATURES,
        )
    )

    print(
        "제거 feature:",
        C_REMOVE_FEATURES,
    )

    print(
        f"사용 feature 수: {len(features_c)}"
    )

    print(
        "✓ Model C 학습 완료"
    )

    # --------------------------------------------------------
    # 5. Model D
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("2/2 Model D 학습")
    print("=" * 72)

    model_d, features_d, weights_d = (
        train_variant(
            base_model,
            df,
            base_features,
            D_REMOVE_FEATURES,
        )
    )

    print(
        "제거 feature:",
        D_REMOVE_FEATURES,
    )

    print(
        f"사용 feature 수: {len(features_d)}"
    )

    print(
        "✓ Model D 학습 완료"
    )

    # --------------------------------------------------------
    # 6. Weight 분포 출력
    # --------------------------------------------------------

    unique_weights, counts = np.unique(
        weights_c,
        return_counts=True,
    )

    print()
    print("=" * 72)
    print("Sample Weight 분포")
    print("=" * 72)

    for weight, count in zip(
        unique_weights,
        counts,
    ):
        print(
            f"weight={weight:.1f}: "
            f"{count} rows"
        )

    # --------------------------------------------------------
    # 7. 저장
    # --------------------------------------------------------

    MODEL_C_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_c,
        MODEL_C_FILE,
    )

    joblib.dump(
        model_d,
        MODEL_D_FILE,
    )

    ensemble_bundle = {
        "version": "1.2",
        "model_type": "weighted_ensemble",
        "market_value_used": False,

        "model_c": model_c,
        "model_d": model_d,

        "alpha_c": ALPHA_C,
        "alpha_d": ALPHA_D,

        "features_c": features_c,
        "features_d": features_d,

        "removed_features_c": (
            C_REMOVE_FEATURES
        ),

        "removed_features_d": (
            D_REMOVE_FEATURES
        ),

        "sample_weight": {
            "under_30m": (
                WEIGHT_UNDER_30M
            ),
            "30m_to_50m": (
                WEIGHT_30_TO_50M
            ),
            "50m_plus": (
                WEIGHT_50M_PLUS
            ),
        },

        "target_transform": "log1p",
        "prediction_inverse": "expm1",
    }

    joblib.dump(
        ensemble_bundle,
        ENSEMBLE_FILE,
    )

    # --------------------------------------------------------
    # 8. metadata JSON
    # --------------------------------------------------------

    metadata = {
        "model_name":
            "transfer_fee_model_v1_2",

        "version":
            "1.2",

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "data_file":
            str(DATA_FILE),

        "training_rows":
            int(len(df)),

        "training_year_min":
            (
                int(years.min())
                if len(years) > 0
                else None
            ),

        "training_year_max":
            (
                int(years.max())
                if len(years) > 0
                else None
            ),

        "target":
            TARGET,

        "target_transform":
            "log1p",

        "market_value_used":
            False,

        "ensemble": {
            "model_c_weight":
                ALPHA_C,

            "model_d_weight":
                ALPHA_D,
        },

        "sample_weight": {
            "under_30m":
                WEIGHT_UNDER_30M,

            "30m_to_50m":
                WEIGHT_30_TO_50M,

            "50m_plus":
                WEIGHT_50M_PLUS,
        },

        "model_c": {
            "removed_features":
                C_REMOVE_FEATURES,

            "features":
                features_c,
        },

        "model_d": {
            "removed_features":
                D_REMOVE_FEATURES,

            "features":
                features_d,
        },

        "validation_selection": {
            "train_years":
                "2020-2023",

            "validation_year":
                2024,

            "selected_alpha_c":
                ALPHA_C,

            "selected_alpha_d":
                ALPHA_D,

            "selection_rule":
                (
                    "GB1 vs ES1 counterfactual "
                    "premium <= 40% 조건 내에서 "
                    "낮은 validation MAE를 우선"
                ),
        },

        "final_test_2025": {
            "mae_m":
                7.564,

            "rmse_m":
                11.432,

            "r2":
                0.682,

            "top10_mae_m":
                20.218,

            "high_value_50m_plus_mae_m":
                23.495,

            "high_value_50m_plus_pred_mean_m":
                53.407,

            "gb1_vs_es1_counterfactual_pct":
                31.639,
        },

        "notes": [
            (
                "transfer_year는 별도 실험에서 "
                "유의미한 성능 개선이 없어 제외"
            ),
            (
                "is_same_league는 ablation 결과 "
                "제거"
            ),
            (
                "to_league_id의 과도한 목적 리그 "
                "민감도를 완화하기 위해 C/D "
                "앙상블 사용"
            ),
        ],
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 9. 저장 결과 재검증
    # --------------------------------------------------------

    bundle = verify_saved_model(
        ENSEMBLE_FILE
    )

    print()
    print("=" * 72)
    print("저장 및 재로드 검증")
    print("=" * 72)

    print(
        "✓ Model C:",
        MODEL_C_FILE,
    )

    print(
        "✓ Model D:",
        MODEL_D_FILE,
    )

    print(
        "✓ Ensemble:",
        ENSEMBLE_FILE,
    )

    print(
        "✓ Metadata:",
        METADATA_FILE,
    )

    print()
    print(
        "Ensemble:"
    )

    print(
        f"C {bundle['alpha_c'] * 100:.0f}% "
        f"+ D {bundle['alpha_d'] * 100:.0f}%"
    )

    print()
    print(
        "Model C features:"
    )

    print(
        bundle["features_c"]
    )

    print()
    print(
        "Model D features:"
    )

    print(
        bundle["features_d"]
    )

    print()
    print("=" * 72)
    print("✓ transfer_fee_model_v1_2 생성 완료")
    print("=" * 72)


if __name__ == "__main__":
    main()