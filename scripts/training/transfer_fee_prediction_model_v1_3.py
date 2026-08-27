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

DATA_FILE = Path(
    "data/processed/training_dataset_european.csv"
)

BASE_MODEL_FILE = Path(
    "models/transfer_fee_model_v1_1.joblib"
)

MODEL_C_FILE = Path(
    "models/transfer_fee_model_v1_3_c.joblib"
)

MODEL_D_FILE = Path(
    "models/transfer_fee_model_v1_3_d.joblib"
)

ENSEMBLE_FILE = Path(
    "models/transfer_fee_model_v1_3.joblib"
)

METADATA_FILE = Path(
    "models/transfer_fee_model_v1_3_metadata.json"
)


TARGET = "transfer_fee"

ALPHA_C = 0.4
ALPHA_D = 0.6


# ============================================================
# Sample Weight
# ============================================================

WEIGHT_UNDER_30M = 1.0
WEIGHT_30_TO_50M = 2.5
WEIGHT_50M_PLUS = 4.0


# ============================================================
# Model C / D
# ============================================================

C_REMOVE_FEATURES = [
    "is_same_league",
]

D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# ============================================================
# 유럽대항전 Feature
# ============================================================

EUROPEAN_FEATURES = [
    "ucl_appearances",
    "ucl_starts",
    "ucl_goals",
    "ucl_assists",

    "uel_appearances",
    "uel_starts",
    "uel_goals",
    "uel_assists",

    "uecl_appearances",
    "uecl_starts",
    "uecl_goals",
    "uecl_assists",
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
        (
            df["goals"]
            + df["assists"]
        )
        / df["minutes"]
        * 90,
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

    df["age_squared"] = (
        df["age_at_transfer"] ** 2
    )

    return df


# ============================================================
# Weight
# ============================================================

def make_weight(y):
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
# Pipeline 유틸
# ============================================================

def clone_model(model):
    try:
        return clone(model)

    except Exception:
        return copy.deepcopy(model)


def find_column_transformer(model):

    if not hasattr(model, "steps"):
        raise ValueError(
            "모델이 sklearn Pipeline 형태가 아닙니다."
        )

    for step_name, step in model.steps:

        if isinstance(
            step,
            ColumnTransformer,
        ):
            return (
                step_name,
                step,
            )

    raise ValueError(
        "ColumnTransformer를 찾지 못했습니다."
    )


def get_estimator_step_name(model):

    if (
        not hasattr(model, "steps")
        or len(model.steps) == 0
    ):
        raise ValueError(
            "Estimator step을 찾지 못했습니다."
        )

    return model.steps[-1][0]


# ============================================================
# 기존 feature 제거
# ============================================================

def remove_features_from_model(
    base_model,
    features_to_remove,
):
    model = clone_model(
        base_model
    )

    remove_set = set(
        features_to_remove
    )

    _, preprocessor = (
        find_column_transformer(
            model
        )
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
# Numeric Transformer에 유럽대항전 Feature 추가
# ============================================================

def add_numeric_features_to_model(
    model,
    features_to_add,
):

    _, preprocessor = (
        find_column_transformer(
            model
        )
    )

    numeric_reference_features = {
        "age_at_transfer",
        "height",
        "matches",
        "started",
        "goals",
        "assists",
        "minutes",
        "rating",
        "goals_per90",
        "assists_per90",
        "goal_contributions_per90",
        "starts_ratio",
        "minutes_per_match",
        "age_squared",
    }

    best_index = None
    best_score = 0

    for index, (
        name,
        transformer,
        columns,
    ) in enumerate(
        preprocessor.transformers
    ):

        if not isinstance(
            columns,
            (
                list,
                tuple,
                np.ndarray,
                pd.Index,
            ),
        ):
            continue

        score = len(
            set(columns)
            & numeric_reference_features
        )

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        raise ValueError(
            "Numeric transformer를 찾지 못했습니다."
        )

    new_transformers = []

    for index, (
        name,
        transformer,
        columns,
    ) in enumerate(
        preprocessor.transformers
    ):

        if (
            index == best_index
            and isinstance(
                columns,
                (
                    list,
                    tuple,
                    np.ndarray,
                    pd.Index,
                ),
            )
        ):

            new_columns = list(
                columns
            )

            for feature in features_to_add:

                if feature not in new_columns:
                    new_columns.append(
                        feature
                    )

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
# Model 생성
# ============================================================

def build_model(
    base_model,
    removed_features,
):

    model = (
        remove_features_from_model(
            base_model,
            removed_features,
        )
    )

    model = (
        add_numeric_features_to_model(
            model,
            EUROPEAN_FEATURES,
        )
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

    features += EUROPEAN_FEATURES

    model = build_model(
        base_model,
        removed_features,
    )

    X = df[
        features
    ]

    y = df[
        TARGET
    ]

    sample_weight = make_weight(
        y
    )

    estimator_step = (
        get_estimator_step_name(
            model
        )
    )

    fit_params = {
        (
            f"{estimator_step}"
            "__sample_weight"
        ):
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
# 저장 모델 검증
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
        "european_features",
    }

    missing_keys = (
        required_keys
        - set(bundle.keys())
    )

    if missing_keys:

        raise ValueError(
            "v1.3 bundle에 값이 없습니다: "
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

    for feature in EUROPEAN_FEATURES:

        if feature not in bundle["features_c"]:
            raise ValueError(
                f"Model C에 Europe feature 없음: "
                f"{feature}"
            )

        if feature not in bundle["features_d"]:
            raise ValueError(
                f"Model D에 Europe feature 없음: "
                f"{feature}"
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
            "Model C에 to_league_id가 없습니다."
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
    print(
        "Transfer Fee Prediction Model v1.3 - Final Training"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"데이터 없음: {DATA_FILE}"
        )

    if not BASE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"v1.1 base model 없음: "
            f"{BASE_MODEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. 데이터 / 모델 로드
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

    if (
        "value_at_transfer"
        in base_features
    ):
        raise ValueError(
            "Market value 모델입니다."
        )

    print()
    print(
        "✓ no-market v1.1 base model 확인"
    )

    print(
        "원본 rows:",
        len(df),
    )

    # --------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------

    df["transfer_date"] = (
        pd.to_datetime(
            df["transfer_date"],
            errors="coerce",
        )
    )

    df = add_v11_features(
        df
    )

    # Europe feature 확인
    missing_european = [
        feature
        for feature in EUROPEAN_FEATURES
        if feature not in df.columns
    ]

    if missing_european:

        raise ValueError(
            "유럽대항전 feature가 없습니다:\n"
            f"{missing_european}"
        )

    for feature in EUROPEAN_FEATURES:

        df[feature] = (
            pd.to_numeric(
                df[feature],
                errors="coerce",
            )
            .fillna(0)
        )

    # 기존 feature 확인
    missing_base = [
        feature
        for feature in base_features
        if feature not in df.columns
    ]

    if missing_base:

        raise ValueError(
            "기존 feature가 없습니다:\n"
            f"{missing_base}"
        )

    # --------------------------------------------------------
    # 4. Target 정리
    # --------------------------------------------------------

    before_rows = len(
        df
    )

    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()

    after_rows = len(
        df
    )

    if before_rows != after_rows:

        print(
            "제거 rows:",
            before_rows - after_rows,
        )

    years = (
        df["transfer_date"]
        .dt.year
        .dropna()
    )

    print(
        "최종 학습 rows:",
        len(df),
    )

    print(
        "학습 연도:",
        int(years.min()),
        "~",
        int(years.max()),
    )

    # --------------------------------------------------------
    # 5. Europe 데이터 통계
    # --------------------------------------------------------

    total_europe_appearances = (
        df["ucl_appearances"]
        + df["uel_appearances"]
        + df["uecl_appearances"]
    )

    print()
    print(
        "유럽대항전 실제 출전 rows:",
        int(
            (
                total_europe_appearances
                > 0
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # 6. Model C
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("1/2 Model C 학습")
    print("=" * 72)

    (
        model_c,
        features_c,
        weights_c,
    ) = train_variant(
        base_model,
        df,
        base_features,
        C_REMOVE_FEATURES,
    )

    print(
        "제거 feature:",
        C_REMOVE_FEATURES,
    )

    print(
        "사용 feature 수:",
        len(features_c),
    )

    print(
        "Europe feature 수:",
        len(EUROPEAN_FEATURES),
    )

    print(
        "✓ Model C 학습 완료"
    )

    # --------------------------------------------------------
    # 7. Model D
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("2/2 Model D 학습")
    print("=" * 72)

    (
        model_d,
        features_d,
        weights_d,
    ) = train_variant(
        base_model,
        df,
        base_features,
        D_REMOVE_FEATURES,
    )

    print(
        "제거 feature:",
        D_REMOVE_FEATURES,
    )

    print(
        "사용 feature 수:",
        len(features_d),
    )

    print(
        "✓ Model D 학습 완료"
    )

    # --------------------------------------------------------
    # 8. Weight 분포
    # --------------------------------------------------------

    unique_weights, counts = (
        np.unique(
            weights_c,
            return_counts=True,
        )
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
    # 9. 저장
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

        "version":
            "1.3",

        "model_type":
            "weighted_ensemble",

        "market_value_used":
            False,

        "model_c":
            model_c,

        "model_d":
            model_d,

        "alpha_c":
            ALPHA_C,

        "alpha_d":
            ALPHA_D,

        "features_c":
            features_c,

        "features_d":
            features_d,

        "european_features":
            EUROPEAN_FEATURES,

        "removed_features_c":
            C_REMOVE_FEATURES,

        "removed_features_d":
            D_REMOVE_FEATURES,

        "sample_weight": {

            "under_30m":
                WEIGHT_UNDER_30M,

            "30m_to_50m":
                WEIGHT_30_TO_50M,

            "50m_plus":
                WEIGHT_50M_PLUS,
        },

        "target_transform":
            "log1p",

        "prediction_inverse":
            "expm1",
    }

    joblib.dump(
        ensemble_bundle,
        ENSEMBLE_FILE,
    )

    # --------------------------------------------------------
    # 10. Metadata
    # --------------------------------------------------------

    metadata = {

        "model_name":
            "transfer_fee_model_v1_3",

        "version":
            "1.3",

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "data_file":
            str(DATA_FILE),

        "training_rows":
            int(len(df)),

        "training_year_min":
            int(years.min()),

        "training_year_max":
            int(years.max()),

        "target":
            TARGET,

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

        "european_features":
            EUROPEAN_FEATURES,

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

        "validation": {

            "comparison":
                "v1.2 vs v1.2 + European features",

            "train_years":
                "2020-2024",

            "test_year":
                2025,

            "v1_2": {

                "mae_m":
                    7.564,

                "rmse_m":
                    11.432,

                "r2":
                    0.6821,

                "30m_plus_mae_m":
                    19.030,

                "50m_plus_mae_m":
                    23.495,

                "top10_mae_m":
                    29.853,
            },

            "v1_3_candidate": {

                "mae_m":
                    6.651,

                "rmse_m":
                    11.020,

                "r2":
                    0.7046,

                "30m_plus_mae_m":
                    14.718,

                "50m_plus_mae_m":
                    21.253,

                "top10_mae_m":
                    30.241,
            },
        },

        "notes": [

            (
                "v1.2의 C/D 앙상블 및 "
                "sample weight 구조 유지"
            ),

            (
                "유럽대항전 UCL/UEL/UECL의 "
                "appearances, starts, goals, assists "
                "12개 feature 추가"
            ),

            (
                "유럽대항전 minutes/rating은 "
                "현재 버전에서 사용하지 않음"
            ),

            (
                "has_ucl/has_uel/has_uecl은 "
                "appearances와 중복 정보이므로 제외"
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
    # 11. 재로드 검증
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
        f"Ensemble: "
        f"C {bundle['alpha_c'] * 100:.0f}% "
        f"+ D {bundle['alpha_d'] * 100:.0f}%"
    )

    print()
    print(
        "Model C feature 수:",
        len(
            bundle["features_c"]
        ),
    )

    print(
        "Model D feature 수:",
        len(
            bundle["features_d"]
        ),
    )

    print(
        "Europe feature 수:",
        len(
            bundle[
                "european_features"
            ]
        ),
    )

    print()
    print("=" * 72)
    print(
        "✓ transfer_fee_model_v1_3 생성 완료"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()