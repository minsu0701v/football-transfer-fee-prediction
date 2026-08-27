import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 설정
# ============================================================

DATA_FILE = Path(
    "data/processed/training_dataset_european.csv"
)

BASE_MODEL_FILE = Path(
    "models/transfer_fee_model_v1_1.joblib"
)

RESULT_DIR = Path(
    "results"
)

PREDICTION_RESULT_FILE = RESULT_DIR / (
    "v13_age_squared_ablation_predictions.csv"
)

SUMMARY_RESULT_FILE = RESULT_DIR / (
    "v13_age_squared_ablation_summary.csv"
)


TARGET = "transfer_fee"

TRAIN_START_YEAR = 2020
TRAIN_END_YEAR = 2024
TEST_YEAR = 2025


# ============================================================
# Ensemble Weight
# ============================================================

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

# 기존 v1.3
BASE_C_REMOVE_FEATURES = [
    "is_same_league",
]

BASE_D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# age_squared 제거 실험
NO_AGE2_C_REMOVE_FEATURES = [
    "is_same_league",
    "age_squared",
]

NO_AGE2_D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
    "age_squared",
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

def add_v11_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["goals_per90"] = np.where(
        df["minutes"] > 0,
        df["goals"]
        / df["minutes"]
        * 90,
        0,
    )

    df["assists_per90"] = np.where(
        df["minutes"] > 0,
        df["assists"]
        / df["minutes"]
        * 90,
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
        df["started"]
        / df["matches"],
        0,
    )

    df["minutes_per_match"] = np.where(
        df["matches"] > 0,
        df["minutes"]
        / df["matches"],
        0,
    )

    df["age_squared"] = (
        df["age_at_transfer"] ** 2
    )

    return df


# ============================================================
# Sample Weight
# ============================================================

def make_weight(
    y,
) -> np.ndarray:

    y = np.asarray(
        y
    )

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
# Pipeline Utility
# ============================================================

def clone_model(
    model,
):

    try:
        return clone(
            model
        )

    except Exception:
        return copy.deepcopy(
            model
        )


def find_column_transformer(
    model,
):

    if not hasattr(
        model,
        "steps",
    ):
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


def get_estimator_step_name(
    model,
) -> str:

    if (
        not hasattr(
            model,
            "steps",
        )
        or len(model.steps) == 0
    ):
        raise ValueError(
            "Estimator step을 찾지 못했습니다."
        )

    return model.steps[-1][0]


# ============================================================
# Feature 제거
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
# Numeric Transformer에 Europe Feature 추가
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
# 모델 생성
# ============================================================

def build_model(
    base_model,
    removed_features,
):

    model = remove_features_from_model(
        base_model,
        removed_features,
    )

    model = add_numeric_features_to_model(
        model,
        EUROPEAN_FEATURES,
    )

    return model


# ============================================================
# Model C/D 학습
# ============================================================

def train_variant(
    base_model,
    train_df,
    base_features,
    removed_features,
):

    features = [
        feature
        for feature in base_features
        if feature not in removed_features
    ]

    for feature in EUROPEAN_FEATURES:

        if feature not in features:

            features.append(
                feature
            )

    model = build_model(
        base_model,
        removed_features,
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        TARGET
    ]

    sample_weight = make_weight(
        y_train
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
        X_train,
        np.log1p(
            y_train
        ),
        **fit_params,
    )

    return (
        model,
        features,
    )


# ============================================================
# Ensemble 학습
# ============================================================

def train_ensemble(
    base_model,
    train_df,
    base_features,
    c_remove_features,
    d_remove_features,
):

    model_c, features_c = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=c_remove_features,
    )

    model_d, features_d = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=d_remove_features,
    )

    return {
        "model_c": model_c,
        "model_d": model_d,
        "features_c": features_c,
        "features_d": features_d,
    }


# ============================================================
# Ensemble 예측
# ============================================================

def predict_ensemble(
    bundle,
    df,
) -> np.ndarray:

    model_c = bundle[
        "model_c"
    ]

    model_d = bundle[
        "model_d"
    ]

    features_c = bundle[
        "features_c"
    ]

    features_d = bundle[
        "features_d"
    ]

    # --------------------------------------------------------
    # Model C
    # --------------------------------------------------------

    pred_c_log = model_c.predict(
        df[
            features_c
        ]
    )

    pred_c = np.expm1(
        pred_c_log
    )

    pred_c = np.maximum(
        pred_c,
        0,
    )

    # --------------------------------------------------------
    # Model D
    # --------------------------------------------------------

    pred_d_log = model_d.predict(
        df[
            features_d
        ]
    )

    pred_d = np.expm1(
        pred_d_log
    )

    pred_d = np.maximum(
        pred_d,
        0,
    )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    pred = (
        ALPHA_C * pred_c
        + ALPHA_D * pred_d
    )

    return pred


# ============================================================
# Metric
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    if len(y_true) == 0:

        return {
            "count": 0,
            "mae_m": np.nan,
            "rmse_m": np.nan,
            "r2": np.nan,
            "actual_mean_m": np.nan,
            "pred_mean_m": np.nan,
        }

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    # R²은 표본 2개 이상일 때만
    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            y_pred,
        )

    else:

        r2 = np.nan

    return {
        "count":
            len(y_true),

        "mae_m":
            mae / 1_000_000,

        "rmse_m":
            rmse / 1_000_000,

        "r2":
            r2,

        "actual_mean_m":
            np.mean(y_true)
            / 1_000_000,

        "pred_mean_m":
            np.mean(y_pred)
            / 1_000_000,
    }


# ============================================================
# 특정 조건 Metric
# ============================================================

def metric_for_mask(
    df,
    pred_column,
    mask,
):

    subset = df[
        mask
    ]

    return calculate_metrics(
        subset[
            TARGET
        ].to_numpy(),

        subset[
            pred_column
        ].to_numpy(),
    )


# ============================================================
# 결과 출력
# ============================================================

def print_metric_row(
    label,
    base_value,
    no_age2_value,
    digits=3,
):

    if (
        pd.isna(base_value)
        or pd.isna(no_age2_value)
    ):

        print(
            f"{label:<26}"
            f"{str(base_value):>14}"
            f"{str(no_age2_value):>14}"
        )

        return

    print(
        f"{label:<26}"
        f"{base_value:>14.{digits}f}"
        f"{no_age2_value:>14.{digits}f}"
    )


def print_group_comparison(
    label,
    base_metrics,
    no_age2_metrics,
):

    print()
    print("-" * 72)
    print(
        f"{label} "
        f"(n={base_metrics['count']})"
    )
    print("-" * 72)

    print_metric_row(
        "MAE (M)",
        base_metrics["mae_m"],
        no_age2_metrics["mae_m"],
    )

    print_metric_row(
        "RMSE (M)",
        base_metrics["rmse_m"],
        no_age2_metrics["rmse_m"],
    )

    print_metric_row(
        "R²",
        base_metrics["r2"],
        no_age2_metrics["r2"],
        digits=4,
    )

    print_metric_row(
        "Actual Mean (M)",
        base_metrics["actual_mean_m"],
        no_age2_metrics["actual_mean_m"],
    )

    print_metric_row(
        "Pred Mean (M)",
        base_metrics["pred_mean_m"],
        no_age2_metrics["pred_mean_m"],
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "v1.3 age_squared Ablation Test"
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
            f"Base model 없음: "
            f"{BASE_MODEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. 데이터 로드
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
            "Market Value 모델입니다."
        )

    print(
        "✓ no-market v1.1 base model"
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
            "Europe feature 없음:\n"
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
            "Base feature 없음:\n"
            f"{missing_base}"
        )

    # --------------------------------------------------------
    # 4. Target / Date 정리
    # --------------------------------------------------------

    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()

    df = df[
        df["transfer_date"].notna()
    ].copy()

    df["transfer_year"] = (
        df[
            "transfer_date"
        ]
        .dt.year
    )

    # --------------------------------------------------------
    # 5. Train / Test Split
    # --------------------------------------------------------

    train_df = df[
        (
            df["transfer_year"]
            >= TRAIN_START_YEAR
        )
        &
        (
            df["transfer_year"]
            <= TRAIN_END_YEAR
        )
    ].copy()

    test_df = df[
        df["transfer_year"]
        == TEST_YEAR
    ].copy()

    print()
    print(
        f"Train: "
        f"{TRAIN_START_YEAR}"
        f"~{TRAIN_END_YEAR}"
        f" / {len(train_df)} rows"
    )

    print(
        f"Test : "
        f"{TEST_YEAR}"
        f" / {len(test_df)} rows"
    )

    if len(train_df) == 0:

        raise ValueError(
            "Train 데이터가 없습니다."
        )

    if len(test_df) == 0:

        raise ValueError(
            "Test 데이터가 없습니다."
        )

    # --------------------------------------------------------
    # 6. 기존 v1.3 구조 학습
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("1/2 BASE v1.3 학습")
    print("=" * 72)

    base_bundle = train_ensemble(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        c_remove_features=(
            BASE_C_REMOVE_FEATURES
        ),
        d_remove_features=(
            BASE_D_REMOVE_FEATURES
        ),
    )

    print(
        "Model C feature:",
        len(
            base_bundle[
                "features_c"
            ]
        ),
    )

    print(
        "Model D feature:",
        len(
            base_bundle[
                "features_d"
            ]
        ),
    )

    print(
        "✓ BASE 학습 완료"
    )

    # --------------------------------------------------------
    # 7. age_squared 제거 구조 학습
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("2/2 NO AGE_SQUARED 학습")
    print("=" * 72)

    no_age2_bundle = train_ensemble(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        c_remove_features=(
            NO_AGE2_C_REMOVE_FEATURES
        ),
        d_remove_features=(
            NO_AGE2_D_REMOVE_FEATURES
        ),
    )

    print(
        "Model C feature:",
        len(
            no_age2_bundle[
                "features_c"
            ]
        ),
    )

    print(
        "Model D feature:",
        len(
            no_age2_bundle[
                "features_d"
            ]
        ),
    )

    print(
        "✓ NO AGE² 학습 완료"
    )

    # --------------------------------------------------------
    # 8. 예측
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("2025 Test 예측")
    print("=" * 72)

    test_df["pred_base"] = (
        predict_ensemble(
            base_bundle,
            test_df,
        )
    )

    test_df["pred_no_age2"] = (
        predict_ensemble(
            no_age2_bundle,
            test_df,
        )
    )

    # --------------------------------------------------------
    # 9. 전체 Metric
    # --------------------------------------------------------

    base_all = calculate_metrics(
        test_df[
            TARGET
        ].to_numpy(),

        test_df[
            "pred_base"
        ].to_numpy(),
    )

    no_age2_all = calculate_metrics(
        test_df[
            TARGET
        ].to_numpy(),

        test_df[
            "pred_no_age2"
        ].to_numpy(),
    )

    # --------------------------------------------------------
    # 10. 그룹별 Metric
    # --------------------------------------------------------

    masks = {
        "ALL":
            pd.Series(
                True,
                index=test_df.index,
            ),

        "Age < 27":
            test_df[
                "age_at_transfer"
            ] < 27,

        "Age >= 27":
            test_df[
                "age_at_transfer"
            ] >= 27,

        "30M+":
            test_df[
                TARGET
            ] >= 30_000_000,

        "50M+":
            test_df[
                TARGET
            ] >= 50_000_000,

        "Age >= 27 & 30M+":
            (
                (
                    test_df[
                        "age_at_transfer"
                    ]
                    >= 27
                )
                &
                (
                    test_df[
                        TARGET
                    ]
                    >= 30_000_000
                )
            ),
    }

    summary_rows = []

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"{'Metric':<26}"
        f"{'BASE v1.3':>14}"
        f"{'NO AGE²':>14}"
    )

    for group_name, mask in masks.items():

        base_metrics = metric_for_mask(
            test_df,
            "pred_base",
            mask,
        )

        no_age2_metrics = metric_for_mask(
            test_df,
            "pred_no_age2",
            mask,
        )

        print_group_comparison(
            group_name,
            base_metrics,
            no_age2_metrics,
        )

        summary_rows.append(
            {
                "group":
                    group_name,

                "count":
                    base_metrics[
                        "count"
                    ],

                "base_mae_m":
                    base_metrics[
                        "mae_m"
                    ],

                "no_age2_mae_m":
                    no_age2_metrics[
                        "mae_m"
                    ],

                "base_rmse_m":
                    base_metrics[
                        "rmse_m"
                    ],

                "no_age2_rmse_m":
                    no_age2_metrics[
                        "rmse_m"
                    ],

                "base_r2":
                    base_metrics[
                        "r2"
                    ],

                "no_age2_r2":
                    no_age2_metrics[
                        "r2"
                    ],

                "actual_mean_m":
                    base_metrics[
                        "actual_mean_m"
                    ],

                "base_pred_mean_m":
                    base_metrics[
                        "pred_mean_m"
                    ],

                "no_age2_pred_mean_m":
                    no_age2_metrics[
                        "pred_mean_m"
                    ],
            }
        )

    # --------------------------------------------------------
    # 11. Top 10 실제 이적료
    # --------------------------------------------------------

    top10 = (
        test_df
        .sort_values(
            TARGET,
            ascending=False,
        )
        .head(10)
        .copy()
    )

    top10_base_mae = (
        mean_absolute_error(
            top10[
                TARGET
            ],
            top10[
                "pred_base"
            ],
        )
        / 1_000_000
    )

    top10_no_age2_mae = (
        mean_absolute_error(
            top10[
                TARGET
            ],
            top10[
                "pred_no_age2"
            ],
        )
        / 1_000_000
    )

    print()
    print("=" * 72)
    print("TOP 10 TRANSFERS")
    print("=" * 72)

    print(
        f"BASE Top10 MAE   : "
        f"{top10_base_mae:.3f}M"
    )

    print(
        f"NO AGE² Top10 MAE: "
        f"{top10_no_age2_mae:.3f}M"
    )

    print()

    # 이름 컬럼이 있으면 출력
    display_columns = []

    if "player_name" in top10.columns:

        display_columns.append(
            "player_name"
        )

    elif "name" in top10.columns:

        display_columns.append(
            "name"
        )

    elif "player_id" in top10.columns:

        display_columns.append(
            "player_id"
        )

    display_columns += [
        "age_at_transfer",
        TARGET,
        "pred_base",
        "pred_no_age2",
    ]

    display_df = top10[
        display_columns
    ].copy()

    display_df[
        TARGET
    ] = (
        display_df[
            TARGET
        ]
        / 1_000_000
    )

    display_df[
        "pred_base"
    ] = (
        display_df[
            "pred_base"
        ]
        / 1_000_000
    )

    display_df[
        "pred_no_age2"
    ] = (
        display_df[
            "pred_no_age2"
        ]
        / 1_000_000
    )

    print(
        display_df.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # 12. Prediction 변화가 큰 선수
    # --------------------------------------------------------

    test_df[
        "prediction_change"
    ] = (
        test_df[
            "pred_no_age2"
        ]
        - test_df[
            "pred_base"
        ]
    )

    biggest_changes = (
        test_df
        .assign(
            abs_change=lambda x:
                x[
                    "prediction_change"
                ].abs()
        )
        .sort_values(
            "abs_change",
            ascending=False,
        )
        .head(15)
        .copy()
    )

    print()
    print("=" * 72)
    print("AGE² 제거 후 예측 변화가 큰 선수 TOP 15")
    print("=" * 72)

    change_columns = []

    if "player_name" in biggest_changes.columns:

        change_columns.append(
            "player_name"
        )

    elif "name" in biggest_changes.columns:

        change_columns.append(
            "name"
        )

    elif "player_id" in biggest_changes.columns:

        change_columns.append(
            "player_id"
        )

    change_columns += [
        "age_at_transfer",
        TARGET,
        "pred_base",
        "pred_no_age2",
        "prediction_change",
    ]

    change_df = biggest_changes[
        change_columns
    ].copy()

    for column in [
        TARGET,
        "pred_base",
        "pred_no_age2",
        "prediction_change",
    ]:

        change_df[
            column
        ] = (
            change_df[
                column
            ]
            / 1_000_000
        )

    print(
        change_df.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # 13. 저장
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = []

    for column in [
        "player_id",
        "player_name",
        "transfer_date",
        "age_at_transfer",
        "from_league_id",
        "to_league_id",
        TARGET,
        "pred_base",
        "pred_no_age2",
        "prediction_change",
    ]:

        if column in test_df.columns:

            output_columns.append(
                column
            )

    test_df[
        output_columns
    ].to_csv(
        PREDICTION_RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Top10도 summary 마지막에 표시할 수 있도록 별도 출력
    summary_df.to_csv(
        SUMMARY_RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 14. 최종 요약
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(
        f"BASE v1.3"
        f"     | "
        f"MAE {base_all['mae_m']:.3f}M"
        f" | "
        f"RMSE {base_all['rmse_m']:.3f}M"
        f" | "
        f"R² {base_all['r2']:.4f}"
    )

    print(
        f"NO AGE²"
        f"       | "
        f"MAE {no_age2_all['mae_m']:.3f}M"
        f" | "
        f"RMSE {no_age2_all['rmse_m']:.3f}M"
        f" | "
        f"R² {no_age2_all['r2']:.4f}"
    )

    mae_change = (
        no_age2_all[
            "mae_m"
        ]
        - base_all[
            "mae_m"
        ]
    )

    rmse_change = (
        no_age2_all[
            "rmse_m"
        ]
        - base_all[
            "rmse_m"
        ]
    )

    r2_change = (
        no_age2_all[
            "r2"
        ]
        - base_all[
            "r2"
        ]
    )

    print()
    print(
        f"MAE 변화 : "
        f"{mae_change:+.3f}M"
    )

    print(
        f"RMSE 변화: "
        f"{rmse_change:+.3f}M"
    )

    print(
        f"R² 변화  : "
        f"{r2_change:+.4f}"
    )

    print()
    print(
        f"Top10 MAE: "
        f"{top10_base_mae:.3f}M "
        f"→ "
        f"{top10_no_age2_mae:.3f}M"
    )

    print()
    print(
        "Prediction CSV:"
    )

    print(
        PREDICTION_RESULT_FILE
    )

    print(
        "Summary CSV:"
    )

    print(
        SUMMARY_RESULT_FILE
    )

    print()
    print("=" * 72)
    print("AGE_SQUARED ABLATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()