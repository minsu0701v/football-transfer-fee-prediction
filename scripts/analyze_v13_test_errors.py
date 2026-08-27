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
# PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "training_dataset_european.csv"
)

BASE_MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "transfer_fee_model_v1_1.joblib"
)

RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

TARGET = "transfer_fee"

ALPHA_C = 0.4
ALPHA_D = 0.6


# Sample Weight
WEIGHT_UNDER_30M = 1.0
WEIGHT_30_TO_50M = 2.5
WEIGHT_50M_PLUS = 4.0


# Model C
C_REMOVE_FEATURES = [
    "is_same_league",
]


# Model D
D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# Europe features
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
# v1.1 FEATURE ENGINEERING
# 실제 v1.3 학습 코드와 동일
# ============================================================

def add_v11_features(df):

    df = df.copy()

    df["goals_per90"] = np.where(
        df["minutes"] > 0,
        (
            df["goals"]
            / df["minutes"]
            * 90
        ),
        0,
    )

    df["assists_per90"] = np.where(
        df["minutes"] > 0,
        (
            df["assists"]
            / df["minutes"]
            * 90
        ),
        0,
    )

    df["goal_contributions_per90"] = np.where(
        df["minutes"] > 0,
        (
            (
                df["goals"]
                + df["assists"]
            )
            / df["minutes"]
            * 90
        ),
        0,
    )

    df["starts_ratio"] = np.where(
        df["matches"] > 0,
        (
            df["started"]
            / df["matches"]
        ),
        0,
    )

    df["minutes_per_match"] = np.where(
        df["matches"] > 0,
        (
            df["minutes"]
            / df["matches"]
        ),
        0,
    )

    df["age_squared"] = (
        df["age_at_transfer"] ** 2
    )

    return df


# ============================================================
# SAMPLE WEIGHT
# ============================================================

def make_weight(y):

    y = np.asarray(y)

    weights = np.ones(
        len(y),
        dtype=float,
    )

    weights[
        (
            (y >= 30_000_000)
            & (y < 50_000_000)
        )
    ] = WEIGHT_30_TO_50M

    weights[
        y >= 50_000_000
    ] = WEIGHT_50M_PLUS

    return weights


# ============================================================
# PIPELINE UTILS
# ============================================================

def clone_model(model):

    try:
        return clone(model)

    except Exception:
        return copy.deepcopy(model)


def find_column_transformer(model):

    if not hasattr(
        model,
        "steps",
    ):
        raise ValueError(
            "모델이 sklearn Pipeline 형태가 아닙니다."
        )

    for (
        step_name,
        step,
    ) in model.steps:

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
# FEATURE 제거
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
# EUROPE FEATURE 추가
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
# MODEL BUILD
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
# TRAIN VARIANT
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

    features += EUROPEAN_FEATURES

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
        sample_weight,
    )


# ============================================================
# METRIC 출력
# ============================================================

def print_metrics(
    name,
    y_true,
    prediction,
    mask=None,
):

    if mask is not None:

        actual = y_true[mask]
        pred = prediction[mask]

    else:

        actual = y_true
        pred = prediction

    if len(actual) == 0:
        return

    mae = (
        mean_absolute_error(
            actual,
            pred,
        )
        / 1_000_000
    )

    rmse = (
        np.sqrt(
            mean_squared_error(
                actual,
                pred,
            )
        )
        / 1_000_000
    )

    if len(actual) >= 2:

        r2 = r2_score(
            actual,
            pred,
        )

    else:

        r2 = np.nan

    print(
        f"{name:<15}"
        f" rows={len(actual):>4}"
        f" | MAE={mae:>7.3f}M"
        f" | RMSE={rmse:>7.3f}M"
        f" | R²={r2:>7.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "V1.3 - TRUE 2025 TEST ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nDataset    : {DATA_PATH}"
    )

    print(
        f"Base Model : {BASE_MODEL_PATH}"
    )


    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    base_model = joblib.load(
        BASE_MODEL_PATH
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
        "\n✓ no-market v1.1 base model 확인"
    )

    print(
        f"Base features: "
        f"{len(base_features)}"
    )


    # ========================================================
    # DATE
    # ========================================================

    df["transfer_date"] = (
        pd.to_datetime(
            df["transfer_date"],
            errors="coerce",
        )
    )

    df["year"] = (
        df["transfer_date"]
        .dt.year
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = add_v11_features(
        df
    )


    # ========================================================
    # EUROPE FEATURE
    # ========================================================

    missing_european = [
        feature
        for feature in EUROPEAN_FEATURES
        if feature not in df.columns
    ]

    if missing_european:

        raise ValueError(
            "Europe features 없음:\n"
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


    # ========================================================
    # TARGET 정리
    # ========================================================

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()


    # ========================================================
    # TRUE TIME SPLIT
    #
    # 2020~2024 = TRAIN
    # 2025      = TEST
    # ========================================================

    train_df = df[
        df["year"] < 2025
    ].copy()

    test_df = df[
        df["year"] == 2025
    ].copy()


    print(
        "\n"
        + "=" * 80
    )

    print(
        "TIME SPLIT"
    )

    print(
        "=" * 80
    )

    print(
        f"Train rows : "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows  : "
        f"{len(test_df):,}"
    )

    print(
        "Train years: "
        f"{int(train_df['year'].min())}"
        " ~ "
        f"{int(train_df['year'].max())}"
    )

    print(
        "Test year  : "
        f"{int(test_df['year'].min())}"
    )


    # ========================================================
    # TRAIN SAMPLE WEIGHT 확인
    # ========================================================

    train_weights = make_weight(
        train_df[TARGET]
    )

    unique_weights, counts = (
        np.unique(
            train_weights,
            return_counts=True,
        )
    )


    print(
        "\nSample Weight:"
    )

    for weight, count in zip(
        unique_weights,
        counts,
    ):

        print(
            f"weight={weight:.1f}: "
            f"{count} rows"
        )


    # ========================================================
    # MODEL C TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL C TRAIN - 2020~2024"
    )

    print(
        "=" * 80
    )


    (
        model_c,
        features_c,
        _,
    ) = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=C_REMOVE_FEATURES,
    )


    print(
        f"Model C features: "
        f"{len(features_c)}"
    )

    print(
        "✓ Model C 학습 완료"
    )


    # ========================================================
    # MODEL D TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MODEL D TRAIN - 2020~2024"
    )

    print(
        "=" * 80
    )


    (
        model_d,
        features_d,
        _,
    ) = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=D_REMOVE_FEATURES,
    )


    print(
        f"Model D features: "
        f"{len(features_d)}"
    )

    print(
        "✓ Model D 학습 완료"
    )


    # ========================================================
    # 2025 PREDICTION
    # ========================================================

    pred_c_log = model_c.predict(
        test_df[
            features_c
        ]
    )

    pred_d_log = model_d.predict(
        test_df[
            features_d
        ]
    )


    # log1p(EUR) -> EUR
    pred_c = np.expm1(
        pred_c_log
    )

    pred_d = np.expm1(
        pred_d_log
    )


    pred_c = np.maximum(
        pred_c,
        0,
    )

    pred_d = np.maximum(
        pred_d,
        0,
    )


    # ========================================================
    # C 40% + D 60%
    # ========================================================

    predictions = (
        ALPHA_C * pred_c
        + ALPHA_D * pred_d
    )


    predictions = np.maximum(
        predictions,
        0,
    )


    y_true = (
        test_df[TARGET]
        .to_numpy()
    )


    # ========================================================
    # METRICS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "2025 TRUE TEST METRICS"
    )

    print(
        "=" * 80
    )


    print_metrics(
        "ALL",
        y_true,
        predictions,
    )


    print_metrics(
        "30M+",
        y_true,
        predictions,
        y_true >= 30_000_000,
    )


    print_metrics(
        "50M+",
        y_true,
        predictions,
        y_true >= 50_000_000,
    )


    print_metrics(
        "70M+",
        y_true,
        predictions,
        y_true >= 70_000_000,
    )


    # ========================================================
    # TOP 10% MAE
    # 기존 v1.3 metadata 검증용
    # ========================================================

    q90 = np.quantile(
        y_true,
        0.90,
    )

    top10_mask = (
        y_true >= q90
    )

    top10_mae = (
        mean_absolute_error(
            y_true[
                top10_mask
            ],
            predictions[
                top10_mask
            ],
        )
        / 1_000_000
    )


    print(
        f"\nTop 10% 기준: "
        f"{q90 / 1_000_000:.3f}M"
    )

    print(
        f"Top 10% MAE: "
        f"{top10_mae:.3f}M"
    )


    # ========================================================
    # RESULT COLUMNS
    # ========================================================

    result = test_df.copy()

    result["actual_fee_m"] = (
        y_true
        / 1_000_000
    )

    result["pred_c_m"] = (
        pred_c
        / 1_000_000
    )

    result["pred_d_m"] = (
        pred_d
        / 1_000_000
    )

    result["predicted_fee_m"] = (
        predictions
        / 1_000_000
    )


    # prediction - actual
    #
    # 음수 = 과소예측
    # 양수 = 과대예측

    result["error_m"] = (
        result["predicted_fee_m"]
        - result["actual_fee_m"]
    )

    result["abs_error_m"] = (
        result["error_m"]
        .abs()
    )

    result["error_pct"] = np.where(
        result["actual_fee_m"] > 0,
        (
            result["error_m"]
            / result["actual_fee_m"]
            * 100
        ),
        np.nan,
    )

    result["error_direction"] = (
        np.where(
            result["error_m"] < 0,
            "UNDER",
            "OVER",
        )
    )


    # ========================================================
    # DISPLAY COLUMNS
    # ========================================================

    candidate_columns = [

        "player_name",

        "year",
        "transfer_date",
        "previous_season",

        "age_at_transfer",

        "from_team_name",
        "to_team_name",

        "from_league_name",
        "to_league_name",

        "main_position",

        "matches",
        "started",

        "goals",
        "assists",

        "minutes",
        "rating",

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

        "actual_fee_m",

        "pred_c_m",
        "pred_d_m",
        "predicted_fee_m",

        "error_m",
        "abs_error_m",
        "error_pct",

        "error_direction",
    ]


    display_columns = [
        column
        for column in candidate_columns
        if column in result.columns
    ]


    # ========================================================
    # ABS ERROR TOP 20
    # ========================================================

    error_top20 = (
        result
        .sort_values(
            "abs_error_m",
            ascending=False,
        )
        .head(20)
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "TOP 20 - LARGEST ABSOLUTE ERROR"
    )

    print(
        "=" * 120
    )


    print(
        error_top20[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # UNDERPREDICTED TOP 20
    # ========================================================

    under_top20 = (
        result[
            result["error_m"] < 0
        ]
        .sort_values(
            "error_m",
            ascending=True,
        )
        .head(20)
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "TOP 20 - MOST UNDERPREDICTED"
    )

    print(
        "=" * 120
    )


    print(
        under_top20[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # OVERPREDICTED TOP 20
    # ========================================================

    over_top20 = (
        result[
            result["error_m"] > 0
        ]
        .sort_values(
            "error_m",
            ascending=False,
        )
        .head(20)
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "TOP 20 - MOST OVERPREDICTED"
    )

    print(
        "=" * 120
    )


    print(
        over_top20[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # HIGH VALUE BIAS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "HIGH VALUE ERROR SUMMARY"
    )

    print(
        "=" * 80
    )


    for threshold in [
        30,
        50,
        70,
    ]:

        subset = result[
            result["actual_fee_m"]
            >= threshold
        ]

        if len(subset) == 0:
            continue

        under_count = (
            subset["error_m"]
            < 0
        ).sum()

        over_count = (
            subset["error_m"]
            > 0
        ).sum()

        mean_error = (
            subset["error_m"]
            .mean()
        )

        median_error = (
            subset["error_m"]
            .median()
        )


        print(
            f"\n{threshold}M+"
        )

        print(
            f"Count        : "
            f"{len(subset)}"
        )

        print(
            f"Underpredict : "
            f"{under_count}"
        )

        print(
            f"Overpredict  : "
            f"{over_count}"
        )

        print(
            f"Mean Error   : "
            f"{mean_error:.3f}M"
        )

        print(
            f"Median Error : "
            f"{median_error:.3f}M"
        )


    # ========================================================
    # SAVE
    # ========================================================

    all_path = (
        RESULT_DIR
        / "v13_true_2025_all_errors.csv"
    )

    error_path = (
        RESULT_DIR
        / "v13_true_2025_error_top20.csv"
    )

    under_path = (
        RESULT_DIR
        / "v13_true_2025_underpredicted_top20.csv"
    )

    over_path = (
        RESULT_DIR
        / "v13_true_2025_overpredicted_top20.csv"
    )


    result[
        display_columns
    ].sort_values(
        "abs_error_m",
        ascending=False,
    ).to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )


    error_top20[
        display_columns
    ].to_csv(
        error_path,
        index=False,
        encoding="utf-8-sig",
    )


    under_top20[
        display_columns
    ].to_csv(
        under_path,
        index=False,
        encoding="utf-8-sig",
    )


    over_top20[
        display_columns
    ].to_csv(
        over_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "SAVE COMPLETE"
    )

    print(
        "=" * 80
    )

    print(all_path)
    print(error_path)
    print(under_path)
    print(over_path)


if __name__ == "__main__":
    main()