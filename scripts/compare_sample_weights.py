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


# ============================================================
# MODEL SELECTION 기준
#
# Validation에서 어떤 범위의 MAE를 기준으로
# Weight Config를 선택할지 결정
#
# 기본값:
# 전체 성능 기준
# ============================================================

SELECTION_RANGE = "ALL"


# ============================================================
# SAMPLE WEIGHT EXPERIMENTS
#
# (<30M, 30~50M, 50M+)
# ============================================================

WEIGHT_CONFIGS = {

    # v1.3 baseline
    "A_1_2.5_4": (
        1.0,
        2.5,
        4.0,
    ),

    "B_1_3_5": (
        1.0,
        3.0,
        5.0,
    ),

    "C_1_3_6": (
        1.0,
        3.0,
        6.0,
    ),

    "D_1_4_6": (
        1.0,
        4.0,
        6.0,
    ),

    "E_1_4_8": (
        1.0,
        4.0,
        8.0,
    ),

    "F_1_5_10": (
        1.0,
        5.0,
        10.0,
    ),
}


# ============================================================
# MODEL VARIANTS
# ============================================================

C_REMOVE_FEATURES = [
    "is_same_league",
]

D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# ============================================================
# EUROPE FEATURES
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
# FEATURE ENGINEERING
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

def make_weight(
    y,
    weight_under_30,
    weight_30_50,
    weight_50_plus,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    weights = np.full(
        len(y),
        weight_under_30,
        dtype=float,
    )

    mask_30_50 = (
        (y >= 30_000_000)
        & (y < 50_000_000)
    )

    mask_50_plus = (
        y >= 50_000_000
    )

    weights[
        mask_30_50
    ] = weight_30_50

    weights[
        mask_50_plus
    ] = weight_50_plus

    return weights


# ============================================================
# PIPELINE UTILS
# ============================================================

def clone_model(model):

    try:
        return clone(model)

    except Exception:

        return copy.deepcopy(
            model
        )


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
# REMOVE FEATURES
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
# ADD EUROPE FEATURES
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
# BUILD MODEL
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
    weight_config,
):

    (
        weight_under_30,
        weight_30_50,
        weight_50_plus,
    ) = weight_config

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

    y_train = (
        train_df[TARGET]
        .astype(float)
        .to_numpy()
    )

    sample_weight = make_weight(
        y=y_train,

        weight_under_30=(
            weight_under_30
        ),

        weight_30_50=(
            weight_30_50
        ),

        weight_50_plus=(
            weight_50_plus
        ),
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

    # ========================================================
    # Target Transform:
    # log1p 고정
    # ========================================================

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
# PREDICT ENSEMBLE
# ============================================================

def predict_ensemble(
    model_c,
    model_d,
    features_c,
    features_d,
    df,
):

    pred_c_log = (
        model_c.predict(
            df[
                features_c
            ]
        )
    )

    pred_d_log = (
        model_d.predict(
            df[
                features_d
            ]
        )
    )

    # log1p(EUR)
    # ↓
    # EUR

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

    predictions = (
        ALPHA_C * pred_c
        + ALPHA_D * pred_d
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    return (
        pred_c,
        pred_d,
        predictions,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    mae = (
        mean_absolute_error(
            y_true,
            predictions,
        )
        / 1_000_000
    )

    rmse = (
        np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        )
        / 1_000_000
    )

    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            predictions,
        )

    else:

        r2 = np.nan

    error_m = (
        predictions
        - y_true
    ) / 1_000_000

    return {

        "count": len(
            y_true
        ),

        "mae_m": mae,

        "rmse_m": rmse,

        "r2": r2,

        "mean_error_m": np.mean(
            error_m
        ),

        "median_error_m": np.median(
            error_m
        ),

        "under_count": int(
            np.sum(
                error_m < 0
            )
        ),

        "over_count": int(
            np.sum(
                error_m > 0
            )
        ),
    }


# ============================================================
# RANGE MASKS
# ============================================================

def build_masks(
    y_true,
):

    q90 = np.quantile(
        y_true,
        0.90,
    )

    masks = {

        "ALL": np.ones(
            len(y_true),
            dtype=bool,
        ),

        "30M+": (
            y_true
            >= 30_000_000
        ),

        "50M+": (
            y_true
            >= 50_000_000
        ),

        "70M+": (
            y_true
            >= 70_000_000
        ),

        "TOP10%": (
            y_true >= q90
        ),
    }

    return (
        masks,
        q90,
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_predictions(
    config_name,
    y_true,
    predictions,
    weight_config,
):

    (
        weight_under_30,
        weight_30_50,
        weight_50_plus,
    ) = weight_config

    masks, q90 = build_masks(
        y_true
    )

    rows = []

    for (
        range_name,
        mask,
    ) in masks.items():

        actual = y_true[
            mask
        ]

        pred = predictions[
            mask
        ]

        if len(actual) == 0:
            continue

        metrics = (
            calculate_metrics(
                actual,
                pred,
            )
        )

        metrics["config"] = (
            config_name
        )

        metrics["range"] = (
            range_name
        )

        metrics[
            "weight_under_30"
        ] = weight_under_30

        metrics[
            "weight_30_50"
        ] = weight_30_50

        metrics[
            "weight_50_plus"
        ] = weight_50_plus

        rows.append(
            metrics
        )

    return (
        rows,
        q90,
    )


# ============================================================
# PRINT RANGE RESULT
# ============================================================

def print_result(
    range_name,
    metrics,
):

    print(
        f"{range_name:<10}"
        f" rows={metrics['count']:>4}"
        f" | MAE={metrics['mae_m']:>7.3f}M"
        f" | RMSE={metrics['rmse_m']:>7.3f}M"
        f" | R²={metrics['r2']:>7.4f}"
        f" | Bias={metrics['mean_error_m']:>8.3f}M"
        f" | U/O="
        f"{metrics['under_count']}/"
        f"{metrics['over_count']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "SAMPLE WEIGHT VALIDATION + FINAL TEST"
    )

    print(
        "=" * 100
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

    print(
        f"\nDataset    : "
        f"{DATA_PATH}"
    )

    print(
        f"Base Model : "
        f"{BASE_MODEL_PATH}"
    )

    print(
        f"Base features: "
        f"{len(base_features)}"
    )

    if (
        "value_at_transfer"
        in base_features
    ):

        raise ValueError(
            "Market Value가 포함된 "
            "base model입니다."
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
    # EUROPE FEATURES
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
    # TARGET
    # ========================================================

    df[TARGET] = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
    )

    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()


    # ========================================================
    # 1차 SPLIT
    #
    # Train      : 2020~2023
    # Validation : 2024
    # Test       : 2025
    # ========================================================

    train_df = df[
        df["year"] <= 2023
    ].copy()

    valid_df = df[
        df["year"] == 2024
    ].copy()

    test_df = df[
        df["year"] == 2025
    ].copy()


    print(
        "\n"
        + "=" * 100
    )

    print(
        "TIME SPLIT"
    )

    print(
        "=" * 100
    )

    print(
        f"Train rows      : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows : "
        f"{len(valid_df):,}"
    )

    print(
        f"Test rows       : "
        f"{len(test_df):,}"
    )

    print(
        "Train years     : "
        f"{int(train_df['year'].min())}"
        " ~ "
        f"{int(train_df['year'].max())}"
    )

    print(
        "Validation year : 2024"
    )

    print(
        "Test year       : 2025"
    )


    # ========================================================
    # VALIDATION
    #
    # 여기에서만 Weight Config를 비교한다.
    # ========================================================

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# PHASE 1 - 2024 VALIDATION"
    )

    print(
        "#" * 100
    )


    y_valid = (
        valid_df[TARGET]
        .astype(float)
        .to_numpy()
    )

    validation_rows = []

    validation_predictions = (
        valid_df[
            [
                "player_name",
                "transfer_date",
                "age_at_transfer",
                "from_team_name",
                "to_team_name",
                "main_position",
                TARGET,
            ]
        ]
        .copy()
    )

    validation_predictions[
        "actual_fee_m"
    ] = (
        validation_predictions[TARGET]
        / 1_000_000
    )


    for (
        config_name,
        weight_config,
    ) in WEIGHT_CONFIGS.items():

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"VALIDATION CONFIG: "
            f"{config_name}"
        )

        print(
            "=" * 100
        )

        print(
            f"Weights: "
            f"{weight_config}"
        )


        # ====================================================
        # Train C
        # ====================================================

        (
            model_c,
            features_c,
        ) = train_variant(
            base_model=base_model,
            train_df=train_df,
            base_features=base_features,
            removed_features=C_REMOVE_FEATURES,
            weight_config=weight_config,
        )


        # ====================================================
        # Train D
        # ====================================================

        (
            model_d,
            features_d,
        ) = train_variant(
            base_model=base_model,
            train_df=train_df,
            base_features=base_features,
            removed_features=D_REMOVE_FEATURES,
            weight_config=weight_config,
        )


        # ====================================================
        # Validation Prediction
        # ====================================================

        (
            pred_c,
            pred_d,
            predictions,
        ) = predict_ensemble(
            model_c=model_c,
            model_d=model_d,
            features_c=features_c,
            features_d=features_d,
            df=valid_df,
        )


        validation_predictions[
            f"{config_name}_pred_m"
        ] = (
            predictions
            / 1_000_000
        )


        # ====================================================
        # Validation Metrics
        # ====================================================

        (
            rows,
            q90_valid,
        ) = evaluate_predictions(
            config_name=config_name,
            y_true=y_valid,
            predictions=predictions,
            weight_config=weight_config,
        )

        validation_rows.extend(
            rows
        )


        for row in rows:

            print_result(
                row["range"],
                row,
            )


    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    validation_summary = (
        pd.DataFrame(
            validation_rows
        )
    )


    validation_summary = (
        validation_summary[
            [
                "config",

                "weight_under_30",
                "weight_30_50",
                "weight_50_plus",

                "range",
                "count",

                "mae_m",
                "rmse_m",
                "r2",

                "mean_error_m",
                "median_error_m",

                "under_count",
                "over_count",
            ]
        ]
    )


    print(
        "\n\n"
        + "=" * 120
    )

    print(
        "2024 VALIDATION FINAL COMPARISON"
    )

    print(
        "=" * 120
    )


    print(
        validation_summary
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # VALIDATION MAE PIVOT
    # ========================================================

    valid_mae_pivot = (
        validation_summary
        .pivot(
            index="config",
            columns="range",
            values="mae_m",
        )
    )

    preferred_ranges = [
        "ALL",
        "30M+",
        "50M+",
        "70M+",
        "TOP10%",
    ]

    valid_mae_pivot = (
        valid_mae_pivot[
            [
                column
                for column
                in preferred_ranges
                if column
                in valid_mae_pivot.columns
            ]
        ]
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "2024 VALIDATION MAE"
    )

    print(
        "=" * 100
    )

    print(
        valid_mae_pivot
        .round(3)
        .to_string()
    )


    # ========================================================
    # BEST CONFIG 선택
    #
    # 2025는 여기서 절대 사용하지 않는다.
    # ========================================================

    selection_df = (
        validation_summary[
            validation_summary["range"]
            == SELECTION_RANGE
        ]
        .copy()
    )

    selection_df = (
        selection_df
        .sort_values(
            by=[
                "mae_m",
                "rmse_m",
            ],
            ascending=True,
        )
    )

    best_row = (
        selection_df
        .iloc[0]
    )

    best_config_name = (
        best_row["config"]
    )

    best_weight_config = (
        WEIGHT_CONFIGS[
            best_config_name
        ]
    )


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# MODEL SELECTION COMPLETE"
    )

    print(
        "#" * 100
    )

    print(
        f"\nSelection metric : "
        f"2024 {SELECTION_RANGE} MAE"
    )

    print(
        f"Selected Config  : "
        f"{best_config_name}"
    )

    print(
        f"Selected Weights : "
        f"{best_weight_config}"
    )

    print(
        f"Validation MAE   : "
        f"{best_row['mae_m']:.3f}M"
    )

    print(
        f"Validation RMSE  : "
        f"{best_row['rmse_m']:.3f}M"
    )

    print(
        f"Validation R²    : "
        f"{best_row['r2']:.4f}"
    )


    # ========================================================
    # PHASE 2
    #
    # 선택한 Weight만 사용
    #
    # 2020~2024 전체로 재학습
    # ========================================================

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# PHASE 2 - RETRAIN 2020~2024"
    )

    print(
        "#" * 100
    )


    final_train_df = df[
        df["year"] <= 2024
    ].copy()


    print(
        f"\nFinal Train rows: "
        f"{len(final_train_df):,}"
    )

    print(
        "Final Train years: "
        f"{int(final_train_df['year'].min())}"
        " ~ "
        f"{int(final_train_df['year'].max())}"
    )


    # ========================================================
    # FINAL MODEL C
    # ========================================================

    print(
        "\nTraining Final Model C..."
    )

    (
        final_model_c,
        final_features_c,
    ) = train_variant(
        base_model=base_model,
        train_df=final_train_df,
        base_features=base_features,
        removed_features=C_REMOVE_FEATURES,
        weight_config=best_weight_config,
    )


    # ========================================================
    # FINAL MODEL D
    # ========================================================

    print(
        "Training Final Model D..."
    )

    (
        final_model_d,
        final_features_d,
    ) = train_variant(
        base_model=base_model,
        train_df=final_train_df,
        base_features=base_features,
        removed_features=D_REMOVE_FEATURES,
        weight_config=best_weight_config,
    )


    # ========================================================
    # PHASE 3
    #
    # FINAL 2025 TEST
    #
    # 선택된 Config 딱 하나만 평가
    # ========================================================

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# PHASE 3 - FINAL 2025 TEST"
    )

    print(
        "#" * 100
    )


    y_test = (
        test_df[TARGET]
        .astype(float)
        .to_numpy()
    )


    (
        test_pred_c,
        test_pred_d,
        test_predictions,
    ) = predict_ensemble(
        model_c=final_model_c,
        model_d=final_model_d,
        features_c=final_features_c,
        features_d=final_features_d,
        df=test_df,
    )


    (
        test_rows,
        q90_test,
    ) = evaluate_predictions(
        config_name=best_config_name,
        y_true=y_test,
        predictions=test_predictions,
        weight_config=best_weight_config,
    )


    print(
        f"\nSelected Config: "
        f"{best_config_name}"
    )

    print(
        f"Selected Weight: "
        f"{best_weight_config}"
    )

    print(
        f"2025 Top10% threshold: "
        f"{q90_test / 1_000_000:.3f}M"
    )

    print()


    for row in test_rows:

        print_result(
            row["range"],
            row,
        )


    # ========================================================
    # TEST SUMMARY
    # ========================================================

    test_summary = pd.DataFrame(
        test_rows
    )


    test_summary = test_summary[
        [
            "config",

            "weight_under_30",
            "weight_30_50",
            "weight_50_plus",

            "range",
            "count",

            "mae_m",
            "rmse_m",
            "r2",

            "mean_error_m",
            "median_error_m",

            "under_count",
            "over_count",
        ]
    ]


    # ========================================================
    # PLAYER TEST OUTPUT
    # ========================================================

    test_predictions_df = (
        test_df[
            [
                "player_name",
                "transfer_date",
                "age_at_transfer",

                "from_team_name",
                "to_team_name",

                "from_league_name",
                "to_league_name",

                "main_position",

                TARGET,
            ]
        ]
        .copy()
    )


    test_predictions_df[
        "actual_fee_m"
    ] = (
        y_test
        / 1_000_000
    )

    test_predictions_df[
        "pred_c_m"
    ] = (
        test_pred_c
        / 1_000_000
    )

    test_predictions_df[
        "pred_d_m"
    ] = (
        test_pred_d
        / 1_000_000
    )

    test_predictions_df[
        "predicted_fee_m"
    ] = (
        test_predictions
        / 1_000_000
    )

    test_predictions_df[
        "error_m"
    ] = (
        (
            test_predictions
            - y_test
        )
        / 1_000_000
    )

    test_predictions_df[
        "abs_error_m"
    ] = (
        test_predictions_df[
            "error_m"
        ]
        .abs()
    )

    test_predictions_df[
        "error_direction"
    ] = np.where(
        test_predictions_df[
            "error_m"
        ] < 0,
        "UNDER",
        "OVER",
    )


    # ========================================================
    # TEST ERROR TOP 20
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "2025 FINAL TEST - ERROR TOP 20"
    )

    print(
        "=" * 120
    )


    print(
        test_predictions_df
        .sort_values(
            "abs_error_m",
            ascending=False,
        )
        .head(20)
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    validation_summary_path = (
        RESULT_DIR
        / "sample_weight_2024_validation.csv"
    )

    validation_prediction_path = (
        RESULT_DIR
        / "sample_weight_2024_predictions.csv"
    )

    test_summary_path = (
        RESULT_DIR
        / "sample_weight_2025_final_test.csv"
    )

    test_prediction_path = (
        RESULT_DIR
        / "sample_weight_2025_final_predictions.csv"
    )


    validation_summary.to_csv(
        validation_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    validation_predictions.to_csv(
        validation_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_summary.to_csv(
        test_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_predictions_df.to_csv(
        test_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAVE COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        validation_summary_path
    )

    print(
        validation_prediction_path
    )

    print(
        test_summary_path
    )

    print(
        test_prediction_path
    )


if __name__ == "__main__":
    main()