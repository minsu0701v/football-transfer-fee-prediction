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
# SAMPLE WEIGHT
# v1.3과 동일
# ============================================================

WEIGHT_UNDER_30M = 1.0
WEIGHT_30_TO_50M = 2.5
WEIGHT_50M_PLUS = 4.0


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
# TARGET TRANSFORMS
# ============================================================

TRANSFORMS = [
    "log1p",
    "sqrt",
    "raw",
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
# TARGET TRANSFORM
# ============================================================

def transform_target(
    y,
    transform_name,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    if transform_name == "log1p":

        return np.log1p(y)

    if transform_name == "sqrt":

        return np.sqrt(y)

    if transform_name == "raw":

        return y

    raise ValueError(
        f"Unknown transform: {transform_name}"
    )


# ============================================================
# TARGET INVERSE
# ============================================================

def inverse_target(
    prediction,
    transform_name,
):

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    if transform_name == "log1p":

        prediction = np.expm1(
            prediction
        )

    elif transform_name == "sqrt":

        # sqrt 공간에서 음수 예측은
        # 실제 금액으로 의미가 없으므로 0 처리 후 제곱
        prediction = np.maximum(
            prediction,
            0,
        )

        prediction = (
            prediction ** 2
        )

    elif transform_name == "raw":

        pass

    else:

        raise ValueError(
            f"Unknown transform: "
            f"{transform_name}"
        )

    return np.maximum(
        prediction,
        0,
    )


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
            "모델이 sklearn Pipeline이 아닙니다."
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
# FEATURE REMOVE
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
            "Numeric transformer를 "
            "찾지 못했습니다."
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
    transform_name,
):

    features = [
        feature
        for feature in base_features
        if feature not in removed_features
    ]

    features += (
        EUROPEAN_FEATURES
    )

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

    sample_weight = (
        make_weight(
            y_train
        )
    )

    y_train_transformed = (
        transform_target(
            y_train,
            transform_name,
        )
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
        y_train_transformed,
        **fit_params,
    )

    return (
        model,
        features,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    prediction,
):

    mae = (
        mean_absolute_error(
            y_true,
            prediction,
        )
        / 1_000_000
    )

    rmse = (
        np.sqrt(
            mean_squared_error(
                y_true,
                prediction,
            )
        )
        / 1_000_000
    )

    r2 = r2_score(
        y_true,
        prediction,
    )

    error_m = (
        prediction
        - y_true
    ) / 1_000_000

    return {
        "count": len(y_true),

        "mae_m": mae,
        "rmse_m": rmse,
        "r2": r2,

        "mean_error_m": (
            np.mean(
                error_m
            )
        ),

        "median_error_m": (
            np.median(
                error_m
            )
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
# EVALUATE RANGE
# ============================================================

def evaluate_range(
    transform_name,
    range_name,
    y_true,
    predictions,
    mask,
):

    actual = y_true[
        mask
    ]

    pred = predictions[
        mask
    ]

    if len(actual) == 0:

        return None

    metrics = (
        calculate_metrics(
            actual,
            pred,
        )
    )

    metrics["transform"] = (
        transform_name
    )

    metrics["range"] = (
        range_name
    )

    return metrics


# ============================================================
# PRINT RESULT
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
        "TARGET TRANSFORM COMPARISON"
    )

    print(
        "log1p vs sqrt vs raw"
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
    # EUROPE
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
    # SPLIT
    # ========================================================

    train_df = df[
        df["year"] < 2025
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
        f"Train rows : "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows  : "
        f"{len(test_df):,}"
    )


    y_true = (
        test_df[TARGET]
        .astype(float)
        .to_numpy()
    )


    # ========================================================
    # Q90
    # ========================================================

    q90 = np.quantile(
        y_true,
        0.90,
    )

    print(
        f"Test Top10% threshold: "
        f"{q90 / 1_000_000:.3f}M"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    summary_rows = []

    prediction_output = (
        test_df[
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

    prediction_output[
        "actual_fee_m"
    ] = (
        prediction_output[TARGET]
        / 1_000_000
    )


    # ========================================================
    # EACH TRANSFORM
    # ========================================================

    for transform_name in TRANSFORMS:

        print(
            "\n\n"
            + "=" * 100
        )

        print(
            f"TARGET TRANSFORM: "
            f"{transform_name.upper()}"
        )

        print(
            "=" * 100
        )


        # ====================================================
        # MODEL C
        # ====================================================

        print(
            "\nTraining Model C..."
        )

        (
            model_c,
            features_c,
        ) = train_variant(
            base_model=base_model,
            train_df=train_df,
            base_features=base_features,
            removed_features=C_REMOVE_FEATURES,
            transform_name=transform_name,
        )


        # ====================================================
        # MODEL D
        # ====================================================

        print(
            "Training Model D..."
        )

        (
            model_d,
            features_d,
        ) = train_variant(
            base_model=base_model,
            train_df=train_df,
            base_features=base_features,
            removed_features=D_REMOVE_FEATURES,
            transform_name=transform_name,
        )


        # ====================================================
        # PREDICT
        # ====================================================

        pred_c_transformed = (
            model_c.predict(
                test_df[
                    features_c
                ]
            )
        )

        pred_d_transformed = (
            model_d.predict(
                test_df[
                    features_d
                ]
            )
        )


        # ====================================================
        # INVERSE
        # ====================================================

        pred_c = inverse_target(
            pred_c_transformed,
            transform_name,
        )

        pred_d = inverse_target(
            pred_d_transformed,
            transform_name,
        )


        # ====================================================
        # ENSEMBLE
        # ====================================================

        predictions = (
            ALPHA_C * pred_c
            + ALPHA_D * pred_d
        )

        predictions = np.maximum(
            predictions,
            0,
        )


        # ====================================================
        # SAVE PLAYER PREDICTION
        # ====================================================

        prediction_output[
            f"{transform_name}_pred_m"
        ] = (
            predictions
            / 1_000_000
        )

        prediction_output[
            f"{transform_name}_error_m"
        ] = (
            (
                predictions
                - y_true
            )
            / 1_000_000
        )


        # ====================================================
        # RANGE MASKS
        # ====================================================

        masks = {
            "ALL": (
                np.ones(
                    len(y_true),
                    dtype=bool,
                )
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
                y_true
                >= q90
            ),
        }


        print()

        for (
            range_name,
            mask,
        ) in masks.items():

            metrics = (
                evaluate_range(
                    transform_name,
                    range_name,
                    y_true,
                    predictions,
                    mask,
                )
            )

            if metrics is None:

                continue

            summary_rows.append(
                metrics
            )

            print_result(
                range_name,
                metrics,
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    column_order = [
        "transform",
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

    summary_df = summary_df[
        column_order
    ]


    print(
        "\n\n"
        + "=" * 120
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 120
    )


    print(
        summary_df
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # PIVOT - MAE
    # ========================================================

    mae_pivot = (
        summary_df
        .pivot(
            index="transform",
            columns="range",
            values="mae_m",
        )
    )


    preferred_columns = [
        "ALL",
        "30M+",
        "50M+",
        "70M+",
        "TOP10%",
    ]

    mae_pivot = mae_pivot[
        [
            column
            for column
            in preferred_columns
            if column
            in mae_pivot.columns
        ]
    ]


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MAE COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        mae_pivot
        .round(3)
        .to_string()
    )


    # ========================================================
    # HIGH VALUE BIAS
    # ========================================================

    bias_pivot = (
        summary_df
        .pivot(
            index="transform",
            columns="range",
            values="mean_error_m",
        )
    )

    bias_columns = [
        "30M+",
        "50M+",
        "70M+",
    ]

    bias_pivot = bias_pivot[
        [
            column
            for column
            in bias_columns
            if column
            in bias_pivot.columns
        ]
    ]


    print(
        "\n"
        + "=" * 100
    )

    print(
        "HIGH VALUE MEAN ERROR"
    )

    print(
        "음수 = 저평가 / 양수 = 고평가"
    )

    print(
        "=" * 100
    )

    print(
        bias_pivot
        .round(3)
        .to_string()
    )


    # ========================================================
    # BASELINE 대비 개선
    # ========================================================

    log_baseline = (
        summary_df[
            summary_df[
                "transform"
            ] == "log1p"
        ]
        .set_index(
            "range"
        )
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "MAE CHANGE VS LOG1P"
    )

    print(
        "음수 = 개선 / 양수 = 악화"
    )

    print(
        "=" * 100
    )


    for transform_name in [
        "sqrt",
        "raw",
    ]:

        transform_df = (
            summary_df[
                summary_df[
                    "transform"
                ] == transform_name
            ]
            .set_index(
                "range"
            )
        )

        print(
            f"\n[{transform_name}]"
        )

        for range_name in [
            "ALL",
            "30M+",
            "50M+",
            "70M+",
        ]:

            if (
                range_name
                not in transform_df.index
                or range_name
                not in log_baseline.index
            ):

                continue

            delta = (
                transform_df.loc[
                    range_name,
                    "mae_m",
                ]
                - log_baseline.loc[
                    range_name,
                    "mae_m",
                ]
            )

            print(
                f"{range_name:<6}: "
                f"{delta:+.3f}M"
            )


    # ========================================================
    # SAVE
    # ========================================================

    summary_path = (
        RESULT_DIR
        / "target_transform_comparison.csv"
    )

    prediction_path = (
        RESULT_DIR
        / "target_transform_2025_predictions.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_output.to_csv(
        prediction_path,
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
        summary_path
    )

    print(
        prediction_path
    )


if __name__ == "__main__":
    main()