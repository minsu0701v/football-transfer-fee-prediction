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

SAMPLE_WEIGHT_CONFIG = (
    1.0,
    2.5,
    4.0,
)


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
# BUYER PROFILE FEATURES
# ============================================================

BUYER_PROFILE_FEATURES = [
    "to_club_hist_count",
    "to_club_avg_buy_fee_m",
    "to_club_median_buy_fee_m",
    "to_club_max_buy_fee_m",
    "to_club_30m_plus_ratio",
]


# ============================================================
# SELLER PROFILE FEATURES
# ============================================================

SELLER_PROFILE_FEATURES = [
    "from_club_hist_count",
    "from_club_avg_sale_fee_m",
    "from_club_median_sale_fee_m",
    "from_club_max_sale_fee_m",
    "from_club_30m_plus_ratio",
]


# ============================================================
# ALL CLUB PROFILE FEATURES
# ============================================================

CLUB_PROFILE_FEATURES = (
    BUYER_PROFILE_FEATURES
    + SELLER_PROFILE_FEATURES
)


# ============================================================
# BUYER PROFILE FEATURES
# ============================================================

BUYER_PROFILE_FEATURES = [
    "to_club_hist_count",
    "to_club_avg_buy_fee_m",
    "to_club_median_buy_fee_m",
    "to_club_max_buy_fee_m",
    "to_club_30m_plus_ratio",
]


# ============================================================
# SELLER PROFILE FEATURES
# 기존 feature 생성 함수 때문에 정의는 유지
# ============================================================

SELLER_PROFILE_FEATURES = [
    "from_club_hist_count",
    "from_club_avg_sale_fee_m",
    "from_club_median_sale_fee_m",
    "from_club_max_sale_fee_m",
    "from_club_30m_plus_ratio",
]


CLUB_PROFILE_FEATURES = (
    BUYER_PROFILE_FEATURES
    + SELLER_PROFILE_FEATURES
)


# ============================================================
# BUYER FEATURE ABLATION
# ============================================================

EXPERIMENTS = {

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------
    "BASE": [],


    # --------------------------------------------------------
    # Buyer 전체
    # --------------------------------------------------------
    "BUYER_ALL": (
        BUYER_PROFILE_FEATURES
    ),


    # --------------------------------------------------------
    # SINGLE FEATURE
    # 하나만 넣었을 때
    # --------------------------------------------------------
    "ONLY_COUNT": [
        "to_club_hist_count",
    ],

    "ONLY_AVG": [
        "to_club_avg_buy_fee_m",
    ],

    "ONLY_MEDIAN": [
        "to_club_median_buy_fee_m",
    ],

    "ONLY_MAX": [
        "to_club_max_buy_fee_m",
    ],

    "ONLY_30M_RATIO": [
        "to_club_30m_plus_ratio",
    ],


    # --------------------------------------------------------
    # LEAVE-ONE-OUT
    # Buyer 전체에서 하나씩 제거
    # --------------------------------------------------------
    "NO_COUNT": [
        "to_club_avg_buy_fee_m",
        "to_club_median_buy_fee_m",
        "to_club_max_buy_fee_m",
        "to_club_30m_plus_ratio",
    ],

    "NO_AVG": [
        "to_club_hist_count",
        "to_club_median_buy_fee_m",
        "to_club_max_buy_fee_m",
        "to_club_30m_plus_ratio",
    ],

    "NO_MEDIAN": [
        "to_club_hist_count",
        "to_club_avg_buy_fee_m",
        "to_club_max_buy_fee_m",
        "to_club_30m_plus_ratio",
    ],

    "NO_MAX": [
        "to_club_hist_count",
        "to_club_avg_buy_fee_m",
        "to_club_median_buy_fee_m",
        "to_club_30m_plus_ratio",
    ],

    "NO_30M_RATIO": [
        "to_club_hist_count",
        "to_club_avg_buy_fee_m",
        "to_club_median_buy_fee_m",
        "to_club_max_buy_fee_m",
    ],
}


# ============================================================
# FEATURE ENGINEERING
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
            (df["goals"] + df["assists"])
            / df["minutes"]
            * 90
        ),
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
# LEAKAGE-SAFE CLUB PROFILE
# ============================================================

def add_club_profile_features(df):

    df = df.copy()

    for feature in CLUB_PROFILE_FEATURES:
        df[feature] = 0.0

    years = sorted(
        df["year"]
        .dropna()
        .astype(int)
        .unique()
    )

    for year in years:

        # ====================================================
        # 현재 연도보다 과거 거래만 사용
        # ====================================================

        history = df[
            df["year"] < year
        ].copy()

        current_mask = (
            df["year"] == year
        )

        current = df[
            current_mask
        ].copy()

        if history.empty:
            continue


        # ====================================================
        # BUYER PROFILE
        #
        # 해당 클럽이 과거에 선수 영입에
        # 얼마를 지불했는지
        # ====================================================

        buyer_history = (
            history[
                history["to_team_id"].notna()
            ]
            .copy()
        )

        buyer_history[
            "is_30m_plus"
        ] = (
            buyer_history[TARGET]
            >= 30_000_000
        ).astype(float)

        buyer_stats = (
            buyer_history
            .groupby(
                "to_team_id"
            )
            .agg(
                to_club_hist_count=(
                    TARGET,
                    "count",
                ),

                to_club_avg_buy_fee=(
                    TARGET,
                    "mean",
                ),

                to_club_median_buy_fee=(
                    TARGET,
                    "median",
                ),

                to_club_max_buy_fee=(
                    TARGET,
                    "max",
                ),

                to_club_30m_plus_ratio=(
                    "is_30m_plus",
                    "mean",
                ),
            )
        )


        # ====================================================
        # SELLER PROFILE
        #
        # 해당 클럽이 과거에 선수 판매로
        # 얼마를 받은 적이 있는지
        # ====================================================

        seller_history = (
            history[
                history["from_team_id"].notna()
            ]
            .copy()
        )

        seller_history[
            "is_30m_plus"
        ] = (
            seller_history[TARGET]
            >= 30_000_000
        ).astype(float)

        seller_stats = (
            seller_history
            .groupby(
                "from_team_id"
            )
            .agg(
                from_club_hist_count=(
                    TARGET,
                    "count",
                ),

                from_club_avg_sale_fee=(
                    TARGET,
                    "mean",
                ),

                from_club_median_sale_fee=(
                    TARGET,
                    "median",
                ),

                from_club_max_sale_fee=(
                    TARGET,
                    "max",
                ),

                from_club_30m_plus_ratio=(
                    "is_30m_plus",
                    "mean",
                ),
            )
        )


        # ====================================================
        # MAP BUYER
        # ====================================================

        to_ids = current[
            "to_team_id"
        ]

        df.loc[
            current_mask,
            "to_club_hist_count",
        ] = (
            to_ids
            .map(
                buyer_stats[
                    "to_club_hist_count"
                ]
            )
            .fillna(0)
            .to_numpy()
        )

        df.loc[
            current_mask,
            "to_club_avg_buy_fee_m",
        ] = (
            to_ids
            .map(
                buyer_stats[
                    "to_club_avg_buy_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "to_club_median_buy_fee_m",
        ] = (
            to_ids
            .map(
                buyer_stats[
                    "to_club_median_buy_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "to_club_max_buy_fee_m",
        ] = (
            to_ids
            .map(
                buyer_stats[
                    "to_club_max_buy_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "to_club_30m_plus_ratio",
        ] = (
            to_ids
            .map(
                buyer_stats[
                    "to_club_30m_plus_ratio"
                ]
            )
            .fillna(0)
            .to_numpy()
        )


        # ====================================================
        # MAP SELLER
        # ====================================================

        from_ids = current[
            "from_team_id"
        ]

        df.loc[
            current_mask,
            "from_club_hist_count",
        ] = (
            from_ids
            .map(
                seller_stats[
                    "from_club_hist_count"
                ]
            )
            .fillna(0)
            .to_numpy()
        )

        df.loc[
            current_mask,
            "from_club_avg_sale_fee_m",
        ] = (
            from_ids
            .map(
                seller_stats[
                    "from_club_avg_sale_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "from_club_median_sale_fee_m",
        ] = (
            from_ids
            .map(
                seller_stats[
                    "from_club_median_sale_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "from_club_max_sale_fee_m",
        ] = (
            from_ids
            .map(
                seller_stats[
                    "from_club_max_sale_fee"
                ]
            )
            .fillna(0)
            .to_numpy()
            / 1_000_000
        )

        df.loc[
            current_mask,
            "from_club_30m_plus_ratio",
        ] = (
            from_ids
            .map(
                seller_stats[
                    "from_club_30m_plus_ratio"
                ]
            )
            .fillna(0)
            .to_numpy()
        )

    return df


# ============================================================
# SAMPLE WEIGHT
# ============================================================

def make_weight(y):

    (
        weight_under_30,
        weight_30_50,
        weight_50_plus,
    ) = SAMPLE_WEIGHT_CONFIG

    y = np.asarray(
        y,
        dtype=float,
    )

    weights = np.full(
        len(y),
        weight_under_30,
        dtype=float,
    )

    weights[
        (
            (y >= 30_000_000)
            & (y < 50_000_000)
        )
    ] = weight_30_50

    weights[
        y >= 50_000_000
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
                if column
                not in remove_set
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
# ADD NUMERIC FEATURES
# ============================================================

def add_numeric_features_to_model(
    model,
    features_to_add,
):

    if not features_to_add:
        return model

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
    extra_features,
):

    model = remove_features_from_model(
        base_model,
        removed_features,
    )

    numeric_features = (
        EUROPEAN_FEATURES
        + extra_features
    )

    model = add_numeric_features_to_model(
        model,
        numeric_features,
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
    extra_features,
):

    features = [
        feature
        for feature in base_features
        if feature
        not in removed_features
    ]

    features += EUROPEAN_FEATURES
    features += extra_features

    model = build_model(
        base_model=base_model,
        removed_features=removed_features,
        extra_features=extra_features,
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

    # target = log1p 유지
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
# PREDICT
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
            df[features_c]
        )
    )

    pred_d_log = (
        model_d.predict(
            df[features_d]
        )
    )

    pred_c = np.maximum(
        np.expm1(
            pred_c_log
        ),
        0,
    )

    pred_d = np.maximum(
        np.expm1(
            pred_d_log
        ),
        0,
    )

    prediction = (
        ALPHA_C * pred_c
        + ALPHA_D * pred_d
    )

    prediction = np.maximum(
        prediction,
        0,
    )

    return prediction


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

    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            prediction,
        )

    else:
        r2 = np.nan

    error_m = (
        prediction
        - y_true
    ) / 1_000_000

    return {
        "count": len(
            y_true
        ),

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
# EVALUATION
# ============================================================

def evaluate(
    experiment_name,
    y_true,
    prediction,
):

    q90 = np.quantile(
        y_true,
        0.90,
    )

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

    rows = []

    for (
        range_name,
        mask,
    ) in masks.items():

        metrics = calculate_metrics(
            y_true[mask],
            prediction[mask],
        )

        metrics[
            "experiment"
        ] = experiment_name

        metrics[
            "range"
        ] = range_name

        rows.append(
            metrics
        )

    return rows


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
# TRAIN + PREDICT HELPER
# ============================================================

def run_experiment(
    experiment_name,
    extra_features,
    base_model,
    base_features,
    train_df,
    eval_df,
):

    (
        model_c,
        features_c,
    ) = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=C_REMOVE_FEATURES,
        extra_features=extra_features,
    )

    (
        model_d,
        features_d,
    ) = train_variant(
        base_model=base_model,
        train_df=train_df,
        base_features=base_features,
        removed_features=D_REMOVE_FEATURES,
        extra_features=extra_features,
    )

    prediction = predict_ensemble(
        model_c=model_c,
        model_d=model_d,
        features_c=features_c,
        features_d=features_d,
        df=eval_df,
    )

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "V1.4 CANDIDATE - CLUB TRANSFER PROFILE"
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
        f"\nDataset    : {DATA_PATH}"
    )

    print(
        f"Base Model : {BASE_MODEL_PATH}"
    )

    print(
        f"Base features: {len(base_features)}"
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
    # NUMERIC
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


    # ========================================================
    # BASE FEATURES
    # ========================================================

    df = add_v11_features(
        df
    )


    # ========================================================
    # EUROPE
    # ========================================================

    for feature in EUROPEAN_FEATURES:

        df[feature] = (
            pd.to_numeric(
                df[feature],
                errors="coerce",
            )
            .fillna(0)
        )


    # ========================================================
    # CLUB PROFILE
    #
    # 이 함수 자체가 year < current year 만 사용
    # ========================================================

    print(
        "\nBuilding leakage-safe club profiles..."
    )

    df = add_club_profile_features(
        df
    )


    # ========================================================
    # COVERAGE
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CLUB PROFILE COVERAGE"
    )

    print(
        "=" * 100
    )

    for year in [
        2024,
        2025,
    ]:

        subset = df[
            df["year"] == year
        ]

        buyer_coverage = (
            subset[
                "to_club_hist_count"
            ] > 0
        ).mean()

        seller_coverage = (
            subset[
                "from_club_hist_count"
            ] > 0
        ).mean()

        print(
            f"{year}"
            f" | rows={len(subset)}"
            f" | buyer history="
            f"{buyer_coverage:.1%}"
            f" | seller history="
            f"{seller_coverage:.1%}"
        )


    # ========================================================
    # PHASE 1
    #
    # Train 2020~2023
    # Validate 2024
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
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(valid_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )


    validation_rows = []

    valid_predictions = valid_df[
        [
            "player_name",
            "transfer_date",
            "from_team_name",
            "to_team_name",
            TARGET,
        ]
    ].copy()

    valid_predictions[
        "actual_fee_m"
    ] = (
        valid_predictions[TARGET]
        / 1_000_000
    )


    # ========================================================
    # VALIDATION EXPERIMENTS
    # ========================================================

    for (
        experiment_name,
        extra_features,
    ) in EXPERIMENTS.items():

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"2024 VALIDATION: "
            f"{experiment_name}"
        )

        print(
            "=" * 100
        )

        prediction = run_experiment(
            experiment_name=experiment_name,
            extra_features=extra_features,
            base_model=base_model,
            base_features=base_features,
            train_df=train_df,
            eval_df=valid_df,
        )

        valid_predictions[
            f"{experiment_name}_pred_m"
        ] = (
            prediction
            / 1_000_000
        )

        y_valid = (
            valid_df[TARGET]
            .astype(float)
            .to_numpy()
        )

        rows = evaluate(
            experiment_name,
            y_valid,
            prediction,
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

    valid_pivot = (
        validation_summary
        .pivot(
            index="experiment",
            columns="range",
            values="mae_m",
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "2024 VALIDATION MAE COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        valid_pivot
        .round(3)
        .to_string()
    )


    # ========================================================
    # SELECT
    #
    # 전체 MAE 기준
    # ========================================================

    all_rows = (
        validation_summary[
            validation_summary[
                "range"
            ] == "ALL"
        ]
        .sort_values(
            "mae_m"
        )
    )

    best_experiment = (
        all_rows
        .iloc[0][
            "experiment"
        ]
    )

    best_extra_features = (
        EXPERIMENTS[
            best_experiment
        ]
    )

    print(
        "\n"
        + "#" * 100
    )

    print(
        "# MODEL SELECTION"
    )

    print(
        "#" * 100
    )

    print(
        f"\nSelected: "
        f"{best_experiment}"
    )

    print(
        f"Validation ALL MAE: "
        f"{all_rows.iloc[0]['mae_m']:.3f}M"
    )


    # ========================================================
    # PHASE 2
    #
    # 2020~2024 retrain
    # ========================================================

    final_train_df = df[
        df["year"] <= 2024
    ].copy()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL RETRAIN 2020~2024"
    )

    print(
        "=" * 100
    )

    print(
        f"Rows: "
        f"{len(final_train_df):,}"
    )


    test_prediction = run_experiment(
        experiment_name=best_experiment,
        extra_features=best_extra_features,
        base_model=base_model,
        base_features=base_features,
        train_df=final_train_df,
        eval_df=test_df,
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    y_test = (
        test_df[TARGET]
        .astype(float)
        .to_numpy()
    )

    test_rows = evaluate(
        best_experiment,
        y_test,
        test_prediction,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "2025 FINAL TEST"
    )

    print(
        "=" * 100
    )

    print(
        f"Selected Experiment: "
        f"{best_experiment}"
    )

    print()

    for row in test_rows:

        print_result(
            row["range"],
            row,
        )


    # ========================================================
    # TEST PLAYER OUTPUT
    # ========================================================

    test_output = test_df[
        [
            "player_name",
            "transfer_date",

            "from_team_name",
            "to_team_name",

            "main_position",

            TARGET,

            *CLUB_PROFILE_FEATURES,
        ]
    ].copy()

    test_output[
        "actual_fee_m"
    ] = (
        y_test
        / 1_000_000
    )

    test_output[
        "predicted_fee_m"
    ] = (
        test_prediction
        / 1_000_000
    )

    test_output[
        "error_m"
    ] = (
        (
            test_prediction
            - y_test
        )
        / 1_000_000
    )

    test_output[
        "abs_error_m"
    ] = (
        test_output[
            "error_m"
        ]
        .abs()
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "ERROR TOP 20"
    )

    print(
        "=" * 120
    )

    print(
        test_output
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

    validation_path = (
        RESULT_DIR
        / "club_profile_2024_validation.csv"
    )

    valid_prediction_path = (
        RESULT_DIR
        / "club_profile_2024_predictions.csv"
    )

    test_path = (
        RESULT_DIR
        / "club_profile_2025_final_test.csv"
    )

    test_prediction_path = (
        RESULT_DIR
        / "club_profile_2025_predictions.csv"
    )


    validation_summary.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    valid_predictions.to_csv(
        valid_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        test_rows
    ).to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_output.to_csv(
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
        validation_path
    )

    print(
        valid_prediction_path
    )

    print(
        test_path
    )

    print(
        test_prediction_path
    )


if __name__ == "__main__":
    main()