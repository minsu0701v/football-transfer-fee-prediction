from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 같은 scripts 폴더의 기존 실험 코드 재사용
from evaluate_club_profile_features import (
    DATA_PATH,
    BASE_MODEL_PATH,
    RESULT_DIR,
    TARGET,
    EUROPEAN_FEATURES,
    BUYER_PROFILE_FEATURES,
    add_v11_features,
    add_club_profile_features,
    run_experiment,
)


# ============================================================
# BLEND CONFIG
#
# final_prediction
# =
# (1 - beta) * BASE
# + beta * BUYER
#
# beta = 0.0 -> 기존 v1.3
# beta = 1.0 -> BUYER_ALL 100%
# ============================================================

BETAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# ROLLING FOLDS
# ============================================================

FOLDS = [
    {
        "valid_year": 2022,
        "train_end": 2021,
    },
    {
        "valid_year": 2023,
        "train_end": 2022,
    },
    {
        "valid_year": 2024,
        "train_end": 2023,
    },
]


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
        "count": len(y_true),

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
# EVALUATE RANGE
# ============================================================

def evaluate_prediction(
    beta,
    y_true,
    prediction,
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
            y_true
            >= q90
        ),
    }

    rows = []

    for (
        range_name,
        mask,
    ) in masks.items():

        actual = y_true[
            mask
        ]

        pred = prediction[
            mask
        ]

        if len(actual) == 0:
            continue

        metrics = calculate_metrics(
            actual,
            pred,
        )

        metrics["beta"] = beta
        metrics["range"] = range_name

        rows.append(
            metrics
        )

    return rows


# ============================================================
# PRINT
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
# BLEND
# ============================================================

def blend_predictions(
    base_prediction,
    buyer_prediction,
    beta,
):

    prediction = (
        (1.0 - beta)
        * base_prediction

        + beta
        * buyer_prediction
    )

    return np.maximum(
        prediction,
        0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "V1.4 BASE + BUYER BLEND"
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
    # BASE FEATURES
    # ========================================================

    df = add_v11_features(
        df
    )


    # ========================================================
    # EUROPE FEATURES
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
    # BUYER PROFILE
    # Leakage-safe:
    # 해당 연도보다 과거 거래만 사용
    # ========================================================

    print(
        "\nBuilding leakage-safe "
        "buyer club profiles..."
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
        "BUYER PROFILE COVERAGE"
    )

    print(
        "=" * 100
    )

    for year in [
        2022,
        2023,
        2024,
        2025,
    ]:

        subset = df[
            df["year"] == year
        ]

        coverage = (
            subset[
                "to_club_hist_count"
            ] > 0
        ).mean()

        print(
            f"{year}"
            f" | rows={len(subset):,}"
            f" | coverage="
            f"{coverage:.1%}"
        )


    # ========================================================
    # ROLLING VALIDATION
    # ========================================================

    validation_rows = []

    fold_prediction_rows = []


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# PHASE 1 - ROLLING VALIDATION"
    )

    print(
        "#" * 100
    )


    for fold in FOLDS:

        train_end = (
            fold["train_end"]
        )

        valid_year = (
            fold["valid_year"]
        )

        train_df = df[
            df["year"] <= train_end
        ].copy()

        valid_df = df[
            df["year"] == valid_year
        ].copy()

        y_valid = (
            valid_df[TARGET]
            .astype(float)
            .to_numpy()
        )


        print(
            "\n\n"
            + "=" * 100
        )

        print(
            f"VALIDATION {valid_year}"
        )

        print(
            f"Train: 2020~{train_end}"
        )

        print(
            f"Train rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Valid rows: "
            f"{len(valid_df):,}"
        )

        print(
            "=" * 100
        )


        # ====================================================
        # BASE
        # ====================================================

        print(
            "\nTraining BASE..."
        )

        base_prediction = (
            run_experiment(
                experiment_name="BASE",

                extra_features=[],

                base_model=base_model,

                base_features=(
                    base_features
                ),

                train_df=train_df,

                eval_df=valid_df,
            )
        )


        # ====================================================
        # BUYER ALL
        # ====================================================

        print(
            "Training BUYER_ALL..."
        )

        buyer_prediction = (
            run_experiment(
                experiment_name=(
                    "BUYER_ALL"
                ),

                extra_features=(
                    BUYER_PROFILE_FEATURES
                ),

                base_model=base_model,

                base_features=(
                    base_features
                ),

                train_df=train_df,

                eval_df=valid_df,
            )
        )


        # ====================================================
        # SAVE FOLD PLAYER PREDICTIONS
        # ====================================================

        fold_output = (
            valid_df[
                [
                    "player_name",
                    "transfer_date",
                    "from_team_name",
                    "to_team_name",
                    TARGET,
                ]
            ]
            .copy()
        )

        fold_output[
            "valid_year"
        ] = valid_year

        fold_output[
            "actual_fee_m"
        ] = (
            y_valid
            / 1_000_000
        )

        fold_output[
            "base_pred_m"
        ] = (
            base_prediction
            / 1_000_000
        )

        fold_output[
            "buyer_pred_m"
        ] = (
            buyer_prediction
            / 1_000_000
        )


        # ====================================================
        # EACH BETA
        # ====================================================

        for beta in BETAS:

            prediction = (
                blend_predictions(
                    base_prediction,
                    buyer_prediction,
                    beta,
                )
            )


            fold_output[
                f"beta_{beta:.2f}_pred_m"
            ] = (
                prediction
                / 1_000_000
            )


            rows = (
                evaluate_prediction(
                    beta,
                    y_valid,
                    prediction,
                )
            )


            print(
                "\n"
                + "-" * 100
            )

            print(
                f"BETA = {beta:.2f}"
            )

            print(
                "-" * 100
            )


            for row in rows:

                row[
                    "valid_year"
                ] = valid_year

                row[
                    "train_end"
                ] = train_end

                validation_rows.append(
                    row
                )

                print_result(
                    row["range"],
                    row,
                )


        fold_prediction_rows.append(
            fold_output
        )


    # ========================================================
    # VALIDATION DATAFRAME
    # ========================================================

    validation_df = pd.DataFrame(
        validation_rows
    )


    # ========================================================
    # YEAR x BETA
    # ALL MAE
    # ========================================================

    all_rows = validation_df[
        validation_df["range"]
        == "ALL"
    ].copy()

    year_pivot = (
        all_rows
        .pivot(
            index="beta",
            columns="valid_year",
            values="mae_m",
        )
    )


    print(
        "\n\n"
        + "=" * 100
    )

    print(
        "ALL MAE BY VALIDATION YEAR"
    )

    print(
        "=" * 100
    )

    print(
        year_pivot
        .round(3)
        .to_string()
    )


    # ========================================================
    # ROLLING SUMMARY
    # ========================================================

    summary_rows = []


    for beta in BETAS:

        beta_df = validation_df[
            validation_df[
                "beta"
            ] == beta
        ]


        def values(
            range_name,
            column,
        ):

            return beta_df[
                beta_df["range"]
                == range_name
            ][column]


        all_mae = values(
            "ALL",
            "mae_m",
        )

        mae_30 = values(
            "30M+",
            "mae_m",
        )

        mae_50 = values(
            "50M+",
            "mae_m",
        )

        mae_70 = values(
            "70M+",
            "mae_m",
        )

        top10 = values(
            "TOP10%",
            "mae_m",
        )

        bias_30 = values(
            "30M+",
            "mean_error_m",
        )

        bias_50 = values(
            "50M+",
            "mean_error_m",
        )

        bias_70 = values(
            "70M+",
            "mean_error_m",
        )


        summary_rows.append(
            {
                "beta": beta,

                "mean_all_mae":
                    all_mae.mean(),

                "std_all_mae":
                    all_mae.std(
                        ddof=0
                    ),

                "worst_all_mae":
                    all_mae.max(),

                "mean_30m_mae":
                    mae_30.mean(),

                "mean_50m_mae":
                    mae_50.mean(),

                "mean_70m_mae":
                    mae_70.mean(),

                "mean_top10_mae":
                    top10.mean(),

                "mean_30m_bias":
                    bias_30.mean(),

                "mean_50m_bias":
                    bias_50.mean(),

                "mean_70m_bias":
                    bias_70.mean(),
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # SELECT
    #
    # 1. Mean ALL MAE
    # 2. Std ALL MAE
    #
    # 2025는 사용하지 않음
    # ========================================================

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "mean_all_mae",
                "std_all_mae",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "ROLLING BLEND SUMMARY"
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
    # BASE 대비
    # ========================================================

    baseline = (
        summary_df[
            summary_df["beta"]
            == 0.0
        ]
        .iloc[0]
    )

    delta_rows = []


    for _, row in (
        summary_df.iterrows()
    ):

        delta_rows.append(
            {
                "beta":
                    row["beta"],

                "ALL_delta":
                    (
                        row[
                            "mean_all_mae"
                        ]
                        - baseline[
                            "mean_all_mae"
                        ]
                    ),

                "30M_delta":
                    (
                        row[
                            "mean_30m_mae"
                        ]
                        - baseline[
                            "mean_30m_mae"
                        ]
                    ),

                "50M_delta":
                    (
                        row[
                            "mean_50m_mae"
                        ]
                        - baseline[
                            "mean_50m_mae"
                        ]
                    ),

                "70M_delta":
                    (
                        row[
                            "mean_70m_mae"
                        ]
                        - baseline[
                            "mean_70m_mae"
                        ]
                    ),

                "TOP10_delta":
                    (
                        row[
                            "mean_top10_mae"
                        ]
                        - baseline[
                            "mean_top10_mae"
                        ]
                    ),
            }
        )


    delta_df = pd.DataFrame(
        delta_rows
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "CHANGE VS BASE"
    )

    print(
        "음수 = 개선"
    )

    print(
        "=" * 100
    )

    print(
        delta_df
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # BEST BETA
    # ========================================================

    best_row = (
        summary_df.iloc[0]
    )

    best_beta = float(
        best_row["beta"]
    )


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# BLEND SELECTION COMPLETE"
    )

    print(
        "#" * 100
    )

    print(
        f"\nSelected Beta    : "
        f"{best_beta:.2f}"
    )

    print(
        f"BASE Weight      : "
        f"{1.0 - best_beta:.2f}"
    )

    print(
        f"BUYER Weight     : "
        f"{best_beta:.2f}"
    )

    print(
        f"Mean ALL MAE     : "
        f"{best_row['mean_all_mae']:.3f}M"
    )

    print(
        f"ALL MAE Std      : "
        f"{best_row['std_all_mae']:.3f}M"
    )

    print(
        f"Mean 30M+ MAE    : "
        f"{best_row['mean_30m_mae']:.3f}M"
    )

    print(
        f"Mean 50M+ MAE    : "
        f"{best_row['mean_50m_mae']:.3f}M"
    )

    print(
        f"Mean 70M+ MAE    : "
        f"{best_row['mean_70m_mae']:.3f}M"
    )

    print(
        f"Mean TOP10 MAE   : "
        f"{best_row['mean_top10_mae']:.3f}M"
    )


    # ========================================================
    # PHASE 2
    #
    # 2020~2024 RETRAIN
    # ========================================================

    final_train_df = df[
        df["year"] <= 2024
    ].copy()

    test_df = df[
        df["year"] == 2025
    ].copy()


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        "# PHASE 2 - FINAL RETRAIN"
    )

    print(
        "#" * 100
    )

    print(
        f"\nTrain 2020~2024: "
        f"{len(final_train_df):,}"
    )

    print(
        f"Test 2025      : "
        f"{len(test_df):,}"
    )


    # ========================================================
    # FINAL BASE
    # ========================================================

    print(
        "\nTraining Final BASE..."
    )

    final_base_prediction = (
        run_experiment(
            experiment_name="BASE",

            extra_features=[],

            base_model=base_model,

            base_features=(
                base_features
            ),

            train_df=(
                final_train_df
            ),

            eval_df=test_df,
        )
    )


    # ========================================================
    # FINAL BUYER
    # ========================================================

    print(
        "Training Final BUYER_ALL..."
    )

    final_buyer_prediction = (
        run_experiment(
            experiment_name=(
                "BUYER_ALL"
            ),

            extra_features=(
                BUYER_PROFILE_FEATURES
            ),

            base_model=base_model,

            base_features=(
                base_features
            ),

            train_df=(
                final_train_df
            ),

            eval_df=test_df,
        )
    )


    # ========================================================
    # SELECTED BLEND
    # ========================================================

    final_prediction = (
        blend_predictions(
            final_base_prediction,
            final_buyer_prediction,
            best_beta,
        )
    )


    y_test = (
        test_df[TARGET]
        .astype(float)
        .to_numpy()
    )


    final_rows = (
        evaluate_prediction(
            best_beta,
            y_test,
            final_prediction,
        )
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
        f"Selected Beta: "
        f"{best_beta:.2f}"
    )

    print(
        f"Final formula:"
    )

    print(
        f"{1.0 - best_beta:.2f}"
        f" * BASE"
        f" + "
        f"{best_beta:.2f}"
        f" * BUYER"
    )

    print()


    for row in final_rows:

        print_result(
            row["range"],
            row,
        )


    # ========================================================
    # PLAYER OUTPUT
    # ========================================================

    test_output = (
        test_df[
            [
                "player_name",
                "transfer_date",

                "from_team_name",
                "to_team_name",

                "main_position",

                TARGET,

                *BUYER_PROFILE_FEATURES,
            ]
        ]
        .copy()
    )


    test_output[
        "actual_fee_m"
    ] = (
        y_test
        / 1_000_000
    )

    test_output[
        "base_pred_m"
    ] = (
        final_base_prediction
        / 1_000_000
    )

    test_output[
        "buyer_pred_m"
    ] = (
        final_buyer_prediction
        / 1_000_000
    )

    test_output[
        "blend_pred_m"
    ] = (
        final_prediction
        / 1_000_000
    )

    test_output[
        "blend_error_m"
    ] = (
        (
            final_prediction
            - y_test
        )
        / 1_000_000
    )

    test_output[
        "abs_error_m"
    ] = (
        test_output[
            "blend_error_m"
        ]
        .abs()
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL ERROR TOP 20"
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
    # SPECIFIC PLAYERS
    # ========================================================

    watch_players = [
        "Alexander Isak",
        "Nick Woltemade",
        "Martín Zubimendi",
        "Tom Bischof",
        "Trent Alexander-Arnold",
        "Jonathan Tah",
        "Rayan Cherki",
        "Eberechi Eze",
    ]


    watch_df = (
        test_output[
            test_output[
                "player_name"
            ].isin(
                watch_players
            )
        ][
            [
                "player_name",
                "actual_fee_m",
                "base_pred_m",
                "buyer_pred_m",
                "blend_pred_m",
                "blend_error_m",
            ]
        ]
        .copy()
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "WATCH PLAYERS"
    )

    print(
        "=" * 100
    )

    print(
        watch_df
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    rolling_path = (
        RESULT_DIR
        / "buyer_blend_rolling_validation.csv"
    )

    summary_path = (
        RESULT_DIR
        / "buyer_blend_rolling_summary.csv"
    )

    delta_path = (
        RESULT_DIR
        / "buyer_blend_vs_base.csv"
    )

    fold_predictions_path = (
        RESULT_DIR
        / "buyer_blend_fold_predictions.csv"
    )

    final_test_path = (
        RESULT_DIR
        / "buyer_blend_2025_final_test.csv"
    )

    final_predictions_path = (
        RESULT_DIR
        / "buyer_blend_2025_predictions.csv"
    )


    validation_df.to_csv(
        rolling_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        delta_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        fold_prediction_rows,
        ignore_index=True,
    ).to_csv(
        fold_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        final_rows
    ).to_csv(
        final_test_path,
        index=False,
        encoding="utf-8-sig",
    )

    test_output.to_csv(
        final_predictions_path,
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

    print(rolling_path)
    print(summary_path)
    print(delta_path)
    print(fold_predictions_path)
    print(final_test_path)
    print(final_predictions_path)


if __name__ == "__main__":
    main()