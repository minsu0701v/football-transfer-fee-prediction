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

DATA_FILE = "data/processed/training_dataset.csv"
BASE_MODEL_FILE = "models/transfer_fee_model_v1_1.joblib"

TARGET = "transfer_fee"

OUTPUT_DIR = Path("results")

TOP5_LEAGUES = [
    "GB1",
    "ES1",
    "IT1",
    "L1",
    "FR1",
]

# C 모델
# is_same_league 제거
C_REMOVE_FEATURES = [
    "is_same_league",
]

# D 모델
# is_same_league + to_league_id 제거
D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]

# C 비중
# 0.0 = D 100%
# 1.0 = C 100%
ALPHAS = np.arange(
    0.0,
    1.01,
    0.1,
)


# ============================================================
# Weight C
# ============================================================

def make_weight_c(y):

    y = np.asarray(y)

    weights = np.ones(
        len(y),
        dtype=float,
    )

    # 30~50M
    weights[
        (y >= 30_000_000)
        & (y < 50_000_000)
    ] = 2.5

    # 50M+
    weights[
        y >= 50_000_000
    ] = 4.0

    return weights


# ============================================================
# v1.1 파생변수
# ============================================================

def add_v11_features(df):

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
# 모델 복제
# ============================================================

def clone_model(model):

    try:
        return clone(model)

    except Exception:
        return copy.deepcopy(model)


# ============================================================
# ColumnTransformer 찾기
# ============================================================

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
            return step_name, step

    raise ValueError(
        "ColumnTransformer를 찾지 못했습니다."
    )


# ============================================================
# 최종 estimator 이름
# ============================================================

def get_estimator_step_name(model):

    if not hasattr(model, "steps"):
        raise ValueError(
            "Pipeline 형태가 아닙니다."
        )

    return model.steps[-1][0]


# ============================================================
# Pipeline에서 feature 제거
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
# 모델 학습
# ============================================================

def train_model(
    base_model,
    train_df,
    base_features,
    remove_features,
):

    features = [
        feature
        for feature in base_features
        if feature not in remove_features
    ]

    model = remove_features_from_model(
        base_model,
        remove_features,
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        TARGET
    ]

    sample_weight = make_weight_c(
        y_train
    )

    estimator_step = (
        get_estimator_step_name(
            model
        )
    )

    fit_params = {
        f"{estimator_step}__sample_weight":
            sample_weight
    }

    model.fit(
        X_train,
        np.log1p(y_train),
        **fit_params,
    )

    return model, features


# ============================================================
# 예측
# ============================================================

def predict_fee(
    model,
    df,
    features,
):

    X = df[
        features
    ]

    prediction_log = (
        model.predict(X)
    )

    prediction = np.expm1(
        prediction_log
    )

    prediction = np.maximum(
        prediction,
        0,
    )

    return prediction


# ============================================================
# 기본 성능 평가
# ============================================================

def evaluate_predictions(
    df,
    predictions,
):

    y_true = df[
        TARGET
    ].values

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

    r2 = r2_score(
        y_true,
        predictions,
    )

    # Top 10%
    q90 = df[
        TARGET
    ].quantile(0.90)

    top10_mask = (
        df[TARGET]
        >= q90
    )

    top10_mae = (
        mean_absolute_error(
            df.loc[
                top10_mask,
                TARGET,
            ],
            predictions[
                top10_mask.values
            ],
        )
        / 1_000_000
    )

    # 50M+
    high_mask = (
        df[TARGET]
        >= 50_000_000
    )

    high_actual = (
        df.loc[
            high_mask,
            TARGET,
        ]
        / 1_000_000
    )

    high_pred = (
        predictions[
            high_mask.values
        ]
        / 1_000_000
    )

    if len(high_actual) > 0:

        high_actual_mean = (
            high_actual.mean()
        )

        high_pred_mean = (
            high_pred.mean()
        )

        high_mae = (
            np.abs(
                high_actual.values
                - high_pred
            ).mean()
        )

    else:

        high_actual_mean = np.nan
        high_pred_mean = np.nan
        high_mae = np.nan

    return {
        "MAE_M": mae,
        "RMSE_M": rmse,
        "R2": r2,
        "Top10_MAE_M": top10_mae,
        "50M+_Actual_Mean_M": (
            high_actual_mean
        ),
        "50M+_Pred_Mean_M": (
            high_pred_mean
        ),
        "50M+_MAE_M": high_mae,
    }


# ============================================================
# 목적 리그 변경
# ============================================================

def make_destination_scenario(
    df,
    league_id,
):

    result = df.copy()

    result[
        "to_league_id"
    ] = league_id

    # 이번 C/D 모델에서는
    # is_same_league를 둘 다 제거했으므로
    # 실제 모델 입력에는 쓰이지 않음.
    #
    # 그래도 데이터 일관성을 위해 갱신.

    result[
        "is_same_league"
    ] = (
        result[
            "from_league_id"
        ].astype(str)
        == league_id
    ).astype(int)

    result[
        "is_top5_destination"
    ] = 1

    return result


# ============================================================
# GB1 / ES1 counterfactual
# ============================================================

def counterfactual_predictions(
    model,
    df,
    features,
):

    gb1_df = make_destination_scenario(
        df,
        "GB1",
    )

    es1_df = make_destination_scenario(
        df,
        "ES1",
    )

    gb1_pred = predict_fee(
        model,
        gb1_df,
        features,
    )

    es1_pred = predict_fee(
        model,
        es1_df,
        features,
    )

    return (
        gb1_pred,
        es1_pred,
    )


# ============================================================
# Ensemble
# ============================================================

def ensemble_predictions(
    c_pred,
    d_pred,
    alpha,
):

    return (
        alpha * c_pred
        + (1 - alpha) * d_pred
    )


# ============================================================
# Counterfactual premium
# ============================================================

def calculate_cf_premium(
    gb1_pred,
    es1_pred,
):

    gb1_mean = (
        gb1_pred.mean()
        / 1_000_000
    )

    es1_mean = (
        es1_pred.mean()
        / 1_000_000
    )

    difference = (
        gb1_mean
        - es1_mean
    )

    if es1_mean > 0:

        premium = (
            (
                gb1_mean
                / es1_mean
            )
            - 1
        ) * 100

    else:

        premium = np.nan

    return (
        gb1_mean,
        es1_mean,
        difference,
        premium,
    )


# ============================================================
# Alpha 실험
# ============================================================

def evaluate_alphas(
    df,
    c_pred,
    d_pred,
    c_gb1,
    c_es1,
    d_gb1,
    d_es1,
):

    results = []

    for alpha in ALPHAS:

        # 일반 예측
        ensemble_pred = (
            ensemble_predictions(
                c_pred,
                d_pred,
                alpha,
            )
        )

        metrics = (
            evaluate_predictions(
                df,
                ensemble_pred,
            )
        )

        # Counterfactual
        ensemble_gb1 = (
            ensemble_predictions(
                c_gb1,
                d_gb1,
                alpha,
            )
        )

        ensemble_es1 = (
            ensemble_predictions(
                c_es1,
                d_es1,
                alpha,
            )
        )

        (
            gb1_mean,
            es1_mean,
            diff,
            premium,
        ) = calculate_cf_premium(
            ensemble_gb1,
            ensemble_es1,
        )

        row = {
            "alpha_C": alpha,
            "weight_C_model_pct": (
                alpha * 100
            ),
            "weight_D_model_pct": (
                (1 - alpha) * 100
            ),
            **metrics,
            "GB1_CF_Mean_M": (
                gb1_mean
            ),
            "ES1_CF_Mean_M": (
                es1_mean
            ),
            "GB1_minus_ES1_M": (
                diff
            ),
            "GB1_vs_ES1_Pct": (
                premium
            ),
        }

        results.append(
            row
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 데이터 로드
    # ========================================================

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
            "no-market 모델이 아닙니다."
        )

    df[
        "transfer_date"
    ] = pd.to_datetime(
        df[
            "transfer_date"
        ],
        errors="coerce",
    )

    df = add_v11_features(
        df
    )

    df[
        "transfer_year"
    ] = (
        df[
            "transfer_date"
        ].dt.year
    )

    print()
    print(
        "=" * 80
    )
    print(
        "v1.2 C + D Ensemble Validation"
    )
    print(
        "=" * 80
    )

    print()
    print(
        "C = Weight C + "
        "is_same_league 제거"
    )

    print(
        "D = Weight C + "
        "is_same_league/to_league_id 제거"
    )


    # ========================================================
    # 1단계
    # 2020~2023 Train
    # 2024 Validation
    # ========================================================

    train_2023 = df[
        df[
            "transfer_year"
        ] <= 2023
    ].copy()

    validation_2024 = df[
        df[
            "transfer_year"
        ] == 2024
    ].copy()

    print()
    print(
        "=" * 80
    )
    print(
        "1단계: Validation"
    )
    print(
        "=" * 80
    )

    print(
        f"Train 2020~2023: "
        f"{len(train_2023)}"
    )

    print(
        f"Validation 2024 : "
        f"{len(validation_2024)}"
    )


    # ========================================================
    # C 학습
    # ========================================================

    c_model_val, c_features = (
        train_model(
            base_model,
            train_2023,
            base_features,
            C_REMOVE_FEATURES,
        )
    )

    # ========================================================
    # D 학습
    # ========================================================

    d_model_val, d_features = (
        train_model(
            base_model,
            train_2023,
            base_features,
            D_REMOVE_FEATURES,
        )
    )


    # ========================================================
    # Validation 일반 예측
    # ========================================================

    c_val_pred = predict_fee(
        c_model_val,
        validation_2024,
        c_features,
    )

    d_val_pred = predict_fee(
        d_model_val,
        validation_2024,
        d_features,
    )


    # ========================================================
    # Validation Counterfactual
    # ========================================================

    (
        c_val_gb1,
        c_val_es1,
    ) = counterfactual_predictions(
        c_model_val,
        validation_2024,
        c_features,
    )

    (
        d_val_gb1,
        d_val_es1,
    ) = counterfactual_predictions(
        d_model_val,
        validation_2024,
        d_features,
    )


    # ========================================================
    # Alpha 탐색
    # ========================================================

    validation_result = (
        evaluate_alphas(
            validation_2024,
            c_val_pred,
            d_val_pred,
            c_val_gb1,
            c_val_es1,
            d_val_gb1,
            d_val_es1,
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "★ 2024 Validation Alpha 결과"
    )
    print(
        "=" * 120
    )

    display_columns = [
        "alpha_C",
        "MAE_M",
        "RMSE_M",
        "R2",
        "Top10_MAE_M",
        "50M+_Pred_Mean_M",
        "50M+_MAE_M",
        "GB1_CF_Mean_M",
        "ES1_CF_Mean_M",
        "GB1_vs_ES1_Pct",
    ]

    print(
        validation_result[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # Validation MAE 기준 최적 alpha
    # ========================================================

    best_alpha = 0.4

    best_row = validation_result[
        np.isclose(
            validation_result["alpha_C"],
            best_alpha,
        )
    ].iloc[0]

    print()
    print(
        "=" * 80
    )
    print(
        "★ Validation MAE 기준 최적 Alpha"
    )
    print(
        "=" * 80
    )

    print(
        "C 비중 : "
        f"{best_alpha * 100:.0f}%"
    )

    print(
        "D 비중 : "
        f"{(1 - best_alpha) * 100:.0f}%"
    )

    print(
        "Validation MAE : "
        f"{best_row['MAE_M']:.3f}M"
    )

    print(
        "Validation R²  : "
        f"{best_row['R2']:.4f}"
    )

    print(
        "GB1 vs ES1     : "
        f"{best_row['GB1_vs_ES1_Pct']:.2f}%"
    )


    # ========================================================
    # 2단계
    # 2020~2024 Train
    # 2025 Test
    # ========================================================

    train_2024 = df[
        df[
            "transfer_year"
        ] <= 2024
    ].copy()

    test_2025 = df[
        df[
            "transfer_year"
        ] == 2025
    ].copy()

    print()
    print(
        "=" * 80
    )
    print(
        "2단계: Final Test"
    )
    print(
        "=" * 80
    )

    print(
        f"Train 2020~2024: "
        f"{len(train_2024)}"
    )

    print(
        f"Test 2025       : "
        f"{len(test_2025)}"
    )

    print(
        "사용 Alpha       : "
        f"{best_alpha:.1f}"
    )


    # ========================================================
    # C 최종 학습
    # ========================================================

    c_model_test, c_features_test = (
        train_model(
            base_model,
            train_2024,
            base_features,
            C_REMOVE_FEATURES,
        )
    )

    # ========================================================
    # D 최종 학습
    # ========================================================

    d_model_test, d_features_test = (
        train_model(
            base_model,
            train_2024,
            base_features,
            D_REMOVE_FEATURES,
        )
    )


    # ========================================================
    # Test 예측
    # ========================================================

    c_test_pred = predict_fee(
        c_model_test,
        test_2025,
        c_features_test,
    )

    d_test_pred = predict_fee(
        d_model_test,
        test_2025,
        d_features_test,
    )

    ensemble_test_pred = (
        ensemble_predictions(
            c_test_pred,
            d_test_pred,
            best_alpha,
        )
    )


    # ========================================================
    # Test Counterfactual
    # ========================================================

    (
        c_test_gb1,
        c_test_es1,
    ) = counterfactual_predictions(
        c_model_test,
        test_2025,
        c_features_test,
    )

    (
        d_test_gb1,
        d_test_es1,
    ) = counterfactual_predictions(
        d_model_test,
        test_2025,
        d_features_test,
    )

    ensemble_test_gb1 = (
        ensemble_predictions(
            c_test_gb1,
            d_test_gb1,
            best_alpha,
        )
    )

    ensemble_test_es1 = (
        ensemble_predictions(
            c_test_es1,
            d_test_es1,
            best_alpha,
        )
    )


    # ========================================================
    # C / D / Ensemble 각각 평가
    # ========================================================

    c_metrics = evaluate_predictions(
        test_2025,
        c_test_pred,
    )

    d_metrics = evaluate_predictions(
        test_2025,
        d_test_pred,
    )

    ensemble_metrics = (
        evaluate_predictions(
            test_2025,
            ensemble_test_pred,
        )
    )


    # ========================================================
    # CF premium
    # ========================================================

    (
        c_gb1_mean,
        c_es1_mean,
        c_diff,
        c_premium,
    ) = calculate_cf_premium(
        c_test_gb1,
        c_test_es1,
    )

    (
        d_gb1_mean,
        d_es1_mean,
        d_diff,
        d_premium,
    ) = calculate_cf_premium(
        d_test_gb1,
        d_test_es1,
    )

    (
        e_gb1_mean,
        e_es1_mean,
        e_diff,
        e_premium,
    ) = calculate_cf_premium(
        ensemble_test_gb1,
        ensemble_test_es1,
    )


    # ========================================================
    # 최종 비교표
    # ========================================================

    final_rows = []

    for (
        model_name,
        metrics,
        gb1_mean,
        es1_mean,
        premium,
    ) in [

        (
            "C_ONLY",
            c_metrics,
            c_gb1_mean,
            c_es1_mean,
            c_premium,
        ),

        (
            "D_ONLY",
            d_metrics,
            d_gb1_mean,
            d_es1_mean,
            d_premium,
        ),

        (
            f"ENSEMBLE_C_{best_alpha:.1f}",
            ensemble_metrics,
            e_gb1_mean,
            e_es1_mean,
            e_premium,
        ),

    ]:

        row = {
            "model": model_name,
            **metrics,
            "GB1_CF_Mean_M": (
                gb1_mean
            ),
            "ES1_CF_Mean_M": (
                es1_mean
            ),
            "GB1_vs_ES1_Pct": (
                premium
            ),
        }

        final_rows.append(
            row
        )

    final_result = pd.DataFrame(
        final_rows
    )

    print()
    print(
        "=" * 120
    )
    print(
        "★ 2025 FINAL TEST"
    )
    print(
        "=" * 120
    )

    print(
        final_result[
            [
                "model",
                "MAE_M",
                "RMSE_M",
                "R2",
                "Top10_MAE_M",
                "50M+_Actual_Mean_M",
                "50M+_Pred_Mean_M",
                "50M+_MAE_M",
                "GB1_CF_Mean_M",
                "ES1_CF_Mean_M",
                "GB1_vs_ES1_Pct",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 저장
    # ========================================================

    validation_file = (
        OUTPUT_DIR
        / "v12_cd_ensemble_validation.csv"
    )

    final_file = (
        OUTPUT_DIR
        / "v12_cd_ensemble_2025_test.csv"
    )

    validation_result.to_csv(
        validation_file,
        index=False,
        encoding="utf-8-sig",
    )

    final_result.to_csv(
        final_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "저장 완료"
    )
    print(
        "=" * 80
    )

    print(
        validation_file
    )

    print(
        final_file
    )


if __name__ == "__main__":
    main()