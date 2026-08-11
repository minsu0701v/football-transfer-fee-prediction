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
# 파일 경로
# ============================================================

DATA_FILE = "data/processed/training_dataset.csv"

BASE_MODEL_FILE = (
    "models/transfer_fee_model_v1_1.joblib"
)

TARGET = "transfer_fee"

OUTPUT_DIR = Path("results")


# ============================================================
# 5대 리그
# ============================================================

TOP5_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}


# ============================================================
# 비교할 모델
# ============================================================

MODEL_VARIANTS = {
    "A_FULL": [],
    "B_NO_TO_LEAGUE": [
        "to_league_id",
    ],
    "C_NO_SAME_LEAGUE": [
        "is_same_league",
    ],
    "D_NO_BOTH": [
        "to_league_id",
        "is_same_league",
    ],
}


# ============================================================
# Weight C
# ============================================================
#
# < 30M      -> 1.0
# 30~50M     -> 2.5
# 50M+       -> 4.0
#
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
    ] = 2.5

    weights[
        y >= 50_000_000
    ] = 4.0

    return weights


# ============================================================
# v1.1 Feature Engineering
# ============================================================

def add_v11_features(df):
    df = df.copy()

    # 90분당 득점
    df["goals_per90"] = np.where(
        df["minutes"] > 0,
        df["goals"] / df["minutes"] * 90,
        0,
    )

    # 90분당 도움
    df["assists_per90"] = np.where(
        df["minutes"] > 0,
        df["assists"] / df["minutes"] * 90,
        0,
    )

    # 90분당 공격포인트
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

    # 선발 비율
    df["starts_ratio"] = np.where(
        df["matches"] > 0,
        df["started"] / df["matches"],
        0,
    )

    # 경기당 출전시간
    df["minutes_per_match"] = np.where(
        df["matches"] > 0,
        df["minutes"] / df["matches"],
        0,
    )

    # 나이 제곱
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
# Pipeline 내부 ColumnTransformer 찾기
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
        "Pipeline에서 ColumnTransformer를 "
        "찾지 못했습니다."
    )


# ============================================================
# Pipeline 마지막 estimator 이름
# ============================================================

def get_estimator_step_name(model):
    if not hasattr(model, "steps"):
        raise ValueError(
            "모델이 sklearn Pipeline 형태가 아닙니다."
        )

    if len(model.steps) == 0:
        raise ValueError(
            "Pipeline에 step이 없습니다."
        )

    return model.steps[-1][0]


# ============================================================
# 특정 feature를 Pipeline에서 제거
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
# 예측
# ============================================================

def predict_fee(
    model,
    X,
):
    pred_log = model.predict(X)

    predictions = np.expm1(
        pred_log
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    return predictions


# ============================================================
# 모델 평가
# ============================================================

def evaluate_model(
    model_name,
    test_df,
    predictions,
):
    result_df = test_df.copy()

    result_df["predicted_fee"] = (
        predictions
    )

    result_df["actual_m"] = (
        result_df[TARGET]
        / 1_000_000
    )

    result_df["predicted_m"] = (
        result_df["predicted_fee"]
        / 1_000_000
    )

    result_df["absolute_error"] = (
        result_df[TARGET]
        - result_df["predicted_fee"]
    ).abs()

    y_true = result_df[
        TARGET
    ]

    # --------------------------------------------------------
    # 전체
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        predictions,
    ) / 1_000_000

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    ) / 1_000_000

    r2 = r2_score(
        y_true,
        predictions,
    )

    # --------------------------------------------------------
    # Top 10%
    # --------------------------------------------------------

    q90 = result_df[
        TARGET
    ].quantile(0.90)

    top10 = result_df[
        result_df[TARGET] >= q90
    ]

    top10_mae = mean_absolute_error(
        top10[TARGET],
        top10["predicted_fee"],
    ) / 1_000_000

    # --------------------------------------------------------
    # 50M+
    # --------------------------------------------------------

    high_value = result_df[
        result_df[TARGET]
        >= 50_000_000
    ]

    if len(high_value) > 0:
        high_actual_mean = (
            high_value[
                "actual_m"
            ].mean()
        )

        high_pred_mean = (
            high_value[
                "predicted_m"
            ].mean()
        )

        high_mae = (
            high_value[
                "absolute_error"
            ].mean()
            / 1_000_000
        )

        high_mean_error = (
            high_actual_mean
            - high_pred_mean
        )

    else:
        high_actual_mean = np.nan
        high_pred_mean = np.nan
        high_mae = np.nan
        high_mean_error = np.nan

    summary = {
        "model": model_name,
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
        "50M+_Mean_Error_M": (
            high_mean_error
        ),
    }

    return summary, result_df


# ============================================================
# 목적 리그 가상 시나리오 생성
# ============================================================

def make_destination_scenario(
    original_df,
    destination_league,
):
    scenario_df = (
        original_df.copy()
    )

    # 목적 리그
    scenario_df[
        "to_league_id"
    ] = destination_league

    # 동일 리그 여부
    scenario_df[
        "is_same_league"
    ] = (
        scenario_df[
            "from_league_id"
        ].astype(str)
        == str(destination_league)
    ).astype(int)

    # 모두 5대 리그이므로 1
    scenario_df[
        "is_top5_destination"
    ] = 1

    return scenario_df


# ============================================================
# Counterfactual 목적 리그 테스트
# ============================================================

def run_counterfactual_test(
    model,
    test_df,
    features,
):
    result_df = pd.DataFrame(
        index=test_df.index
    )

    if (
        "player_name"
        in test_df.columns
    ):
        result_df[
            "player_name"
        ] = test_df[
            "player_name"
        ]

    if (
        "from_league_id"
        in test_df.columns
    ):
        result_df[
            "from_league_id"
        ] = test_df[
            "from_league_id"
        ]

    for (
        league_id,
        league_name,
    ) in TOP5_LEAGUES.items():

        scenario_df = (
            make_destination_scenario(
                test_df,
                league_id,
            )
        )

        X_scenario = (
            scenario_df[
                features
            ]
        )

        predictions = predict_fee(
            model,
            X_scenario,
        )

        result_df[
            f"{league_id}_pred_m"
        ] = (
            predictions
            / 1_000_000
        )

    return result_df


# ============================================================
# Counterfactual 리그별 요약
# ============================================================

def summarize_counterfactual(
    model_name,
    counterfactual_df,
):
    rows = []

    for (
        league_id,
        league_name,
    ) in TOP5_LEAGUES.items():

        column = (
            f"{league_id}_pred_m"
        )

        rows.append(
            {
                "model": model_name,
                "league_id": league_id,
                "league_name": league_name,
                "mean_pred_m": (
                    counterfactual_df[
                        column
                    ].mean()
                ),
                "median_pred_m": (
                    counterfactual_df[
                        column
                    ].median()
                ),
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    es1_rows = summary_df[
        summary_df[
            "league_id"
        ] == "ES1"
    ]

    if len(es1_rows) == 0:
        raise ValueError(
            "ES1 Counterfactual 결과가 없습니다."
        )

    es1_mean = (
        es1_rows[
            "mean_pred_m"
        ].iloc[0]
    )

    summary_df[
        "vs_ES1_diff_m"
    ] = (
        summary_df[
            "mean_pred_m"
        ]
        - es1_mean
    )

    if es1_mean > 0:
        summary_df[
            "vs_ES1_pct"
        ] = (
            (
                summary_df[
                    "mean_pred_m"
                ]
                / es1_mean
            )
            - 1
        ) * 100

    else:
        summary_df[
            "vs_ES1_pct"
        ] = np.nan

    return summary_df


# ============================================================
# GB1 vs ES1 핵심 수치
# ============================================================

def get_gb1_es1_metrics(
    counterfactual_df,
):
    gb1_mean = (
        counterfactual_df[
            "GB1_pred_m"
        ].mean()
    )

    es1_mean = (
        counterfactual_df[
            "ES1_pred_m"
        ].mean()
    )

    diff = (
        gb1_mean
        - es1_mean
    )

    if es1_mean > 0:
        premium_pct = (
            (
                gb1_mean
                / es1_mean
            )
            - 1
        ) * 100
    else:
        premium_pct = np.nan

    return {
        "GB1_CF_Mean_M": gb1_mean,
        "ES1_CF_Mean_M": es1_mean,
        "GB1_minus_ES1_M": diff,
        "GB1_vs_ES1_Pct": premium_pct,
    }


# ============================================================
# GB1 vs ES1 선수별
# ============================================================

def make_player_gb1_es1_compare(
    counterfactual_df,
):
    result_df = (
        counterfactual_df.copy()
    )

    result_df[
        "GB1_minus_ES1_m"
    ] = (
        result_df[
            "GB1_pred_m"
        ]
        - result_df[
            "ES1_pred_m"
        ]
    )

    result_df[
        "GB1_vs_ES1_pct"
    ] = np.where(
        result_df[
            "ES1_pred_m"
        ] > 0,
        (
            (
                result_df[
                    "GB1_pred_m"
                ]
                / result_df[
                    "ES1_pred_m"
                ]
            )
            - 1
        )
        * 100,
        np.nan,
    )

    result_df = (
        result_df
        .sort_values(
            "GB1_minus_ES1_m",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result_df


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 데이터 / 모델 로드
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

    print()
    print(
        "=" * 70
    )
    print(
        "v1.2 Feature Ablation Test"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Base FEATURES:"
    )
    print(
        base_features
    )

    if (
        "value_at_transfer"
        in base_features
    ):
        raise ValueError(
            "현재 모델에 value_at_transfer가 "
            "포함되어 있습니다."
        )

    print()
    print(
        "✓ no-market 모델 확인"
    )


    # ========================================================
    # 2. Feature Engineering
    # ========================================================

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

    missing_columns = [
        column
        for column in base_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 다음 feature가 없습니다:\n"
            f"{missing_columns}"
        )


    # ========================================================
    # 3. 시간 분할
    # ========================================================

    train_df = df[
        df[
            "transfer_date"
        ].dt.year < 2025
    ].copy()

    test_df = df[
        df[
            "transfer_date"
        ].dt.year == 2025
    ].copy()

    y_train = train_df[
        TARGET
    ]

    print()
    print(
        "=" * 70
    )
    print(
        "Data Split"
    )
    print(
        "=" * 70
    )

    print(
        f"Train rows : {len(train_df)}"
    )

    print(
        f"Test rows  : {len(test_df)}"
    )

    print(
        "Train year :",
        int(
            train_df[
                "transfer_date"
            ].dt.year.min()
        ),
        "~",
        int(
            train_df[
                "transfer_date"
            ].dt.year.max()
        ),
    )

    print(
        "Test year  :",
        int(
            test_df[
                "transfer_date"
            ].dt.year.min()
        ),
    )


    # ========================================================
    # 4. Weight C 확인
    # ========================================================

    sample_weight = (
        make_weight_c(
            y_train
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Weight C"
    )
    print(
        "=" * 70
    )

    print(
        "<30M       : 1.0"
    )

    print(
        "30M~50M    : 2.5"
    )

    print(
        "50M+       : 4.0"
    )

    print()
    print(
        "Train weight 분포:"
    )

    unique_weights, counts = (
        np.unique(
            sample_weight,
            return_counts=True,
        )
    )

    for weight, count in zip(
        unique_weights,
        counts,
    ):
        print(
            f"weight={weight:.1f}: "
            f"{count}명"
        )


    # ========================================================
    # 5. 각 모델 실험
    # ========================================================

    overall_results = []
    counterfactual_summaries = []
    player_results = {}

    for (
        model_name,
        removed_features,
    ) in MODEL_VARIANTS.items():

        print()
        print(
            "=" * 70
        )
        print(
            f"Training: {model_name}"
        )
        print(
            "=" * 70
        )

        print(
            "제거 feature:",
            (
                removed_features
                if removed_features
                else "없음"
            ),
        )

        # ----------------------------------------------------
        # 사용 feature
        # ----------------------------------------------------

        features = [
            feature
            for feature in base_features
            if feature
            not in removed_features
        ]

        print()
        print(
            f"Feature count: {len(features)}"
        )

        print(
            features
        )

        # ----------------------------------------------------
        # 모델 생성
        # ----------------------------------------------------

        model = (
            remove_features_from_model(
                base_model,
                removed_features,
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

        # ----------------------------------------------------
        # 학습
        # ----------------------------------------------------

        X_train = train_df[
            features
        ]

        X_test = test_df[
            features
        ]

        model.fit(
            X_train,
            np.log1p(
                y_train
            ),
            **fit_params,
        )

        # ----------------------------------------------------
        # 예측
        # ----------------------------------------------------

        predictions = (
            predict_fee(
                model,
                X_test,
            )
        )

        # ----------------------------------------------------
        # 성능 평가
        # ----------------------------------------------------

        summary, result_df = (
            evaluate_model(
                model_name,
                test_df,
                predictions,
            )
        )

        # ----------------------------------------------------
        # Counterfactual
        # ----------------------------------------------------

        counterfactual_df = (
            run_counterfactual_test(
                model,
                test_df,
                features,
            )
        )

        cf_metrics = (
            get_gb1_es1_metrics(
                counterfactual_df
            )
        )

        summary.update(
            cf_metrics
        )

        overall_results.append(
            summary
        )

        # ----------------------------------------------------
        # 리그 전체 Counterfactual
        # ----------------------------------------------------

        cf_summary = (
            summarize_counterfactual(
                model_name,
                counterfactual_df,
            )
        )

        counterfactual_summaries.append(
            cf_summary
        )

        # ----------------------------------------------------
        # 선수별 GB1 vs ES1
        # ----------------------------------------------------

        player_compare = (
            make_player_gb1_es1_compare(
                counterfactual_df
            )
        )

        player_results[
            model_name
        ] = player_compare

        # ----------------------------------------------------
        # 진행 결과
        # ----------------------------------------------------

        print()
        print(
            f"✓ {model_name} 완료"
        )

        print(
            "MAE              : "
            f"{summary['MAE_M']:.2f}M"
        )

        print(
            "RMSE             : "
            f"{summary['RMSE_M']:.2f}M"
        )

        print(
            "R²               : "
            f"{summary['R2']:.4f}"
        )

        print(
            "50M+ MAE         : "
            f"{summary['50M+_MAE_M']:.2f}M"
        )

        print(
            "50M+ 예측 평균    : "
            f"{summary['50M+_Pred_Mean_M']:.2f}M"
        )

        print(
            "GB1 CF 평균       : "
            f"{summary['GB1_CF_Mean_M']:.2f}M"
        )

        print(
            "ES1 CF 평균       : "
            f"{summary['ES1_CF_Mean_M']:.2f}M"
        )

        print(
            "GB1 vs ES1       : "
            f"{summary['GB1_vs_ES1_Pct']:.2f}%"
        )


    # ========================================================
    # 6. 핵심 결과표
    # ========================================================

    summary_df = pd.DataFrame(
        overall_results
    )

    display_columns = [
        "model",
        "MAE_M",
        "RMSE_M",
        "R2",
        "Top10_MAE_M",
        "50M+_Actual_Mean_M",
        "50M+_Pred_Mean_M",
        "50M+_MAE_M",
        "50M+_Mean_Error_M",
        "GB1_CF_Mean_M",
        "ES1_CF_Mean_M",
        "GB1_minus_ES1_M",
        "GB1_vs_ES1_Pct",
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "★ v1.2 Feature Ablation 결과"
    )
    print(
        "=" * 100
    )

    print(
        summary_df[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 7. Counterfactual 전체 리그
    # ========================================================

    cf_all_df = pd.concat(
        counterfactual_summaries,
        ignore_index=True,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "★ 모델별 목적 리그 Counterfactual"
    )
    print(
        "=" * 100
    )

    print(
        cf_all_df
        .round(2)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 8. 각 모델 GB1 프리미엄 TOP 10
    # ========================================================

    for (
        model_name,
        player_df,
    ) in player_results.items():

        print()
        print(
            "=" * 100
        )
        print(
            f"{model_name} - GB1 프리미엄 TOP 10"
        )
        print(
            "=" * 100
        )

        display_player_columns = [
            column
            for column in [
                "player_name",
                "from_league_id",
                "GB1_pred_m",
                "ES1_pred_m",
                "IT1_pred_m",
                "L1_pred_m",
                "FR1_pred_m",
                "GB1_minus_ES1_m",
                "GB1_vs_ES1_pct",
            ]
            if column
            in player_df.columns
        ]

        print(
            player_df[
                display_player_columns
            ]
            .head(10)
            .round(2)
            .to_string(
                index=False
            )
        )


    # ========================================================
    # 9. 결과 저장
    # ========================================================

    summary_file = (
        OUTPUT_DIR
        / "v12_feature_ablation_summary.csv"
    )

    cf_file = (
        OUTPUT_DIR
        / "v12_feature_ablation_counterfactual.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    cf_all_df.to_csv(
        cf_file,
        index=False,
        encoding="utf-8-sig",
    )

    for (
        model_name,
        player_df,
    ) in player_results.items():

        safe_name = (
            model_name
            .lower()
        )

        output_file = (
            OUTPUT_DIR
            / (
                "v12_ablation_"
                f"{safe_name}"
                "_gb1_vs_es1.csv"
            )
        )

        player_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )


    print()
    print(
        "=" * 70
    )
    print(
        "저장 완료"
    )
    print(
        "=" * 70
    )

    print(
        summary_file
    )

    print(
        cf_file
    )

    print()
    print(
        "★ 다음 판단 기준"
    )

    print(
        "1. MAE / RMSE / R²가 얼마나 유지되는지"
    )

    print(
        "2. 50M+ MAE가 악화되지 않는지"
    )

    print(
        "3. GB1 vs ES1 Counterfactual 차이가 "
        "얼마나 줄어드는지"
    )


if __name__ == "__main__":
    main()