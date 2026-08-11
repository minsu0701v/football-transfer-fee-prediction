import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
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


# ============================================================
# 리그 설정
# ============================================================

TOP5_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}


# ============================================================
# Weight C
# ============================================================
#
# < 30M       -> 1.0
# 30M ~ 50M   -> 2.5
# >= 50M      -> 4.0
#
# ============================================================

def make_weight_c(y):
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
# v1.1 파생변수 생성
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
# Pipeline의 최종 estimator 이름 찾기
# ============================================================

def get_estimator_step_name(model):
    """
    sklearn Pipeline의 마지막 모델 step 이름 반환.

    예:
        preprocess -> model

    이 경우 "model" 반환.
    """

    if not hasattr(model, "steps"):
        raise ValueError(
            "불러온 모델이 sklearn Pipeline 형태가 아닙니다."
        )

    if len(model.steps) == 0:
        raise ValueError(
            "Pipeline에 step이 없습니다."
        )

    return model.steps[-1][0]


# ============================================================
# 예측 함수
# ============================================================

def predict_transfer_fee(
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
# 전체 모델 성능 평가
# ============================================================

def evaluate_overall(
    model_name,
    y_true,
    predictions,
):
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

    return {
        "model": model_name,
        "MAE_M": mae,
        "RMSE_M": rmse,
        "R2": r2,
    }


# ============================================================
# 실제 목적 리그별 모델 성능
# ============================================================

def evaluate_by_destination_league(
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

    result_df["absolute_error_m"] = (
        (
            result_df[TARGET]
            - result_df["predicted_fee"]
        )
        .abs()
        / 1_000_000
    )

    result_df["error_m"] = (
        result_df[TARGET]
        - result_df["predicted_fee"]
    ) / 1_000_000

    league_result = (
        result_df
        .groupby(
            "to_league_id",
            observed=True,
        )
        .agg(
            count=(
                TARGET,
                "size",
            ),
            actual_mean=(
                "actual_m",
                "mean",
            ),
            predicted_mean=(
                "predicted_m",
                "mean",
            ),
            mae=(
                "absolute_error_m",
                "mean",
            ),
            mean_error=(
                "error_m",
                "mean",
            ),
        )
        .reset_index()
    )

    league_result["league_name"] = (
        league_result[
            "to_league_id"
        ].map(
            TOP5_LEAGUES
        )
    )

    # 보기 좋게 top5 먼저
    league_order = list(
        TOP5_LEAGUES.keys()
    )

    league_result["sort_order"] = (
        league_result[
            "to_league_id"
        ]
        .apply(
            lambda x:
            (
                league_order.index(x)
                if x in league_order
                else 999
            )
        )
    )

    league_result = (
        league_result
        .sort_values(
            [
                "sort_order",
                "to_league_id",
            ]
        )
        .drop(
            columns=["sort_order"]
        )
        .reset_index(
            drop=True
        )
    )

    return league_result


# ============================================================
# Counterfactual 목적 리그 실험
# ============================================================

def make_destination_scenario(
    original_df,
    destination_league,
):
    """
    같은 선수 데이터를 그대로 두고
    목적 리그만 바꾼 가상 데이터 생성.

    실제 프런트 예측과 최대한 동일하게:

    - to_league_id 변경
    - is_same_league 재계산
    - is_top5_destination 재계산
    """

    scenario_df = (
        original_df.copy()
    )

    scenario_df[
        "to_league_id"
    ] = destination_league

    # 현재 리그와 목적 리그가 같은가?
    scenario_df[
        "is_same_league"
    ] = (
        scenario_df[
            "from_league_id"
        ].astype(str)
        == str(
            destination_league
        )
    ).astype(int)

    # 5대 리그 여부
    scenario_df[
        "is_top5_destination"
    ] = (
        destination_league
        in TOP5_LEAGUES
    )

    scenario_df[
        "is_top5_destination"
    ] = (
        scenario_df[
            "is_top5_destination"
        ].astype(int)
    )

    return scenario_df


# ============================================================
# 모든 선수에게 각 목적 리그를 적용
# ============================================================

def run_counterfactual_test(
    model,
    test_df,
    features,
):
    result_df = pd.DataFrame(
        index=test_df.index
    )

    # 선수 이름이 있으면 같이 저장
    if "player_name" in test_df.columns:
        result_df[
            "player_name"
        ] = test_df[
            "player_name"
        ]

    if "from_league_id" in test_df.columns:
        result_df[
            "from_league_id"
        ] = test_df[
            "from_league_id"
        ]

    for league_id, league_name in (
        TOP5_LEAGUES.items()
    ):

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

        pred = (
            predict_transfer_fee(
                model,
                X_scenario,
            )
        )

        result_df[
            f"{league_id}_pred_m"
        ] = (
            pred
            / 1_000_000
        )

    return result_df


# ============================================================
# Counterfactual 평균 결과
# ============================================================

def summarize_counterfactual(
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

    # --------------------------------------------------------
    # La Liga를 기준으로 프리미엄 비교
    # --------------------------------------------------------

    es1_mean = (
        summary_df.loc[
            summary_df[
                "league_id"
            ] == "ES1",
            "mean_pred_m",
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

    summary_df[
        "vs_ES1_pct"
    ] = np.where(
        es1_mean > 0,
        (
            summary_df[
                "mean_pred_m"
            ]
            / es1_mean
            - 1
        )
        * 100,
        np.nan,
    )

    return summary_df


# ============================================================
# GB1 vs ES1 선수별 비교
# ============================================================

def compare_gb1_vs_es1(
    counterfactual_df,
):
    result = (
        counterfactual_df.copy()
    )

    result[
        "GB1_minus_ES1_m"
    ] = (
        result[
            "GB1_pred_m"
        ]
        - result[
            "ES1_pred_m"
        ]
    )

    result[
        "GB1_vs_ES1_pct"
    ] = np.where(
        result[
            "ES1_pred_m"
        ] > 0,
        (
            result[
                "GB1_pred_m"
            ]
            / result[
                "ES1_pred_m"
            ]
            - 1
        )
        * 100,
        np.nan,
    )

    result = (
        result
        .sort_values(
            "GB1_minus_ES1_m",
            ascending=False,
        )
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

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

    features = list(
        base_model.feature_names_in_
    )

    print(
        "\n========================================"
    )
    print(
        "v1.1 모델 확인"
    )
    print(
        "========================================"
    )

    print(
        "\nFEATURES:"
    )
    print(
        features
    )

    if (
        "value_at_transfer"
        in features
    ):
        raise ValueError(
            "현재 모델에 "
            "value_at_transfer가 포함되어 있습니다."
        )

    print(
        "\n✓ no-market v1.1 확인"
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
        for column in features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "다음 feature가 없습니다:\n"
            f"{missing_columns}"
        )


    # ========================================================
    # 3. Train / Test
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

    print(
        "\n========================================"
    )
    print(
        "데이터 분할"
    )
    print(
        "========================================"
    )

    print(
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Test rows : {len(test_df)}"
    )

    X_train = (
        train_df[
            features
        ]
    )

    X_test = (
        test_df[
            features
        ]
    )

    y_train = (
        train_df[
            TARGET
        ]
    )

    y_test = (
        test_df[
            TARGET
        ]
    )


    # ========================================================
    # 4. Baseline 모델
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "1/2 Baseline v1.1 학습"
    )
    print(
        "========================================"
    )

    baseline_model = clone_model(
        base_model
    )

    baseline_model.fit(
        X_train,
        np.log1p(
            y_train
        ),
    )

    baseline_pred = (
        predict_transfer_fee(
            baseline_model,
            X_test,
        )
    )

    print(
        "✓ Baseline 학습 완료"
    )


    # ========================================================
    # 5. Weight C 모델
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "2/2 Weight C 학습"
    )
    print(
        "========================================"
    )

    weight_model = clone_model(
        base_model
    )

    sample_weight = (
        make_weight_c(
            y_train
        )
    )

    estimator_step = (
        get_estimator_step_name(
            weight_model
        )
    )

    print(
        f"Estimator step: "
        f"{estimator_step}"
    )

    print(
        "\nWeight C:"
    )

    print(
        "< 30M     : 1.0"
    )
    print(
        "30~50M    : 2.5"
    )
    print(
        "50M+      : 4.0"
    )

    fit_params = {
        (
            f"{estimator_step}"
            "__sample_weight"
        ):
        sample_weight
    }

    weight_model.fit(
        X_train,
        np.log1p(
            y_train
        ),
        **fit_params,
    )

    weight_pred = (
        predict_transfer_fee(
            weight_model,
            X_test,
        )
    )

    print(
        "✓ Weight C 학습 완료"
    )


    # ========================================================
    # 6. 전체 성능 비교
    # ========================================================

    baseline_overall = (
        evaluate_overall(
            "v1.1 Baseline",
            y_test,
            baseline_pred,
        )
    )

    weight_overall = (
        evaluate_overall(
            "Weight C",
            y_test,
            weight_pred,
        )
    )

    overall_df = pd.DataFrame(
        [
            baseline_overall,
            weight_overall,
        ]
    )

    print(
        "\n========================================"
    )
    print(
        "★ 전체 성능"
    )
    print(
        "========================================"
    )

    print(
        overall_df
        .round(3)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 7. 실제 목적 리그별 성능
    # ========================================================

    baseline_leagues = (
        evaluate_by_destination_league(
            test_df,
            baseline_pred,
        )
    )

    weight_leagues = (
        evaluate_by_destination_league(
            test_df,
            weight_pred,
        )
    )

    print(
        "\n========================================"
    )
    print(
        "Baseline - 실제 목적 리그별 성능"
    )
    print(
        "========================================"
    )

    print(
        baseline_leagues
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )
    print(
        "Weight C - 실제 목적 리그별 성능"
    )
    print(
        "========================================"
    )

    print(
        weight_leagues
        .round(2)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 8. Counterfactual - Baseline
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "Baseline - 동일 선수 목적 리그 변경 실험"
    )
    print(
        "========================================"
    )

    baseline_cf = (
        run_counterfactual_test(
            baseline_model,
            test_df,
            features,
        )
    )

    baseline_cf_summary = (
        summarize_counterfactual(
            baseline_cf
        )
    )

    print(
        baseline_cf_summary
        .round(2)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 9. Counterfactual - Weight C
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "Weight C - 동일 선수 목적 리그 변경 실험"
    )
    print(
        "========================================"
    )

    weight_cf = (
        run_counterfactual_test(
            weight_model,
            test_df,
            features,
        )
    )

    weight_cf_summary = (
        summarize_counterfactual(
            weight_cf
        )
    )

    print(
        weight_cf_summary
        .round(2)
        .to_string(
            index=False
        )
    )


    # ========================================================
# 10. GB1 vs ES1
# ========================================================

    baseline_gb1_es1 = compare_gb1_vs_es1(
        baseline_cf
    )

    weight_gb1_es1 = compare_gb1_vs_es1(
        weight_cf
    )


# --------------------------------------------------------
# Baseline 평균 계산
# --------------------------------------------------------

    baseline_gb1_mean = baseline_cf[
        "GB1_pred_m"
    ].mean()

    baseline_es1_mean = baseline_cf[
        "ES1_pred_m"
    ].mean()

    baseline_diff = (
        baseline_gb1_mean
        - baseline_es1_mean
    )

    baseline_premium = (
        (
            baseline_gb1_mean
            / baseline_es1_mean
        )
    -    1
    ) * 100


    print(
        "\n========================================"
    )
    print(
        "Baseline - GB1 vs ES1 평균"
    )
    print(
        "========================================"
    )

    print(
        f"GB1 평균 : {baseline_gb1_mean:.2f}M"
    )

    print(
        f"ES1 평균 : {baseline_es1_mean:.2f}M"
    )

    print(
        f"차이     : {baseline_diff:.2f}M"
    )

    print(
        f"프리미엄 : {baseline_premium:.2f}%"
    )


# --------------------------------------------------------
# Weight C 평균 계산
# --------------------------------------------------------

    weight_gb1_mean = weight_cf[
        "GB1_pred_m"
    ].mean()

    weight_es1_mean = weight_cf[
        "ES1_pred_m"
    ].mean()

    weight_diff = (
        weight_gb1_mean
        - weight_es1_mean
    )

    weight_premium = (
        (
            weight_gb1_mean
            / weight_es1_mean
        )
        - 1
    )* 100


    print(
        "\n========================================"
    )
    print(
        "Weight C - GB1 vs ES1 평균"
    )
    print(
        "========================================"
    )

    print(
        f"GB1 평균 : {weight_gb1_mean:.2f}M"
    )

    print(
        f"ES1 평균 : {weight_es1_mean:.2f}M"
    )

    print(
        f"차이     : {weight_diff:.2f}M"
    )

    print(
        f"프리미엄 : {weight_premium:.2f}%"
    )

    # ========================================================
    # 11. PL 프리미엄 가장 큰 선수 TOP 20
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "Weight C - GB1 프리미엄 TOP 20"
    )
    print(
        "========================================"
    )

    display_columns = [
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
        in weight_gb1_es1.columns
    ]

    print(
        weight_gb1_es1[
            display_columns
        ]
        .head(20)
        .round(2)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 12. 결과 저장
    # ========================================================

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        output_dir
        / "v12_league_bias_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_leagues.to_csv(
        output_dir
        / "v11_baseline_league_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_leagues.to_csv(
        output_dir
        / "v12_weight_c_league_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_cf_summary.to_csv(
        output_dir
        / "v11_baseline_counterfactual_leagues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_cf_summary.to_csv(
        output_dir
        / "v12_weight_c_counterfactual_leagues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_gb1_es1.to_csv(
        output_dir
        / "v12_weight_c_gb1_vs_es1_players.csv",
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\n========================================"
    )
    print(
        "저장 완료"
    )
    print(
        "========================================"
    )

    print(
        "results/v12_league_bias_overall.csv"
    )

    print(
        "results/v11_baseline_league_performance.csv"
    )

    print(
        "results/v12_weight_c_league_performance.csv"
    )

    print(
        "results/v11_baseline_counterfactual_leagues.csv"
    )

    print(
        "results/v12_weight_c_counterfactual_leagues.csv"
    )

    print(
        "results/v12_weight_c_gb1_vs_es1_players.csv"
    )


if __name__ == "__main__":
    main()