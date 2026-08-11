import copy
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 파일 경로
# ============================================================

DATA_FILE = "data/processed/training_dataset.csv"
BASE_MODEL_FILE = "models/transfer_fee_model_v1_1.joblib"

TARGET = "transfer_fee"


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
        (df["goals"] + df["assists"]) / df["minutes"] * 90,
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

    df["age_squared"] = df["age_at_transfer"] ** 2

    return df


# ============================================================
# 평가 함수
# ============================================================

def evaluate_model(name, y_true, predictions, test_df):
    result_df = test_df.copy()

    result_df["predicted_fee"] = predictions
    result_df["absolute_error"] = np.abs(
        result_df[TARGET] - result_df["predicted_fee"]
    )

    result_df["actual_m"] = result_df[TARGET] / 1_000_000
    result_df["predicted_m"] = result_df["predicted_fee"] / 1_000_000
    result_df["error_m"] = (
        result_df[TARGET] - result_df["predicted_fee"]
    ) / 1_000_000

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

    q80 = result_df[TARGET].quantile(0.80)
    q90 = result_df[TARGET].quantile(0.90)

    top20 = result_df[
        result_df[TARGET] >= q80
    ]

    top10 = result_df[
        result_df[TARGET] >= q90
    ]

    top20_mae = mean_absolute_error(
        top20[TARGET],
        top20["predicted_fee"],
    ) / 1_000_000

    top10_mae = mean_absolute_error(
        top10[TARGET],
        top10["predicted_fee"],
    ) / 1_000_000

    top20_actual_mean = top20["actual_m"].mean()
    top20_pred_mean = top20["predicted_m"].mean()

    top10_actual_mean = top10["actual_m"].mean()
    top10_pred_mean = top10["predicted_m"].mean()

    # 가격 구간
    bins = [
        0,
        10_000_000,
        30_000_000,
        50_000_000,
        np.inf,
    ]

    labels = [
        "0~10M",
        "10~30M",
        "30~50M",
        "50M+",
    ]

    result_df["fee_group"] = pd.cut(
        result_df[TARGET],
        bins=bins,
        labels=labels,
        right=False,
    )

    group_result = (
        result_df
        .groupby(
            "fee_group",
            observed=True,
        )
        .agg(
            count=(TARGET, "size"),
            actual_mean=("actual_m", "mean"),
            predicted_mean=("predicted_m", "mean"),
            mae=(
                "absolute_error",
                lambda x: x.mean() / 1_000_000,
            ),
            mean_error=("error_m", "mean"),
        )
    )

    # 원하는 비교용 값 추출
    def group_value(group_name, column):
        if group_name not in group_result.index:
            return np.nan
        return group_result.loc[group_name, column]

    summary = {
        "model": name,
        "MAE_M": mae,
        "RMSE_M": rmse,
        "R2": r2,
        "Top20_MAE_M": top20_mae,
        "Top20_Actual_Mean_M": top20_actual_mean,
        "Top20_Pred_Mean_M": top20_pred_mean,
        "Top10_MAE_M": top10_mae,
        "Top10_Actual_Mean_M": top10_actual_mean,
        "Top10_Pred_Mean_M": top10_pred_mean,
        "30~50M_Actual_Mean_M": group_value(
            "30~50M",
            "actual_mean",
        ),
        "30~50M_Pred_Mean_M": group_value(
            "30~50M",
            "predicted_mean",
        ),
        "30~50M_MAE_M": group_value(
            "30~50M",
            "mae",
        ),
        "50M+_Actual_Mean_M": group_value(
            "50M+",
            "actual_mean",
        ),
        "50M+_Pred_Mean_M": group_value(
            "50M+",
            "predicted_mean",
        ),
        "50M+_MAE_M": group_value(
            "50M+",
            "mae",
        ),
        "50M+_Mean_Error_M": group_value(
            "50M+",
            "mean_error",
        ),
    }

    return summary, group_result, result_df


# ============================================================
# 메인
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 데이터 / 기존 v1.1 파이프라인 로드
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    base_model = joblib.load(
        BASE_MODEL_FILE
    )

    FEATURES = list(
        base_model.feature_names_in_
    )

    print("\n========================================")
    print("기존 v1.1 모델 확인")
    print("========================================")

    print("\nFEATURES:")
    print(FEATURES)

    if "value_at_transfer" in FEATURES:
        raise ValueError(
            "현재 모델은 value_at_transfer를 사용합니다. "
            "no-market v1.1 모델이 아닙니다."
        )

    print("\n✓ no-market v1.1 확인")


    # --------------------------------------------------------
    # 2. 날짜 처리 / Feature Engineering
    # --------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df = add_v11_features(df)

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"다음 feature가 데이터에 없습니다: {missing_columns}"
        )


    # --------------------------------------------------------
    # 3. 동일한 시간 분할
    # --------------------------------------------------------
    # 2020~2024 → Train
    # 2025      → Test
    #
    # 두 모델 모두 완전히 같은 데이터로 다시 학습한다.
    # 따라서 차이는 target 변환(Log vs Raw)에서 발생한다.
    # --------------------------------------------------------

    train_df = df[
        df["transfer_date"].dt.year < 2025
    ].copy()

    test_df = df[
        df["transfer_date"].dt.year == 2025
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\n========================================")
    print("데이터 분할")
    print("========================================")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")


    # --------------------------------------------------------
    # 4. 파이프라인 복제
    # --------------------------------------------------------
    # 기존 v1.1의 전처리 + XGBoost 하이퍼파라미터를 그대로 복제
    # --------------------------------------------------------

    try:
        log_model = clone(base_model)
        raw_model = clone(base_model)
    except Exception:
        print(
            "\n주의: sklearn.clone이 실패해 deepcopy로 복제합니다."
        )
        log_model = copy.deepcopy(base_model)
        raw_model = copy.deepcopy(base_model)


    # --------------------------------------------------------
    # 5. Log Target 모델 학습
    # --------------------------------------------------------

    print("\n========================================")
    print("1/2 Log Target 모델 학습")
    print("========================================")

    log_model.fit(
        X_train,
        np.log1p(y_train),
    )

    log_pred = np.expm1(
        log_model.predict(X_test)
    )

    log_pred = np.maximum(
        log_pred,
        0,
    )

    print("✓ Log 모델 학습/예측 완료")


    # --------------------------------------------------------
    # 6. Raw Target 모델 학습
    # --------------------------------------------------------

    print("\n========================================")
    print("2/2 Raw Target 모델 학습")
    print("========================================")

    raw_model.fit(
        X_train,
        y_train,
    )

    raw_pred = raw_model.predict(
        X_test
    )

    raw_pred = np.maximum(
        raw_pred,
        0,
    )

    print("✓ Raw 모델 학습/예측 완료")


    # --------------------------------------------------------
    # 7. 평가
    # --------------------------------------------------------

    log_summary, log_groups, log_result = evaluate_model(
        "Log Target",
        y_test,
        log_pred,
        test_df,
    )

    raw_summary, raw_groups, raw_result = evaluate_model(
        "Raw Target",
        y_test,
        raw_pred,
        test_df,
    )

    summary_df = pd.DataFrame(
        [
            log_summary,
            raw_summary,
        ]
    )


    # --------------------------------------------------------
    # 8. 핵심 비교표
    # --------------------------------------------------------

    display_columns = [
        "model",
        "MAE_M",
        "RMSE_M",
        "R2",
        "Top10_MAE_M",
        "30~50M_Pred_Mean_M",
        "50M+_Pred_Mean_M",
        "50M+_MAE_M",
        "50M+_Mean_Error_M",
    ]

    print("\n========================================")
    print("★ Log vs Raw 핵심 비교")
    print("========================================")

    print(
        summary_df[
            display_columns
        ]
        .round(2)
        .to_string(
            index=False,
        )
    )


    # --------------------------------------------------------
    # 9. 가격 구간별 상세 비교
    # --------------------------------------------------------

    print("\n========================================")
    print("Log Target - 가격 구간별")
    print("========================================")

    print(
        log_groups
        .round(2)
        .to_string()
    )

    print("\n========================================")
    print("Raw Target - 가격 구간별")
    print("========================================")

    print(
        raw_groups
        .round(2)
        .to_string()
    )


    # --------------------------------------------------------
    # 10. 50M+ 선수 개별 비교
    # --------------------------------------------------------

    compare_df = test_df.copy()

    compare_df["actual_m"] = (
        compare_df[TARGET] / 1_000_000
    )

    compare_df["log_pred_m"] = (
        log_pred / 1_000_000
    )

    compare_df["raw_pred_m"] = (
        raw_pred / 1_000_000
    )

    compare_df["log_error_m"] = (
        compare_df["actual_m"]
        - compare_df["log_pred_m"]
    )

    compare_df["raw_error_m"] = (
        compare_df["actual_m"]
        - compare_df["raw_pred_m"]
    )

    expensive = (
        compare_df[
            compare_df[TARGET] >= 50_000_000
        ]
        .sort_values(
            TARGET,
            ascending=False,
        )
        .copy()
    )

    player_columns = [
        column
        for column in [
            "player_name",
            "actual_m",
            "log_pred_m",
            "raw_pred_m",
            "log_error_m",
            "raw_error_m",
        ]
        if column in expensive.columns
    ]

    print("\n========================================")
    print("50M+ 선수 Log vs Raw")
    print("========================================")

    print(
        expensive[
            player_columns
        ]
        .round(2)
        .to_string(
            index=False,
        )
    )


    # --------------------------------------------------------
    # 11. 결과 CSV 저장
    # --------------------------------------------------------

    from pathlib import Path

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_dir / "log_vs_raw_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expensive[
        player_columns
    ].to_csv(
        output_dir / "log_vs_raw_50m_players.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========================================")
    print("저장 완료")
    print("========================================")

    print(
        "results/log_vs_raw_summary.csv"
    )

    print(
        "results/log_vs_raw_50m_players.csv"
    )


if __name__ == "__main__":
    main()