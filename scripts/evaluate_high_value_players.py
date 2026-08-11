import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 파일 경로
# ============================================================

DATA_FILE = "data/processed/training_dataset.csv"

MODEL_FILE = "models/transfer_fee_model_v1_1.joblib"

TARGET = "transfer_fee"


def add_v11_features(df):
    """
    no-market v1.1 모델에서 사용한 파생변수 생성
    """

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
        (df["goals"] + df["assists"]) / df["minutes"] * 90,
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
    df["age_squared"] = df["age_at_transfer"] ** 2

    return df


def print_group_result(name, subset):
    """
    특정 선수 그룹 평가 결과 출력
    """

    if subset.empty:
        print(f"\n{name}: 데이터 없음")
        return

    mae = mean_absolute_error(
        subset["transfer_fee"],
        subset["predicted_fee"],
    ) / 1_000_000

    actual_mean = subset["actual_m"].mean()
    predicted_mean = subset["predicted_m"].mean()
    mean_error = subset["error_m"].mean()

    print(f"\n{name}")
    print("-" * 40)
    print(f"선수 수             : {len(subset)}")
    print(f"실제 평균 이적료    : {actual_mean:.2f}M")
    print(f"예측 평균 이적료    : {predicted_mean:.2f}M")
    print(f"MAE                 : {mae:.2f}M")
    print(f"평균 Actual - Pred  : {mean_error:.2f}M")


def main():

    # ========================================================
    # 1. 데이터 / 모델 로드
    # ========================================================

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    model = joblib.load(MODEL_FILE)

    print("\n========================================")
    print("모델 확인")
    print("========================================")

    # 학습 당시 사용된 feature를 모델에서 직접 가져오기
    FEATURES = list(model.feature_names_in_)

    print("\n모델 feature:")
    print(FEATURES)

    # no-market 모델 검증
    if "value_at_transfer" in FEATURES:
        raise ValueError(
            "\n현재 모델은 value_at_transfer를 사용합니다.\n"
            "no-market v1.1 모델이 아닙니다."
        )

    print("\n✓ value_at_transfer 없음")
    print("✓ no-market 모델 확인")


    # ========================================================
    # 2. 2025 Test Set 생성
    # ========================================================

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    # ========================================================
    # 2. Train 이적료 분포 확인
    # ========================================================

    train_df = df[
        df["transfer_date"].dt.year < 2025
    ].copy()

    print("\n========================================")
    print("Train 이적료 분포")
    print("========================================")

    for threshold in [
        10_000_000,
        30_000_000,
        50_000_000,
        70_000_000,
        100_000_000,
    ]:
        count = (
            train_df["transfer_fee"] >= threshold
        ).sum()

        ratio = (
            count / len(train_df) * 100
            if len(train_df) > 0
            else 0
        )

        print(
            f"{threshold / 1_000_000:.0f}M+ : "
            f"{count}명 ({ratio:.2f}%)"
        )

    print("\nTrain 통계")
    print(
        train_df["transfer_fee"]
        .div(1_000_000)
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
    )

    # ========================================================
    # 3. 2025 Test Set 생성
    # ========================================================

    test_df = df[
        df["transfer_date"].dt.year == 2025
    ].copy()

    print("\n========================================")
    print("Test Set")
    print("========================================")

    print(f"2025 Test rows: {len(test_df)}")


    # ========================================================
    # 4. v1.1 Feature Engineering
    # ========================================================

    test_df = add_v11_features(test_df)

    # 모델에서 요구하는 컬럼이 모두 있는지 검사
    missing_columns = [
        column
        for column in FEATURES
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nTest dataset에 다음 feature가 없습니다:\n"
            f"{missing_columns}"
        )

    print("✓ v1.1 파생변수 생성 완료")


    # ========================================================
    # 5. 예측
    # ========================================================

    X_test = test_df[FEATURES]

    y_test = test_df[TARGET]

    # 모델은 log1p(transfer_fee)를 target으로 학습했다고 가정
    pred_log = model.predict(X_test)

    predictions = np.expm1(pred_log)

    # 음수 예측 방지
    predictions = np.maximum(
        predictions,
        0,
    )

    test_df["predicted_fee"] = predictions

    # 실제 - 예측
    # 양수 -> 과소예측
    # 음수 -> 과대예측
    test_df["error"] = (
        test_df[TARGET]
        - test_df["predicted_fee"]
    )

    test_df["absolute_error"] = (
        test_df["error"].abs()
    )

    # million EUR 단위
    test_df["actual_m"] = (
        test_df[TARGET] / 1_000_000
    )

    test_df["predicted_m"] = (
        test_df["predicted_fee"] / 1_000_000
    )

    test_df["error_m"] = (
        test_df["error"] / 1_000_000
    )


    # ========================================================
    # 6. 전체 성능
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n========================================")
    print("전체 성능")
    print("========================================")

    print(f"MAE  : {mae / 1_000_000:.2f}M")
    print(f"RMSE : {rmse / 1_000_000:.2f}M")
    print(f"R²   : {r2:.4f}")


    # ========================================================
    # 7. 실제 이적료 상위 20% / 10%
    # ========================================================

    q80 = test_df[TARGET].quantile(0.80)
    q90 = test_df[TARGET].quantile(0.90)

    top20 = test_df[
        test_df[TARGET] >= q80
    ].copy()

    top10 = test_df[
        test_df[TARGET] >= q90
    ].copy()

    print("\n========================================")
    print("상위권 선수 성능")
    print("========================================")

    print(
        f"\n상위 20% 기준 이적료: "
        f"{q80 / 1_000_000:.2f}M 이상"
    )

    print(
        f"상위 10% 기준 이적료: "
        f"{q90 / 1_000_000:.2f}M 이상"
    )

    print_group_result(
        "상위 20%",
        top20,
    )

    print_group_result(
        "상위 10%",
        top10,
    )


    # ========================================================
    # 8. 가격 구간별 성능
    # ========================================================

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

    test_df["fee_group"] = pd.cut(
        test_df[TARGET],
        bins=bins,
        labels=labels,
        right=False,
    )

    group_result = (
        test_df
        .groupby(
            "fee_group",
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
                "absolute_error",
                lambda x: x.mean() / 1_000_000,
            ),
            mean_error=(
                "error_m",
                "mean",
            ),
        )
    )

    print("\n========================================")
    print("가격 구간별 성능")
    print("========================================")

    print(
        group_result
        .round(2)
        .to_string()
    )


    # ========================================================
    # 9. 30M+ 선수 개별 확인
    # ========================================================

    expensive = (
        test_df[
            test_df[TARGET] >= 30_000_000
        ]
        .sort_values(
            TARGET,
            ascending=False,
        )
        .copy()
    )

    display_columns = [
        column
        for column in [
            "player_name",
            "transfer_date",
            "from_club_name",
            "to_club_name",
            "actual_m",
            "predicted_m",
            "error_m",
        ]
        if column in expensive.columns
    ]

    print("\n========================================")
    print("30M+ 선수 개별 예측")
    print("========================================")

    print(
        expensive[
            display_columns
        ]
        .round(2)
        .to_string(
            index=False,
        )
    )


    # ========================================================
    # 10. 가장 심하게 과소예측한 선수 TOP 15
    # ========================================================

    underpredicted = (
        test_df
        .sort_values(
            "error_m",
            ascending=False,
        )
        .head(15)
    )

    print("\n========================================")
    print("과소예측 TOP 15")
    print("========================================")

    under_columns = [
        column
        for column in [
            "player_name",
            "actual_m",
            "predicted_m",
            "error_m",
            "rating",
            "age_at_transfer",
            "main_position",
            "from_league_id",
            "to_league_id",
        ]
        if column in underpredicted.columns
    ]

    print(
        underpredicted[
            under_columns
        ]
        .round(2)
        .to_string(
            index=False,
        )
    )


    # ========================================================
    # 11. Scatter Plot
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        test_df["actual_m"],
        test_df["predicted_m"],
        alpha=0.6,
    )

    max_value = max(
        test_df["actual_m"].max(),
        test_df["predicted_m"].max(),
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
    )

    plt.xlabel(
        "Actual Transfer Fee (Million EUR)"
    )

    plt.ylabel(
        "Predicted Transfer Fee (Million EUR)"
    )

    plt.title(
        "No-Market v1.1 - 2025 Test Set"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = "results/high_value_prediction.png"

    from pathlib import Path
    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\n✓ Scatter Plot 저장 완료: {output_path}"
    )


if __name__ == "__main__":
    main()