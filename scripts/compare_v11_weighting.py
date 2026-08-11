import copy
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
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

    df["age_squared"] = (
        df["age_at_transfer"] ** 2
    )

    return df


# ============================================================
# Pipeline 마지막 estimator step 이름 찾기
# ============================================================

def get_estimator_step_name(model):
    if not hasattr(model, "steps"):
        raise ValueError(
            "현재 모델이 sklearn Pipeline 형태가 아닙니다."
        )

    if len(model.steps) == 0:
        raise ValueError(
            "Pipeline steps가 비어 있습니다."
        )

    return model.steps[-1][0]


# ============================================================
# Weight 생성
# ============================================================

def make_weights(y, scheme):
    """
    scheme:
      baseline
      weight_a
      weight_b
    """

    weights = np.ones(
        len(y),
        dtype=float,
    )

    y_array = np.asarray(y)

    if scheme == "baseline":
        return weights

    if scheme == "weight_a":
        # 30M~50M : 1.5
        # 50M+    : 2.0
        weights[
            y_array >= 30_000_000
        ] = 1.5

        weights[
            y_array >= 50_000_000
        ] = 2.0

        return weights

    if scheme == "weight_b":
        # 30M~50M : 2.0
        # 50M+    : 3.0
        weights[
            y_array >= 30_000_000
        ] = 2.0

        weights[
            y_array >= 50_000_000
        ] = 3.0

        return weights

    if scheme == "weight_c":
        # 30M~50M : 2.5
        # 50M+    : 4.0
        weights[
            y_array >= 30_000_000
        ] = 2.5

        weights[
            y_array >= 50_000_000
        ] = 4.0

        return weights

    if scheme == "weight_d":
        # 30M~50M : 3.0
        # 50M+    : 5.0
        weights[
            y_array >= 30_000_000
        ] = 3.0

        weights[
            y_array >= 50_000_000
        ] = 5.0

        return weights

    raise ValueError(
        f"알 수 없는 weighting scheme: {scheme}"
    )


# ============================================================
# 모델 학습
# ============================================================

def fit_model(
    base_model,
    X_train,
    y_train,
    weights=None,
):
    try:
        model = clone(
            base_model
        )
    except Exception:
        model = copy.deepcopy(
            base_model
        )

    y_train_log = np.log1p(
        y_train
    )

    if weights is None:
        model.fit(
            X_train,
            y_train_log,
        )
    else:
        estimator_step = (
            get_estimator_step_name(
                model
            )
        )

        fit_params = {
            f"{estimator_step}__sample_weight": weights
        }

        model.fit(
            X_train,
            y_train_log,
            **fit_params,
        )

    return model


# ============================================================
# 평가
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    test_df,
):
    result_df = test_df.copy()

    result_df["predicted_fee"] = (
        predictions
    )

    result_df["absolute_error"] = np.abs(
        result_df[TARGET]
        - result_df["predicted_fee"]
    )

    result_df["actual_m"] = (
        result_df[TARGET] / 1_000_000
    )

    result_df["predicted_m"] = (
        result_df["predicted_fee"]
        / 1_000_000
    )

    result_df["error_m"] = (
        result_df[TARGET]
        - result_df["predicted_fee"]
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

    q80 = result_df[TARGET].quantile(
        0.80
    )

    q90 = result_df[TARGET].quantile(
        0.90
    )

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

    def group_value(
        group_name,
        column,
    ):
        if group_name not in group_result.index:
            return np.nan

        return group_result.loc[
            group_name,
            column,
        ]

    summary = {
        "model": model_name,
        "MAE_M": mae,
        "RMSE_M": rmse,
        "R2": r2,
        "Top20_MAE_M": top20_mae,
        "Top10_MAE_M": top10_mae,
        "0~10M_MAE_M": group_value(
            "0~10M",
            "mae",
        ),
        "10~30M_MAE_M": group_value(
            "10~30M",
            "mae",
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

    return (
        summary,
        group_result,
        result_df,
    )


# ============================================================
# 메인
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 데이터 / 기존 v1.1 모델 로드
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

    estimator_step = (
        get_estimator_step_name(
            base_model
        )
    )

    print(
        f"✓ estimator step: {estimator_step}"
    )


    # --------------------------------------------------------
    # 2. 날짜 / Feature Engineering
    # --------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df = add_v11_features(
        df
    )

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"다음 feature가 없습니다: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # 3. 동일 시간 분할
    # --------------------------------------------------------

    train_df = df[
        df["transfer_date"].dt.year < 2025
    ].copy()

    test_df = df[
        df["transfer_date"].dt.year == 2025
    ].copy()

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    print("\n========================================")
    print("데이터 분할")
    print("========================================")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")


    # --------------------------------------------------------
    # 4. Weight 분포 확인
    # --------------------------------------------------------

    baseline_weights = make_weights(
        y_train,
        "baseline",
    )

    weight_a = make_weights(
        y_train,
        "weight_a",
    )

    weight_b = make_weights(
        y_train,
        "weight_b",
    )

    weight_c = make_weights(
        y_train,
        "weight_c",
    )

    weight_d = make_weights(
        y_train,
        "weight_d",
    )

    print("\n========================================")
    print("Weight 설정")
    print("========================================")

    print(
        "Baseline: 모든 샘플 1.0"
    )

    print(
        "Weight A : <30M=1.0 / "
        "30~50M=1.5 / 50M+=2.0"
    )

    print(
        "Weight B : <30M=1.0 / "
        "30~50M=2.0 / 50M+=3.0"
    )

    print(
        "Weight C : <30M=1.0 / "
        "30~50M=2.5 / 50M+=4.0"
    )

    print(
        "Weight D : <30M=1.0 / "
        "30~50M=3.0 / 50M+=5.0"
    )

    print("\nTrain 구간별 샘플 수:")

    print(
        f"<30M     : "
        f"{(y_train < 30_000_000).sum()}"
    )

    print(
        f"30~50M   : "
        f"{((y_train >= 30_000_000) & (y_train < 50_000_000)).sum()}"
    )

    print(
        f"50M+     : "
        f"{(y_train >= 50_000_000).sum()}"
    )


    # --------------------------------------------------------
    # 5. Baseline
    # --------------------------------------------------------

    print("\n========================================")
    print("1/5 Baseline 학습")
    print("========================================")

    baseline_model = fit_model(
        base_model,
        X_train,
        y_train,
        weights=None,
    )

    baseline_pred = np.expm1(
        baseline_model.predict(
            X_test
        )
    )

    baseline_pred = np.maximum(
        baseline_pred,
        0,
    )

    print("✓ Baseline 완료")


    # --------------------------------------------------------
    # 6. Weight A
    # --------------------------------------------------------

    print("\n========================================")
    print("2/5 Weight A 학습")
    print("========================================")

    model_a = fit_model(
        base_model,
        X_train,
        y_train,
        weights=weight_a,
    )

    pred_a = np.expm1(
        model_a.predict(
            X_test
        )
    )

    pred_a = np.maximum(
        pred_a,
        0,
    )

    print("✓ Weight A 완료")


    # --------------------------------------------------------
    # 7. Weight B
    # --------------------------------------------------------

    print("\n========================================")
    print("3/5 Weight B 학습")
    print("========================================")

    model_b = fit_model(
        base_model,
        X_train,
        y_train,
        weights=weight_b,
    )

    pred_b = np.expm1(
        model_b.predict(
            X_test
        )
    )

    pred_b = np.maximum(
        pred_b,
        0,
    )

    print("✓ Weight B 완료")


    # --------------------------------------------------------
    # 8. Weight C
    # --------------------------------------------------------

    print("\n========================================")
    print("4/5 Weight C 학습")
    print("========================================")

    model_c = fit_model(
        base_model,
        X_train,
        y_train,
        weights=weight_c,
    )

    pred_c = np.expm1(
        model_c.predict(
            X_test
        )
    )

    pred_c = np.maximum(
        pred_c,
        0,
    )

    print("✓ Weight C 완료")


    # --------------------------------------------------------
    # 9. Weight D
    # --------------------------------------------------------

    print("\n========================================")
    print("5/5 Weight D 학습")
    print("========================================")

    model_d = fit_model(
        base_model,
        X_train,
        y_train,
        weights=weight_d,
    )

    pred_d = np.expm1(
        model_d.predict(
            X_test
        )
    )

    pred_d = np.maximum(
        pred_d,
        0,
    )

    print("✓ Weight D 완료")


    # --------------------------------------------------------
    # 10. 평가
    # --------------------------------------------------------

    baseline_summary, baseline_groups, _ = evaluate_model(
        "Baseline",
        y_test,
        baseline_pred,
        test_df,
    )

    a_summary, a_groups, _ = evaluate_model(
        "Weight A",
        y_test,
        pred_a,
        test_df,
    )

    b_summary, b_groups, _ = evaluate_model(
        "Weight B",
        y_test,
        pred_b,
        test_df,
    )

    c_summary, c_groups, _ = evaluate_model(
        "Weight C",
        y_test,
        pred_c,
        test_df,
    )

    d_summary, d_groups, _ = evaluate_model(
        "Weight D",
        y_test,
        pred_d,
        test_df,
    )

    summary_df = pd.DataFrame(
        [
            baseline_summary,
            a_summary,
            b_summary,
            c_summary,
            d_summary,
        ]
    )


    # --------------------------------------------------------
    # 9. 핵심 비교표
    # --------------------------------------------------------

    display_columns = [
        "model",
        "MAE_M",
        "RMSE_M",
        "R2",
        "Top10_MAE_M",
        "0~10M_MAE_M",
        "10~30M_MAE_M",
        "30~50M_Pred_Mean_M",
        "30~50M_MAE_M",
        "50M+_Pred_Mean_M",
        "50M+_MAE_M",
        "50M+_Mean_Error_M",
    ]

    print("\n========================================")
    print("★ Baseline vs Weight A/B/C/D")
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
    # 10. 가격 구간별 상세
    # --------------------------------------------------------

    for model_name, groups in [
        ("Baseline", baseline_groups),
        ("Weight A", a_groups),
        ("Weight B", b_groups),
        ("Weight C", c_groups),
        ("Weight D", d_groups),
    ]:
        print("\n========================================")
        print(f"{model_name} - 가격 구간별")
        print("========================================")

        print(
            groups
            .round(2)
            .to_string()
        )


    # --------------------------------------------------------
    # 11. 50M+ 선수 개별 비교
    # --------------------------------------------------------

    compare_df = test_df.copy()

    compare_df["actual_m"] = (
        compare_df[TARGET]
        / 1_000_000
    )

    compare_df["baseline_pred_m"] = (
        baseline_pred
        / 1_000_000
    )

    compare_df["weight_a_pred_m"] = (
        pred_a
        / 1_000_000
    )

    compare_df["weight_b_pred_m"] = (
        pred_b
        / 1_000_000
    )

    compare_df["weight_c_pred_m"] = (
        pred_c
        / 1_000_000
    )

    compare_df["weight_d_pred_m"] = (
        pred_d
        / 1_000_000
    )

    compare_df["baseline_error_m"] = (
        compare_df["actual_m"]
        - compare_df["baseline_pred_m"]
    )

    compare_df["weight_a_error_m"] = (
        compare_df["actual_m"]
        - compare_df["weight_a_pred_m"]
    )

    compare_df["weight_b_error_m"] = (
        compare_df["actual_m"]
        - compare_df["weight_b_pred_m"]
    )

    compare_df["weight_c_error_m"] = (
        compare_df["actual_m"]
        - compare_df["weight_c_pred_m"]
    )

    compare_df["weight_d_error_m"] = (
        compare_df["actual_m"]
        - compare_df["weight_d_pred_m"]
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
            "baseline_pred_m",
            "weight_a_pred_m",
            "weight_b_pred_m",
            "weight_c_pred_m",
            "weight_d_pred_m",
            "baseline_error_m",
            "weight_a_error_m",
            "weight_b_error_m",
            "weight_c_error_m",
            "weight_d_error_m",
        ]
        if column in expensive.columns
    ]

    print("\n========================================")
    print("50M+ 선수 개별 비교")
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
    # 12. 결과 저장
    # --------------------------------------------------------

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_dir / "v11_weighting_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expensive[
        player_columns
    ].to_csv(
        output_dir / "v11_weighting_50m_players.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========================================")
    print("저장 완료")
    print("========================================")

    print(
        "results/v11_weighting_summary.csv"
    )

    print(
        "results/v11_weighting_50m_players.csv"
    )


if __name__ == "__main__":
    main()