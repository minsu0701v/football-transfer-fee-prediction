import copy
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 파일 경로
# ============================================================

DATA_FILE = "data/processed/training_dataset.csv"
BASE_MODEL_FILE = "models/transfer_fee_model_v1_1.joblib"

TARGET = "transfer_fee"
YEAR_FEATURE = "transfer_year"


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
# Pipeline에서 ColumnTransformer 찾기
# ============================================================

def find_column_transformer(pipeline):
    if not hasattr(pipeline, "steps"):
        raise ValueError(
            "불러온 모델이 sklearn Pipeline 형태가 아닙니다."
        )

    for step_name, step in pipeline.steps:
        if isinstance(step, ColumnTransformer):
            return step_name, step

    raise ValueError(
        "Pipeline 안에서 ColumnTransformer를 찾지 못했습니다.\n"
        f"현재 steps: {[name for name, _ in pipeline.steps]}"
    )


# ============================================================
# transfer_year를 numeric feature에 추가한 모델 생성
# ============================================================

def make_year_model(base_model):
    try:
        year_model = clone(base_model)
    except Exception:
        year_model = copy.deepcopy(base_model)

    step_name, preprocessor = find_column_transformer(
        year_model
    )

    new_transformers = []
    added = False

    for name, transformer, columns in preprocessor.transformers:

        # 문자열 drop / passthrough 등은 그대로
        if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            new_columns = list(columns)

            # v1.1 numeric block을 식별
            numeric_markers = {
                "age_at_transfer",
                "height",
                "matches",
                "minutes",
                "rating",
            }

            if numeric_markers.intersection(new_columns):
                if YEAR_FEATURE not in new_columns:
                    new_columns.append(YEAR_FEATURE)
                added = True

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

    if not added:
        raise ValueError(
            "numeric transformer를 찾지 못해 "
            "transfer_year를 추가하지 못했습니다."
        )

    preprocessor.transformers = new_transformers

    return year_model


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

    result_df["predicted_fee"] = predictions
    result_df["absolute_error"] = np.abs(
        result_df[TARGET]
        - result_df["predicted_fee"]
    )

    result_df["actual_m"] = (
        result_df[TARGET] / 1_000_000
    )

    result_df["predicted_m"] = (
        result_df["predicted_fee"] / 1_000_000
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

    def group_value(group_name, column):
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
    # 1. 데이터 / 기존 v1.1 모델 로드
    # --------------------------------------------------------

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

    print("\n========================================")
    print("기존 v1.1 모델 확인")
    print("========================================")

    print("\n기존 FEATURES:")
    print(base_features)

    if "value_at_transfer" in base_features:
        raise ValueError(
            "현재 모델은 value_at_transfer를 사용합니다. "
            "no-market v1.1 모델이 아닙니다."
        )

    print("\n✓ no-market v1.1 확인")


    # --------------------------------------------------------
    # 2. 날짜 / v1.1 feature 생성
    # --------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df[YEAR_FEATURE] = (
        df["transfer_date"].dt.year
    )

    df = add_v11_features(df)

    missing_columns = [
        column
        for column in base_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"다음 feature가 데이터에 없습니다: "
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

    print("\n========================================")
    print("데이터 분할")
    print("========================================")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")

    print(
        "Train year:",
        int(train_df[YEAR_FEATURE].min()),
        "~",
        int(train_df[YEAR_FEATURE].max()),
    )

    print(
        "Test year :",
        int(test_df[YEAR_FEATURE].min()),
    )


    # --------------------------------------------------------
    # 4. Baseline v1.1 준비
    # --------------------------------------------------------

    try:
        baseline_model = clone(
            base_model
        )
    except Exception:
        baseline_model = copy.deepcopy(
            base_model
        )

    X_train_base = train_df[
        base_features
    ]

    X_test_base = test_df[
        base_features
    ]


    # --------------------------------------------------------
    # 5. + transfer_year 모델 준비
    # --------------------------------------------------------

    year_model = make_year_model(
        base_model
    )

    year_features = (
        base_features
        + [YEAR_FEATURE]
    )

    X_train_year = train_df[
        year_features
    ]

    X_test_year = test_df[
        year_features
    ]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print("\n+Year FEATURES:")
    print(year_features)


    # --------------------------------------------------------
    # 6. Baseline v1.1 재학습
    # --------------------------------------------------------

    print("\n========================================")
    print("1/2 Baseline v1.1 학습")
    print("========================================")

    baseline_model.fit(
        X_train_base,
        np.log1p(y_train),
    )

    baseline_pred = np.expm1(
        baseline_model.predict(
            X_test_base
        )
    )

    baseline_pred = np.maximum(
        baseline_pred,
        0,
    )

    print("✓ Baseline 예측 완료")


    # --------------------------------------------------------
    # 7. + transfer_year 학습
    # --------------------------------------------------------

    print("\n========================================")
    print("2/2 + transfer_year 학습")
    print("========================================")

    year_model.fit(
        X_train_year,
        np.log1p(y_train),
    )

    year_pred = np.expm1(
        year_model.predict(
            X_test_year
        )
    )

    year_pred = np.maximum(
        year_pred,
        0,
    )

    print("✓ +Year 예측 완료")


    # --------------------------------------------------------
    # 8. 평가
    # --------------------------------------------------------

    base_summary, base_groups, _ = evaluate_model(
        "v1.1 Baseline",
        y_test,
        baseline_pred,
        test_df,
    )

    year_summary, year_groups, _ = evaluate_model(
        "v1.1 + transfer_year",
        y_test,
        year_pred,
        test_df,
    )

    summary_df = pd.DataFrame(
        [
            base_summary,
            year_summary,
        ]
    )


    # --------------------------------------------------------
    # 9. 핵심 비교
    # --------------------------------------------------------

    display_columns = [
        "model",
        "MAE_M",
        "RMSE_M",
        "R2",
        "Top10_MAE_M",
        "30~50M_Pred_Mean_M",
        "30~50M_MAE_M",
        "50M+_Pred_Mean_M",
        "50M+_MAE_M",
        "50M+_Mean_Error_M",
    ]

    print("\n========================================")
    print("★ v1.1 vs + transfer_year")
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
    # 10. 가격 구간 상세
    # --------------------------------------------------------

    print("\n========================================")
    print("Baseline - 가격 구간별")
    print("========================================")

    print(
        base_groups
        .round(2)
        .to_string()
    )

    print("\n========================================")
    print("+ transfer_year - 가격 구간별")
    print("========================================")

    print(
        year_groups
        .round(2)
        .to_string()
    )


    # --------------------------------------------------------
    # 11. 50M+ 선수 개별 비교
    # --------------------------------------------------------

    compare_df = test_df.copy()

    compare_df["actual_m"] = (
        compare_df[TARGET] / 1_000_000
    )

    compare_df["baseline_pred_m"] = (
        baseline_pred / 1_000_000
    )

    compare_df["year_pred_m"] = (
        year_pred / 1_000_000
    )

    compare_df["baseline_error_m"] = (
        compare_df["actual_m"]
        - compare_df["baseline_pred_m"]
    )

    compare_df["year_error_m"] = (
        compare_df["actual_m"]
        - compare_df["year_pred_m"]
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
            "year_pred_m",
            "baseline_error_m",
            "year_error_m",
        ]
        if column in expensive.columns
    ]

    print("\n========================================")
    print("50M+ 선수 Baseline vs +Year")
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

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_dir / "v11_vs_transfer_year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expensive[
        player_columns
    ].to_csv(
        output_dir / "v11_vs_transfer_year_50m_players.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========================================")
    print("저장 완료")
    print("========================================")

    print(
        "results/v11_vs_transfer_year_summary.csv"
    )

    print(
        "results/v11_vs_transfer_year_50m_players.csv"
    )

    print(
        "\n주의: XGBoost는 2020~2024의 연도 패턴을 "
        "2025 이후로 선형 외삽하지는 않습니다."
    )


if __name__ == "__main__":
    main()