from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transfer_fee_model_v1_1.joblib"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v1_1.csv"
)


# ============================================================
# 유틸리티
# ============================================================

def clean_feature_name(feature_name: str) -> str:
    """
    ColumnTransformer가 붙인 prefix 제거.
    """
    return (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 모델 / 데이터 로드
# ============================================================

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다.\n"
            f"{MODEL_FILE}"
        )

    pipeline = joblib.load(MODEL_FILE)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    print("모델 로드 완료")

    return pipeline, preprocessor, model


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다.\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    return df


# ============================================================
# 선수 선택
# ============================================================

def select_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    우선 테스트 목적으로 첫 번째 선수 1명 사용.
    나중에 player_name 검색 방식으로 바꿀 수 있음.
    """

    sample = df.iloc[[0]].copy()

    print_section("선택 선수")

    display_columns = [
        "player_name",
        "transfer_date",
        "from_team_name",
        "to_team_name",
        "transfer_fee",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in sample.columns
    ]

    print(
        sample[
            available_columns
        ].to_string(index=False)
    )

    return sample


# ============================================================
# SHAP Contribution 계산
# ============================================================

def calculate_contributions(
    pipeline,
    preprocessor,
    model,
    sample: pd.DataFrame,
):
    # 모델이 실제로 기대하는 원본 Feature 목록
    feature_names_before = (
        pipeline.feature_names_in_
    )

    sample_features = sample[
        feature_names_before
    ].copy()

    # 전처리
    transformed = preprocessor.transform(
        sample_features
    )

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    transformed_feature_names = [
        clean_feature_name(name)
        for name in transformed_feature_names
    ]

    print()
    print(
        f"전처리 후 Feature 수: "
        f"{len(transformed_feature_names)}"
    )

    # --------------------------------------------------------
    # XGBoost 자체 SHAP contribution 계산
    # --------------------------------------------------------

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        transformed,
        feature_names=transformed_feature_names,
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
    )

    # 마지막 값은 bias/base value
    shap_values = contributions[
        0,
        :-1
    ]

    base_value = contributions[
        0,
        -1
    ]

    return (
        sample_features,
        transformed_feature_names,
        shap_values,
        base_value,
    )


# ============================================================
# 예측값 계산
# ============================================================

def calculate_prediction(
    pipeline,
    sample_features: pd.DataFrame,
) -> tuple[float, float]:
    """
    모델은 log1p(transfer_fee)를 학습했으므로
    pipeline.predict() 결과도 log 공간 값.
    """

    predicted_log = pipeline.predict(
        sample_features
    )[0]

    predicted_fee = np.expm1(
        predicted_log
    )

    predicted_fee = max(
        predicted_fee,
        0,
    )

    return (
        float(predicted_log),
        float(predicted_fee),
    )


# ============================================================
# 결과 테이블 생성
# ============================================================

def create_explanation_table(
    feature_names: list[str],
    shap_values: np.ndarray,
) -> pd.DataFrame:
    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values,
        }
    )

    explanation[
        "absolute_shap"
    ] = explanation[
        "shap_value"
    ].abs()

    explanation = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    explanation[
        "direction"
    ] = np.where(
        explanation["shap_value"] >= 0,
        "UP",
        "DOWN",
    )

    return explanation


# ============================================================
# 출력
# ============================================================

def print_explanation(
    explanation: pd.DataFrame,
    base_value: float,
    predicted_log: float,
    predicted_fee: float,
    top_n: int = 15,
) -> None:
    print_section("예측 결과")

    print(
        f"Base value (log scale) : "
        f"{base_value:.6f}"
    )

    print(
        f"Predicted log fee      : "
        f"{predicted_log:.6f}"
    )

    print(
        f"Predicted transfer fee : "
        f"€{predicted_fee:,.0f}"
    )

    print_section(
        f"Top {top_n} SHAP Contributions"
    )

    display = (
        explanation
        .head(top_n)
        .copy()
    )

    for _, row in display.iterrows():
        symbol = (
            "+"
            if row["shap_value"] >= 0
            else "-"
        )

        print(
            f"{row['feature']:<40} "
            f"{symbol}"
            f"{abs(row['shap_value']):.6f}"
        )


# ============================================================
# JSON 형태 결과 생성
# ============================================================

def create_json_result(
    sample: pd.DataFrame,
    explanation: pd.DataFrame,
    predicted_fee: float,
    top_n: int = 10,
) -> dict:
    player_name = (
        sample["player_name"].iloc[0]
        if "player_name" in sample.columns
        else None
    )

    top_features = []

    for _, row in (
        explanation
        .head(top_n)
        .iterrows()
    ):
        top_features.append(
            {
                "feature": row["feature"],
                "shap_value": float(
                    row["shap_value"]
                ),
                "direction": row[
                    "direction"
                ],
            }
        )

    result = {
        "player_name": player_name,
        "predicted_transfer_fee": (
            predicted_fee
        ),
        "top_factors": top_features,
    }

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:
    print_section(
        "Transfer Fee Prediction Explanation V1.1"
    )

    (
        pipeline,
        preprocessor,
        model,
    ) = load_model()

    df = load_data()

    sample = select_sample(
        df
    )

    (
        sample_features,
        feature_names,
        shap_values,
        base_value,
    ) = calculate_contributions(
        pipeline=pipeline,
        preprocessor=preprocessor,
        model=model,
        sample=sample,
    )

    (
        predicted_log,
        predicted_fee,
    ) = calculate_prediction(
        pipeline=pipeline,
        sample_features=sample_features,
    )

    explanation = (
        create_explanation_table(
            feature_names=feature_names,
            shap_values=shap_values,
        )
    )

    print_explanation(
        explanation=explanation,
        base_value=base_value,
        predicted_log=predicted_log,
        predicted_fee=predicted_fee,
        top_n=15,
    )

    result = create_json_result(
        sample=sample,
        explanation=explanation,
        predicted_fee=predicted_fee,
        top_n=10,
    )

    print_section("API용 결과 예시")

    print(result)

    print()
    print(
        "※ SHAP 값은 log1p(transfer_fee) 공간의 "
        "기여도입니다."
    )

    print(
        "※ 아직 '+€5M' 형태의 금액 영향도로 "
        "해석하면 안 됩니다."
    )


if __name__ == "__main__":
    main()