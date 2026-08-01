from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# =========================================================
# 기본 설정
# =========================================================

DATA_FILE = Path("data/processed/training_dataset.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("data/results")

MODEL_A_FILE = MODEL_DIR / "xgb_without_market.pkl"
MODEL_B_FILE = MODEL_DIR / "xgb_with_market.pkl"

COMPARISON_FILE = RESULT_DIR / "xgboost_comparison.csv"
PREDICTION_FILE = RESULT_DIR / "xgboost_predictions.csv"
IMPORTANCE_A_FILE = RESULT_DIR / "xgb_without_market_importance.csv"
IMPORTANCE_B_FILE = RESULT_DIR / "xgb_with_market_importance.csv"

TARGET = "transfer_fee"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================================================
# Feature 설정
# =========================================================

BASE_NUMERIC_FEATURES = [
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "height",
    "age_at_transfer",
    "is_same_league",
    "is_top5_destination",
]

CATEGORICAL_FEATURES = [
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]

MODEL_A_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES.copy()

MODEL_B_NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + ["value_at_transfer"]
)


# =========================================================
# 데이터 불러오기
# =========================================================

def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"학습 데이터가 없습니다: {DATA_FILE}\n"
            "먼저 create_training_dataset.py를 실행하세요."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = set(
        MODEL_B_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "training_dataset.csv에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    print("[데이터 불러오기]")
    print(f"행 수: {len(data):,}")
    print(f"컬럼 수: {len(data.columns)}")

    return data


# =========================================================
# 데이터 정리
# =========================================================

def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()

    numeric_columns = list(
        dict.fromkeys(
            MODEL_B_NUMERIC_FEATURES
            + [TARGET]
        )
    )

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    missing_target_count = data[TARGET].isna().sum()

    if missing_target_count > 0:
        print(
            f"\ntransfer_fee 결측 행 "
            f"{missing_target_count:,}건을 제거합니다."
        )

        data = data.dropna(subset=[TARGET])

    non_positive_target_count = (
        data[TARGET] <= 0
    ).sum()

    if non_positive_target_count > 0:
        print(
            f"\ntransfer_fee가 0 이하인 행 "
            f"{non_positive_target_count:,}건을 제거합니다."
        )

        data = data[
            data[TARGET] > 0
        ]

    data = data.reset_index(drop=True)

    print("\n[정리된 데이터]")
    print(f"최종 행 수: {len(data):,}")
    print(
        f"이적료 중앙값: "
        f"€{data[TARGET].median():,.0f}"
    )
    print(
        f"이적료 평균값: "
        f"€{data[TARGET].mean():,.0f}"
    )

    return data


# =========================================================
# Train / Test 분리
# =========================================================

def split_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    feature_columns = list(
        dict.fromkeys(
            MODEL_B_NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        )
    )

    X = data[feature_columns].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print("\n[Train / Test 분리]")
    print(f"Train: {len(X_train):,}건")
    print(f"Test : {len(X_test):,}건")

    print("\n[Target 분포]")
    print(
        f"Train 중앙값: "
        f"€{y_train.median():,.0f}"
    )
    print(
        f"Test 중앙값 : "
        f"€{y_test.median():,.0f}"
    )
    print(
        f"Train 평균값: "
        f"€{y_train.mean():,.0f}"
    )
    print(
        f"Test 평균값 : "
        f"€{y_test.mean():,.0f}"
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# 전처리기
# =========================================================

def create_preprocessor(
    numeric_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# =========================================================
# XGBoost Pipeline
# =========================================================

def create_xgboost_pipeline(
    numeric_features: list[str],
) -> Pipeline:
    preprocessor = create_preprocessor(
        numeric_features=numeric_features,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# =========================================================
# 평가 함수
# =========================================================

def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    y_train_log = np.log1p(y_train)

    pipeline.fit(
        X_train,
        y_train_log,
    )

    prediction_log = pipeline.predict(X_test)

    prediction_log = np.clip(
        prediction_log,
        a_min=0,
        a_max=None,
    )

    predictions = np.expm1(
        prediction_log
    )

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

    actual_values = y_test.to_numpy()

    absolute_error_rates = (
        np.abs(
            actual_values - predictions
        )
        / actual_values
        * 100
    )

    median_error_rate = np.median(
        absolute_error_rates
    )

    result = {
        "model": model_name,
        "uses_market_value": (
            "포함"
            if "With Market" in model_name
            else "제외"
        ),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_error_rate": median_error_rate,
    }

    return result, predictions


# =========================================================
# 결과 출력
# =========================================================

def print_model_result(
    result: dict,
) -> None:
    print(f"\n[{result['model']}]")
    print(
        "value_at_transfer: "
        f"{result['uses_market_value']}"
    )
    print(f"MAE  : €{result['mae']:,.0f}")
    print(f"RMSE : €{result['rmse']:,.0f}")
    print(f"R²   : {result['r2']:.4f}")
    print(
        "중앙값 기준 절대 오차율: "
        f"{result['median_error_rate']:.2f}%"
    )


def print_model_comparison(
    results: list[dict],
) -> pd.DataFrame:
    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "model",
            "uses_market_value",
            "mae",
            "rmse",
            "r2",
            "median_error_rate",
        ]
    ]

    comparison = comparison.sort_values(
        by="r2",
        ascending=False,
    ).reset_index(drop=True)

    display_data = comparison.copy()

    display_data["mae"] = (
        display_data["mae"]
        .map(lambda value: f"€{value:,.0f}")
    )

    display_data["rmse"] = (
        display_data["rmse"]
        .map(lambda value: f"€{value:,.0f}")
    )

    display_data["r2"] = (
        display_data["r2"]
        .map(lambda value: f"{value:.4f}")
    )

    display_data["median_error_rate"] = (
        display_data["median_error_rate"]
        .map(lambda value: f"{value:.2f}%")
    )

    print("\n[XGBoost A/B 성능 비교]")
    print(
        display_data.to_string(
            index=False,
        )
    )

    return comparison


# =========================================================
# Feature Importance
# =========================================================

def clean_feature_name(
    feature_name: str,
) -> str:
    return (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )


def get_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importance_values = (
        model.feature_importances_
    )

    if len(feature_names) != len(
        importance_values
    ):
        raise ValueError(
            "Feature 이름 수와 Importance 수가 "
            "일치하지 않습니다."
        )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_table["feature"] = (
        importance_table["feature"]
        .map(clean_feature_name)
    )

    importance_table = (
        importance_table
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_table["importance_percent"] = (
        importance_table["importance"]
        * 100
    )

    return importance_table


def print_feature_importance(
    model_name: str,
    importance_table: pd.DataFrame,
    top_n: int = 20,
) -> None:
    display_data = (
        importance_table
        .head(top_n)
        .copy()
    )

    display_data["importance"] = (
        display_data["importance"]
        .map(lambda value: f"{value:.6f}")
    )

    display_data["importance_percent"] = (
        display_data["importance_percent"]
        .map(lambda value: f"{value:.2f}%")
    )

    print(
        f"\n[{model_name} 주요 Feature Importance]"
    )

    print(
        display_data[
            [
                "feature",
                "importance",
                "importance_percent",
            ]
        ].to_string(index=False)
    )


# =========================================================
# 실제값 / 예측값 비교
# =========================================================

def create_prediction_table(
    data: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
) -> pd.DataFrame:
    prediction_table = pd.DataFrame(
        {
            "actual_fee": y_test.to_numpy(),
            "xgb_without_market_prediction": (
                predictions_a
            ),
            "xgb_with_market_prediction": (
                predictions_b
            ),
        },
        index=y_test.index,
    )

    prediction_table[
        "without_market_absolute_error"
    ] = np.abs(
        prediction_table["actual_fee"]
        - prediction_table[
            "xgb_without_market_prediction"
        ]
    )

    prediction_table[
        "with_market_absolute_error"
    ] = np.abs(
        prediction_table["actual_fee"]
        - prediction_table[
            "xgb_with_market_prediction"
        ]
    )

    prediction_table[
        "without_market_error_rate"
    ] = (
        prediction_table[
            "without_market_absolute_error"
        ]
        / prediction_table["actual_fee"]
        * 100
    )

    prediction_table[
        "with_market_error_rate"
    ] = (
        prediction_table[
            "with_market_absolute_error"
        ]
        / prediction_table["actual_fee"]
        * 100
    )

    info_columns = [
        "player_id",
        "player_name",
        "transfer_date",
        "from_team_name",
        "to_team_name",
        "from_league_id",
        "to_league_id",
        "main_position",
        "age_at_transfer",
        "rating",
        "goals",
        "assists",
        "value_at_transfer",
    ]

    available_info_columns = [
        column
        for column in info_columns
        if column in data.columns
    ]

    prediction_table = prediction_table.join(
        data.loc[
            X_test.index,
            available_info_columns,
        ]
    )

    prediction_table = (
        prediction_table
        .sort_values(
            by="actual_fee",
            ascending=False,
        )
    )

    return prediction_table


def print_prediction_sample(
    prediction_table: pd.DataFrame,
    row_count: int = 15,
) -> None:
    display_columns = [
        "player_name",
        "actual_fee",
        "value_at_transfer",
        "xgb_without_market_prediction",
        "without_market_error_rate",
        "xgb_with_market_prediction",
        "with_market_error_rate",
        "age_at_transfer",
        "main_position",
        "from_league_id",
        "to_league_id",
        "rating",
        "goals",
        "assists",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in prediction_table.columns
    ]

    display_data = (
        prediction_table[
            display_columns
        ]
        .head(row_count)
        .copy()
    )

    money_columns = [
        "actual_fee",
        "value_at_transfer",
        "xgb_without_market_prediction",
        "xgb_with_market_prediction",
    ]

    for column in money_columns:
        if column in display_data.columns:
            display_data[column] = (
                display_data[column]
                .map(
                    lambda value: (
                        f"€{value:,.0f}"
                        if pd.notna(value)
                        else "NaN"
                    )
                )
            )

    percentage_columns = [
        "without_market_error_rate",
        "with_market_error_rate",
    ]

    for column in percentage_columns:
        if column in display_data.columns:
            display_data[column] = (
                display_data[column]
                .map(
                    lambda value: f"{value:.1f}%"
                )
            )

    print("\n[실제값과 XGBoost 예측값 샘플]")
    print(
        display_data.to_string(
            index=False,
        )
    )


# =========================================================
# 모델 저장
# =========================================================

def save_model(
    pipeline: Pipeline,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        output_file,
    )

    print(f"모델 저장 완료: {output_file}")


def save_results(
    comparison: pd.DataFrame,
    prediction_table: pd.DataFrame,
    importance_a: pd.DataFrame,
    importance_b: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_table.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    importance_a.to_csv(
        IMPORTANCE_A_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    importance_b.to_csv(
        IMPORTANCE_B_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[결과 파일 저장]")
    print(COMPARISON_FILE)
    print(PREDICTION_FILE)
    print(IMPORTANCE_A_FILE)
    print(IMPORTANCE_B_FILE)


# =========================================================
# Main
# =========================================================

def main() -> None:
    data = load_data()
    data = prepare_data(data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(data)

    print("\n" + "=" * 60)
    print("XGBoost A 학습")
    print("value_at_transfer 제외")
    print("=" * 60)

    model_a = create_xgboost_pipeline(
        numeric_features=MODEL_A_NUMERIC_FEATURES,
    )

    result_a, predictions_a = evaluate_model(
        model_name="XGBoost Without Market",
        pipeline=model_a,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print_model_result(result_a)

    print("\n" + "=" * 60)
    print("XGBoost B 학습")
    print("value_at_transfer 포함")
    print("=" * 60)

    model_b = create_xgboost_pipeline(
        numeric_features=MODEL_B_NUMERIC_FEATURES,
    )

    result_b, predictions_b = evaluate_model(
        model_name="XGBoost With Market",
        pipeline=model_b,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print_model_result(result_b)

    comparison = print_model_comparison(
        results=[
            result_a,
            result_b,
        ]
    )

    importance_a = get_feature_importance(
        pipeline=model_a,
    )

    importance_b = get_feature_importance(
        pipeline=model_b,
    )

    print_feature_importance(
        model_name="XGBoost A - 시장가치 제외",
        importance_table=importance_a,
        top_n=20,
    )

    print_feature_importance(
        model_name="XGBoost B - 시장가치 포함",
        importance_table=importance_b,
        top_n=20,
    )

    prediction_table = create_prediction_table(
        data=data,
        X_test=X_test,
        y_test=y_test,
        predictions_a=predictions_a,
        predictions_b=predictions_b,
    )

    print_prediction_sample(
        prediction_table=prediction_table,
        row_count=15,
    )

    save_model(
        pipeline=model_a,
        output_file=MODEL_A_FILE,
    )

    save_model(
        pipeline=model_b,
        output_file=MODEL_B_FILE,
    )

    save_results(
        comparison=comparison,
        prediction_table=prediction_table,
        importance_a=importance_a,
        importance_b=importance_b,
    )

    print("\n" + "=" * 60)
    print("XGBoost A/B 학습 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()