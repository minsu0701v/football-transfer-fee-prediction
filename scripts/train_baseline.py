from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("data/processed/training_dataset.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "transfer_fee"


NUMERIC_FEATURES = [
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


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"학습 데이터가 없습니다: {DATA_FILE}\n"
            "먼저 create_training_dataset.py를 실행하세요."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "학습 데이터에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    print("[데이터 불러오기]")
    print(f"행 수: {len(data):,}")
    print(f"컬럼 수: {len(data.columns)}")

    return data


def prepare_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    data = data.copy()

    for column in NUMERIC_FEATURES + [TARGET]:
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

    non_positive_target_count = (data[TARGET] <= 0).sum()

    if non_positive_target_count > 0:
        print(
            f"\ntransfer_fee가 0 이하인 행 "
            f"{non_positive_target_count:,}건을 제거합니다."
        )

        data = data[data[TARGET] > 0]

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = data[feature_columns].copy()
    y = data[TARGET].copy()

    print("\n[사용 Feature]")
    print(f"수치형: {len(NUMERIC_FEATURES)}개")
    print(f"범주형: {len(CATEGORICAL_FEATURES)}개")
    print("value_at_transfer: 제외")

    print("\n[Target]")
    print("transfer_fee")
    print("학습 시 log1p 변환")

    return X, y


def create_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
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
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def create_models() -> dict[str, Pipeline]:
    dummy_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                DummyRegressor(strategy="median"),
            ),
        ]
    )

    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    return {
        "DummyRegressor": dummy_model,
        "LinearRegression": linear_model,
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    y_train_log = np.log1p(y_train)

    model.fit(
        X_train,
        y_train_log,
    )

    prediction_log = model.predict(X_test)

    prediction_log = np.clip(
        prediction_log,
        a_min=0,
        a_max=None,
    )

    predictions = np.expm1(prediction_log)

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

    median_error_rate = np.median(
        np.abs(
            (
                y_test.to_numpy()
                - predictions
            )
            / y_test.to_numpy()
        )
    ) * 100

    result = {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_error_rate": median_error_rate,
    }

    return result, predictions


def print_model_result(result: dict) -> None:
    print(f"\n[{result['model']}]")
    print(f"MAE  : €{result['mae']:,.0f}")
    print(f"RMSE : €{result['rmse']:,.0f}")
    print(f"R²   : {result['r2']:.4f}")
    print(
        "중앙값 기준 절대 오차율: "
        f"{result['median_error_rate']:.2f}%"
    )


def print_model_comparison(
    results: list[dict],
) -> None:
    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "model",
            "mae",
            "rmse",
            "r2",
            "median_error_rate",
        ]
    ]

    formatted = comparison.copy()

    formatted["mae"] = formatted["mae"].map(
        lambda value: f"€{value:,.0f}"
    )

    formatted["rmse"] = formatted["rmse"].map(
        lambda value: f"€{value:,.0f}"
    )

    formatted["r2"] = formatted["r2"].map(
        lambda value: f"{value:.4f}"
    )

    formatted["median_error_rate"] = (
        formatted["median_error_rate"]
        .map(lambda value: f"{value:.2f}%")
    )

    print("\n[모델 성능 비교]")
    print(formatted.to_string(index=False))


def clean_feature_name(
    feature_name: str,
) -> str:
    return (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )


def print_linear_coefficients(
    linear_pipeline: Pipeline,
    top_n: int = 30,
) -> None:
    preprocessor = linear_pipeline.named_steps[
        "preprocessor"
    ]

    linear_model = linear_pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = linear_model.coef_

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficient_table["feature"] = (
        coefficient_table["feature"]
        .map(clean_feature_name)
    )

    coefficient_table["abs_coefficient"] = (
        coefficient_table["coefficient"].abs()
    )

    coefficient_table["direction"] = np.where(
        coefficient_table["coefficient"] >= 0,
        "+",
        "-",
    )

    coefficient_table["approx_change_percent"] = (
        np.exp(coefficient_table["coefficient"])
        - 1
    ) * 100

    coefficient_table = coefficient_table.sort_values(
        by="abs_coefficient",
        ascending=False,
    )

    positive_coefficients = (
        coefficient_table[
            coefficient_table["coefficient"] > 0
        ]
        .head(top_n)
        .copy()
    )

    negative_coefficients = (
        coefficient_table[
            coefficient_table["coefficient"] < 0
        ]
        .head(top_n)
        .copy()
    )

    print("\n[LinearRegression 양의 계수]")
    print(
        positive_coefficients[
            [
                "feature",
                "coefficient",
                "direction",
                "approx_change_percent",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "coefficient": (
                    lambda value: f"{value:.4f}"
                ),
                "approx_change_percent": (
                    lambda value: f"{value:+.1f}%"
                ),
            },
        )
    )

    print("\n[LinearRegression 음의 계수]")
    print(
        negative_coefficients[
            [
                "feature",
                "coefficient",
                "direction",
                "approx_change_percent",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "coefficient": (
                    lambda value: f"{value:.4f}"
                ),
                "approx_change_percent": (
                    lambda value: f"{value:+.1f}%"
                ),
            },
        )
    )


def create_prediction_sample(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    sample = pd.DataFrame(
        {
            "actual_fee": y_test.to_numpy(),
            "dummy_prediction": (
                predictions_by_model[
                    "DummyRegressor"
                ]
            ),
            "linear_prediction": (
                predictions_by_model[
                    "LinearRegression"
                ]
            ),
        },
        index=y_test.index,
    )

    sample["linear_absolute_error"] = np.abs(
        sample["actual_fee"]
        - sample["linear_prediction"]
    )

    sample["linear_error_rate"] = (
        sample["linear_absolute_error"]
        / sample["actual_fee"]
        * 100
    )

    sample = sample.join(
        X_test[
            [
                "age_at_transfer",
                "main_position",
                "from_league_id",
                "to_league_id",
                "rating",
                "goals",
                "assists",
            ]
        ]
    )

    sample = sample.sort_values(
        by="actual_fee",
        ascending=False,
    )

    return sample


def print_prediction_sample(
    sample: pd.DataFrame,
    row_count: int = 15,
) -> None:
    display_data = sample.head(
        row_count
    ).copy()

    money_columns = [
        "actual_fee",
        "dummy_prediction",
        "linear_prediction",
        "linear_absolute_error",
    ]

    for column in money_columns:
        display_data[column] = (
            display_data[column]
            .map(
                lambda value: f"€{value:,.0f}"
            )
        )

    display_data["linear_error_rate"] = (
        display_data["linear_error_rate"]
        .map(
            lambda value: f"{value:.1f}%"
        )
    )

    print("\n[실제값과 예측값 샘플]")
    print(display_data.to_string())


def main() -> None:
    data = load_data()

    X, y = prepare_data(data)

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

    models = create_models()

    results = []
    predictions_by_model = {}

    for model_name, model in models.items():
        result, predictions = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(result)

        predictions_by_model[
            model_name
        ] = predictions

        print_model_result(result)

    print_model_comparison(results)

    print_linear_coefficients(
        linear_pipeline=models[
            "LinearRegression"
        ],
        top_n=20,
    )

    prediction_sample = create_prediction_sample(
        X_test=X_test,
        y_test=y_test,
        predictions_by_model=predictions_by_model,
    )

    print_prediction_sample(
        prediction_sample,
        row_count=15,
    )

    print("\n베이스라인 학습 완료")


if __name__ == "__main__":
    main()