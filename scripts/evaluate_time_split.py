# scripts/evaluate_time_split.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_FILE = Path("data/processed/training_dataset.csv")

OUTPUT_DIR = Path("outputs/time_split")
MODEL_DIR = Path("models")

PREDICTION_FILE = OUTPUT_DIR / "time_split_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "time_split_metrics.csv"
XGBOOST_MODEL_FILE = MODEL_DIR / "transfer_fee_prediction_model_v1.joblib"


# 우선 1차 검증에서는 시장가를 제외한다.
# 이후 True로 변경해서 시장가 포함 모델과 비교할 수 있다.
USE_MARKET_VALUE = True


NUMERIC_FEATURE_CANDIDATES = [
    # 선수 기록
    "matches",
    "started",
    "minutes",
    "rating",
    "goals",
    "assists",

    # 선수 프로필
    "age_at_transfer",
    "height",

    # 파생 변수
    "is_same_league",
    "is_top5_destination",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "league_name",       # 출발 리그
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
    "citizenship",
]

MARKET_VALUE_CANDIDATES = [
    "value_at_transfer",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"학습 데이터 파일을 찾을 수 없습니다: {DATA_FILE}\n"
            "먼저 create_training_dataset.py를 실행하세요."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "transfer_date",
        "transfer_fee",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {sorted(missing_columns)}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df["transfer_fee"] = pd.to_numeric(
        df["transfer_fee"],
        errors="coerce",
    )

    before_count = len(df)

    df = df.dropna(
        subset=[
            "transfer_date",
            "transfer_fee",
        ]
    ).copy()

    df = df[df["transfer_fee"] > 0].copy()

    removed_count = before_count - len(df)

    print("=" * 70)
    print("데이터 로드 완료")
    print("=" * 70)
    print(f"원본 행 수       : {before_count:,}")
    print(f"제외된 행 수     : {removed_count:,}")
    print(f"최종 사용 행 수  : {len(df):,}")
    print()

    return df


def select_features(df: pd.DataFrame):
    numeric_features = [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in df.columns
    ]

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURE_CANDIDATES
        if column in df.columns
    ]

    if USE_MARKET_VALUE:
        market_value_features = [
            column
            for column in MARKET_VALUE_CANDIDATES
            if column in df.columns
        ]

        numeric_features.extend(market_value_features)

    selected_features = numeric_features + categorical_features

    if not selected_features:
        raise ValueError(
            "사용 가능한 피처가 없습니다.\n"
            f"현재 데이터셋 컬럼: {df.columns.tolist()}"
        )

    print("=" * 70)
    print("사용 피처")
    print("=" * 70)
    print(f"시장가 포함 여부: {USE_MARKET_VALUE}")
    print(f"수치형 피처     : {numeric_features}")
    print(f"범주형 피처     : {categorical_features}")

    missing_numeric = [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column not in df.columns
    ]

    missing_categorical = [
        column
        for column in CATEGORICAL_FEATURE_CANDIDATES
        if column not in df.columns
    ]

    if missing_numeric:
        print(f"없는 수치형 후보 : {missing_numeric}")

    if missing_categorical:
        print(f"없는 범주형 후보 : {missing_categorical}")

    print()

    return numeric_features, categorical_features


def split_by_time(df: pd.DataFrame):
    transfer_year = df["transfer_date"].dt.year

    train_df = df[
        transfer_year.between(2020, 2023)
    ].copy()

    validation_df = df[
        transfer_year == 2024
    ].copy()

    test_df = df[
        transfer_year == 2025
    ].copy()

    print("=" * 70)
    print("시간 분할 결과")
    print("=" * 70)
    print(f"Train      2020~2023 : {len(train_df):,}건")
    print(f"Validation 2024      : {len(validation_df):,}건")
    print(f"Test       2025      : {len(test_df):,}건")
    print()

    if train_df.empty:
        raise ValueError("Train 데이터가 비어 있습니다.")

    if validation_df.empty:
        raise ValueError("Validation 데이터가 비어 있습니다.")

    if test_df.empty:
        raise ValueError("Test 데이터가 비어 있습니다.")

    return train_df, validation_df, test_df


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
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
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
):
    models = {
        "DummyRegressor": DummyRegressor(
            strategy="median",
        ),
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    }

    pipelines = {}

    for model_name, model in models.items():
        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        pipelines[model_name] = Pipeline(
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

    return pipelines


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def print_metrics(
    model_name: str,
    split_name: str,
    metrics: dict,
):
    print(
        f"[{model_name}] {split_name}\n"
        f"  MAE  : €{metrics['MAE']:,.0f}\n"
        f"  RMSE : €{metrics['RMSE']:,.0f}\n"
        f"  R²   : {metrics['R2']:.4f}"
    )


def add_fee_band(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["fee_band"] = pd.cut(
        result["actual_fee"],
        bins=[
            -np.inf,
            5_000_000,
            15_000_000,
            30_000_000,
            np.inf,
        ],
        labels=[
            "0~5M",
            "5~15M",
            "15~30M",
            "30M+",
        ],
        right=False,
    )

    return result


def print_band_metrics(
    prediction_df: pd.DataFrame,
    model_name: str,
):
    model_df = prediction_df[
        prediction_df["model"] == model_name
    ].copy()

    model_df = add_fee_band(model_df)

    print()
    print(f"[{model_name}] 2025 Test 구간별 MAE")

    for fee_band, group in model_df.groupby(
        "fee_band",
        observed=False,
    ):
        if group.empty:
            continue

        mae = mean_absolute_error(
            group["actual_fee"],
            group["predicted_fee"],
        )

        print(
            f"  {fee_band:>6} "
            f"| {len(group):>4}건 "
            f"| MAE €{mae:,.0f}"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    numeric_features, categorical_features = select_features(df)

    selected_features = (
        numeric_features
        + categorical_features
    )

    train_df, validation_df, test_df = split_by_time(df)

    X_train = train_df[selected_features]
    X_validation = validation_df[selected_features]
    X_test = test_df[selected_features]

    # 이적료 분포가 오른쪽으로 매우 치우쳐 있으므로 log1p 변환
    y_train_log = np.log1p(
        train_df["transfer_fee"]
    )

    y_validation = validation_df["transfer_fee"]
    y_test = test_df["transfer_fee"]

    pipelines = build_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    metrics_rows = []
    prediction_frames = []

    for model_name, pipeline in pipelines.items():
        print()
        print("=" * 70)
        print(f"{model_name} 학습")
        print("=" * 70)

        pipeline.fit(
            X_train,
            y_train_log,
        )

        validation_prediction_log = pipeline.predict(
            X_validation
        )

        test_prediction_log = pipeline.predict(
            X_test
        )

        validation_prediction = np.maximum(
            np.expm1(validation_prediction_log),
            0,
        )

        test_prediction = np.maximum(
            np.expm1(test_prediction_log),
            0,
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_prediction,
        )

        test_metrics = calculate_metrics(
            y_test,
            test_prediction,
        )

        print_metrics(
            model_name=model_name,
            split_name="2024 Validation",
            metrics=validation_metrics,
        )

        print_metrics(
            model_name=model_name,
            split_name="2025 Test",
            metrics=test_metrics,
        )

        for split_name, metrics in [
            ("validation_2024", validation_metrics),
            ("test_2025", test_metrics),
        ]:
            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "use_market_value": USE_MARKET_VALUE,
                    "mae": metrics["MAE"],
                    "rmse": metrics["RMSE"],
                    "r2": metrics["R2"],
                }
            )

        prediction_frame = test_df[
            [
                column
                for column in [
                    "player_id",
                    "player_name",
                    "transfer_date",
                    "from_team_name",
                    "to_team_name",
                    "transfer_fee",
                ]
                if column in test_df.columns
            ]
        ].copy()

        prediction_frame = prediction_frame.rename(
            columns={
                "transfer_fee": "actual_fee",
            }
        )

        prediction_frame["predicted_fee"] = test_prediction
        prediction_frame["absolute_error"] = np.abs(
            prediction_frame["actual_fee"]
            - prediction_frame["predicted_fee"]
        )
        prediction_frame["model"] = model_name

        prediction_frames.append(prediction_frame)

        if model_name == "XGBoost":
            joblib.dump(
                pipeline,
                XGBOOST_MODEL_FILE,
            )

            print(
                f"\nXGBoost 모델 저장: "
                f"{XGBOOST_MODEL_FILE}"
            )

    metrics_df = pd.DataFrame(metrics_rows)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("최종 성능 비교")
    print("=" * 70)

    test_metrics_df = metrics_df[
        metrics_df["split"] == "test_2025"
    ].sort_values("mae")

    print(
        test_metrics_df[
            [
                "model",
                "mae",
                "rmse",
                "r2",
            ]
        ].to_string(
            index=False,
            formatters={
                "mae": lambda value: f"€{value:,.0f}",
                "rmse": lambda value: f"€{value:,.0f}",
                "r2": lambda value: f"{value:.4f}",
            },
        )
    )

    for model_name in pipelines:
        print_band_metrics(
            prediction_df=predictions_df,
            model_name=model_name,
        )

    xgb_predictions = predictions_df[
        predictions_df["model"] == "XGBoost"
    ].sort_values(
        "absolute_error",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("XGBoost 오차가 큰 10건")
    print("=" * 70)

    display_columns = [
        column
        for column in [
            "player_name",
            "transfer_date",
            "from_team_name",
            "to_team_name",
            "actual_fee",
            "predicted_fee",
            "absolute_error",
        ]
        if column in xgb_predictions.columns
    ]

    print(
        xgb_predictions[
            display_columns
        ].head(10).to_string(
            index=False,
            formatters={
                "actual_fee": lambda value: f"€{value:,.0f}",
                "predicted_fee": lambda value: f"€{value:,.0f}",
                "absolute_error": lambda value: f"€{value:,.0f}",
            },
        )
    )

    print()
    print("=" * 70)
    print("파일 저장 완료")
    print("=" * 70)
    print(f"성능 지표 : {METRICS_FILE}")
    print(f"예측 결과 : {PREDICTION_FILE}")
    print(f"저장 모델 : {XGBOOST_MODEL_FILE}")


if __name__ == "__main__":
    main()