from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_FILE = PROJECT_ROOT / "data" / "processed" / "training_dataset_v1_1.csv"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_v1_1"

MODEL_FILE = MODEL_DIR / "transfer_fee_model_v1_1.joblib"
METADATA_FILE = MODEL_DIR / "transfer_fee_model_v1_1_metadata.json"

EVALUATION_FILE = OUTPUT_DIR / "model_v1_1_evaluation.csv"
PREDICTION_FILE = OUTPUT_DIR / "model_v1_1_test_predictions.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "model_v1_1_feature_importance.csv"


# ============================================================
# 모델 설정
# ============================================================

TARGET = "transfer_fee"
DATE_COLUMN = "transfer_date"

# 2025년 이적 데이터를 테스트 데이터로 사용
TEST_YEAR = 2025

NUMERIC_FEATURES = [
    "age_at_transfer",
    "height",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "is_same_league",
    "is_top5_destination",
]

CATEGORICAL_FEATURES = [
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


# ============================================================
# 유틸리티 함수
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필수 컬럼이 없습니다.\n"
            f"누락 컬럼: {missing_columns}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    숫자 컬럼에 쉼표, 통화기호 등이 포함되어 있어도
    최대한 숫자로 변환한다.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "null": np.nan,
        }
    )

    return pd.to_numeric(cleaned, errors="coerce")


def load_training_data() -> pd.DataFrame:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"학습 데이터 파일을 찾을 수 없습니다.\n"
            f"확인 경로: {TRAINING_FILE}"
        )

    print(f"학습 데이터 로드: {TRAINING_FILE}")

    df = pd.read_csv(
        TRAINING_FILE,
        low_memory=False,
    )

    required_columns = FEATURES + [
        TARGET,
        DATE_COLUMN,
        "player_id",
        "player_name",
    ]

    check_required_columns(df, required_columns)

    return df


def prepare_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    prepared = df.copy()

    # 날짜 변환
    prepared[DATE_COLUMN] = pd.to_datetime(
        prepared[DATE_COLUMN],
        errors="coerce",
    )

    # 숫자형 변환
    numeric_columns = NUMERIC_FEATURES + [TARGET]

    for column in numeric_columns:
        prepared[column] = clean_numeric_column(
            prepared[column]
        )

    # 범주형 컬럼 정리
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].astype("object")

        prepared[column] = prepared[column].apply(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

        prepared[column] = prepared[column].replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "<NA>": np.nan,
                "null": np.nan,
            }
        )

        prepared[column] = prepared[column].where(
            prepared[column].notna(),
            np.nan,
        )

    initial_count = len(prepared)

    # 날짜 없는 행 제외
    prepared = prepared.dropna(
        subset=[DATE_COLUMN]
    ).copy()

    # 타깃 없는 행 제외
    prepared = prepared.dropna(
        subset=[TARGET]
    ).copy()

    # 0 이하 이적료 제외
    prepared = prepared[
        prepared[TARGET] > 0
    ].copy()

    # 연도 생성
    prepared["transfer_year"] = (
        prepared[DATE_COLUMN].dt.year
    )

    removed_count = initial_count - len(prepared)

    print(f"원본 행 수       : {initial_count:,}")
    print(f"사용 가능 행 수 : {len(prepared):,}")
    print(f"제외 행 수       : {removed_count:,}")

    if prepared.empty:
        raise ValueError(
            "전처리 후 사용할 수 있는 학습 데이터가 없습니다."
        )

    return prepared


def create_preprocessor(
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                missing_values=np.nan,
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

    return ColumnTransformer(
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
        ],
        remainder="drop",
    )


def create_dummy_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                DummyRegressor(
                    strategy="median"
                ),
            ),
        ]
    )


def create_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=True
                ),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )


def create_xgboost_pipeline() -> Pipeline:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    # 로그 예측값 역변환 과정에서 음수가 나올 경우 방지
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    median_ae = np.median(
        np.abs(
            np.asarray(y_true) - y_pred
        )
    )

    # 실제 이적료가 0보다 큰 데이터만 사용하므로 계산 가능
    mape = np.mean(
        np.abs(
            (
                np.asarray(y_true) - y_pred
            )
            / np.asarray(y_true)
        )
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Median_AE": float(median_ae),
        "MAPE": float(mape),
    }


def print_metrics(
    model_name: str,
    metrics: dict,
) -> None:
    print()
    print(f"[{model_name}]")
    print(f"MAE       : €{metrics['MAE']:,.0f}")
    print(f"RMSE      : €{metrics['RMSE']:,.0f}")
    print(f"Median AE : €{metrics['Median_AE']:,.0f}")
    print(f"MAPE      : {metrics['MAPE']:.2f}%")
    print(f"R²        : {metrics['R2']:.4f}")


def fit_and_evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, np.ndarray, dict]:
    print()
    print(f"{model_name} 학습 중...")

    # 이적료 분포가 매우 치우쳐 있으므로 log1p 변환
    y_train_log = np.log1p(y_train)

    pipeline.fit(
        x_train,
        y_train_log,
    )

    predicted_log = pipeline.predict(
        x_test
    )

    predictions = np.expm1(
        predicted_log
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print_metrics(
        model_name,
        metrics,
    )

    return pipeline, predictions, metrics


def save_test_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    result_columns = [
        "player_id",
        "player_name",
        "transfer_date",
        "from_team_name",
        "from_league_name",
        "to_team_name",
        "to_league_name",
        "transfer_fee",
    ]

    available_columns = [
        column
        for column in result_columns
        if column in test_df.columns
    ]

    result = test_df[
        available_columns
    ].copy()

    result["predicted_transfer_fee"] = predictions

    result["absolute_error"] = (
        result["transfer_fee"]
        - result["predicted_transfer_fee"]
    ).abs()

    result["percentage_error"] = (
        result["absolute_error"]
        / result["transfer_fee"]
        * 100
    )

    result = result.sort_values(
        "absolute_error",
        ascending=False,
    )

    result.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"테스트 예측 결과 저장: {PREDICTION_FILE}"
    )


def save_feature_importance(
    fitted_pipeline: Pipeline,
) -> None:
    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    model = fitted_pipeline.named_steps[
        "model"
    ]

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )

        importances = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )

        importance_df["feature"] = (
            importance_df["feature"]
            .str.replace(
                "numeric__",
                "",
                regex=False,
            )
            .str.replace(
                "categorical__",
                "",
                regex=False,
            )
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False,
        )

        importance_df.to_csv(
            FEATURE_IMPORTANCE_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "피처 중요도 저장: "
            f"{FEATURE_IMPORTANCE_FILE}"
        )

    except Exception as error:
        print(
            "피처 중요도 저장을 건너뜁니다: "
            f"{error}"
        )


def train_final_model(
    df: pd.DataFrame,
) -> Pipeline:
    print_section("전체 데이터 최종 모델 학습")

    x_all = df[FEATURES].copy()
    y_all = df[TARGET].copy()

    final_pipeline = create_xgboost_pipeline()

    y_all_log = np.log1p(y_all)

    final_pipeline.fit(
        x_all,
        y_all_log,
    )

    joblib.dump(
        final_pipeline,
        MODEL_FILE,
    )

    print(f"최종 학습 행 수 : {len(df):,}")
    print(f"모델 저장       : {MODEL_FILE}")

    return final_pipeline


def save_metadata(
    df: pd.DataFrame,
    evaluation_results: list[dict],
) -> None:
    metadata = {
        "model_name": "transfer_fee_model_v1_1",
        "model_type": "XGBRegressor",
        "training_file": str(TRAINING_FILE),
        "target": TARGET,
        "target_transformation": "log1p",
        "inverse_transformation": "expm1",
        "test_year": TEST_YEAR,
        "training_rows": int(len(df)),
        "training_year_min": int(
            df["transfer_year"].min()
        ),
        "training_year_max": int(
            df["transfer_year"].max()
        ),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": FEATURES,
        "value_at_transfer_included": False,
        "evaluation": evaluation_results,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"모델 설정 저장  : {METADATA_FILE}")


# ============================================================
# 메인 실행
# ============================================================

def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section("Transfer Fee Prediction Model V1.1")

    raw_df = load_training_data()
    df = prepare_training_data(raw_df)

    print_section("학습 데이터 상태")

    print(f"전체 행 수 : {len(df):,}")
    print(
        "이적 연도   : "
        f"{df['transfer_year'].min()} ~ "
        f"{df['transfer_year'].max()}"
    )

    print()
    print("모델 입력 피처:")
    for feature in FEATURES:
        missing_count = int(
            df[feature].isna().sum()
        )

        print(
            f"  - {feature:<24}"
            f"결측 {missing_count:>5,}"
        )

    # --------------------------------------------------------
    # 시간 기준 분할
    # --------------------------------------------------------

    train_df = df[
        df["transfer_year"] < TEST_YEAR
    ].copy()

    test_df = df[
        df["transfer_year"] == TEST_YEAR
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"{TEST_YEAR}년 이전 학습 데이터가 없습니다."
        )

    if test_df.empty:
        raise ValueError(
            f"{TEST_YEAR}년 테스트 데이터가 없습니다.\n"
            "TEST_YEAR 값 또는 transfer_date를 확인하세요."
        )

    print_section("시간 기준 데이터 분할")

    print(
        f"학습 데이터 : {len(train_df):,}행 "
        f"({train_df['transfer_year'].min()}~"
        f"{train_df['transfer_year'].max()})"
    )

    print(
        f"테스트 데이터: {len(test_df):,}행 "
        f"({TEST_YEAR})"
    )

    x_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()

    x_test = test_df[FEATURES].copy()
    y_test = test_df[TARGET].copy()

    # --------------------------------------------------------
    # 모델 비교
    # --------------------------------------------------------

    print_section("모델 성능 평가")

    model_configs = [
        (
            "DummyRegressor",
            create_dummy_pipeline(),
        ),
        (
            "LinearRegression",
            create_linear_pipeline(),
        ),
        (
            "XGBoost",
            create_xgboost_pipeline(),
        ),
    ]

    evaluation_results = []
    xgboost_pipeline = None
    xgboost_predictions = None

    for model_name, pipeline in model_configs:
        (
            fitted_pipeline,
            predictions,
            metrics,
        ) = fit_and_evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        evaluation_results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        if model_name == "XGBoost":
            xgboost_pipeline = fitted_pipeline
            xgboost_predictions = predictions

    evaluation_df = pd.DataFrame(
        evaluation_results
    )

    evaluation_df.to_csv(
        EVALUATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"평가 결과 저장: {EVALUATION_FILE}"
    )

    if (
        xgboost_pipeline is None
        or xgboost_predictions is None
    ):
        raise RuntimeError(
            "XGBoost 평가 결과가 생성되지 않았습니다."
        )

    save_test_predictions(
        test_df=test_df,
        predictions=xgboost_predictions,
    )

    save_feature_importance(
        fitted_pipeline=xgboost_pipeline,
    )

    # --------------------------------------------------------
    # 전체 데이터로 최종 모델 학습
    # --------------------------------------------------------

    train_final_model(df)

    save_metadata(
        df=df,
        evaluation_results=evaluation_results,
    )

    print_section("Model V1.1 생성 완료")

    print(f"학습 데이터     : {TRAINING_FILE}")
    print(f"저장 모델       : {MODEL_FILE}")
    print(f"모델 메타데이터 : {METADATA_FILE}")
    print(f"평가 결과       : {EVALUATION_FILE}")
    print(f"테스트 예측     : {PREDICTION_FILE}")
    print(f"피처 중요도     : {FEATURE_IMPORTANCE_FILE}")

    print()
    print(
    "다음 단계: V1과 V1.1 성능 및 Feature Importance 비교"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section("Model V1 실행 실패")
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)