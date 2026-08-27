import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 설정
# ============================================================

DATA_FILE = Path(
    "data/processed/training_dataset_european.csv"
)

BASE_MODEL_FILE = Path(
    "models/transfer_fee_model_v1_1.joblib"
)

OUTPUT_FILE = Path(
    "results/european_feature_evaluation_2025.csv"
)

TARGET = "transfer_fee"

TRAIN_END_YEAR = 2024
TEST_YEAR = 2025


# 기존 v1.2 앙상블 가중치
ALPHA_C = 0.4
ALPHA_D = 0.6


# 기존 v1.2 sample weight
WEIGHT_UNDER_30M = 1.0
WEIGHT_30_TO_50M = 2.5
WEIGHT_50M_PLUS = 4.0


# Model C
C_REMOVE_FEATURES = [
    "is_same_league",
]


# Model D
D_REMOVE_FEATURES = [
    "is_same_league",
    "to_league_id",
]


# ============================================================
# 유럽대항전 Feature
# ============================================================

EUROPEAN_FEATURES = [
    "ucl_appearances",
    "ucl_starts",
    "ucl_goals",
    "ucl_assists",

    "uel_appearances",
    "uel_starts",
    "uel_goals",
    "uel_assists",

    "uecl_appearances",
    "uecl_starts",
    "uecl_goals",
    "uecl_assists",
]


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
        (
            df["goals"]
            + df["assists"]
        )
        / df["minutes"]
        * 90,
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
# Sample Weight
# ============================================================

def make_weight(y):
    y = np.asarray(y)

    weights = np.ones(
        len(y),
        dtype=float,
    )

    weights[
        (y >= 30_000_000)
        & (y < 50_000_000)
    ] = WEIGHT_30_TO_50M

    weights[
        y >= 50_000_000
    ] = WEIGHT_50M_PLUS

    return weights


# ============================================================
# Pipeline 유틸
# ============================================================

def clone_model(model):
    try:
        return clone(model)

    except Exception:
        return copy.deepcopy(model)


def find_column_transformer(model):
    if not hasattr(
        model,
        "steps",
    ):
        raise ValueError(
            "모델이 sklearn Pipeline이 아닙니다."
        )

    for step_name, step in model.steps:
        if isinstance(
            step,
            ColumnTransformer,
        ):
            return (
                step_name,
                step,
            )

    raise ValueError(
        "ColumnTransformer를 찾지 못했습니다."
    )


def get_estimator_step_name(model):
    if (
        not hasattr(model, "steps")
        or len(model.steps) == 0
    ):
        raise ValueError(
            "estimator step을 찾지 못했습니다."
        )

    return model.steps[-1][0]


# ============================================================
# Feature 제거
# ============================================================

def remove_features_from_model(
    base_model,
    features_to_remove,
):
    model = clone_model(
        base_model
    )

    remove_set = set(
        features_to_remove
    )

    _, preprocessor = (
        find_column_transformer(
            model
        )
    )

    new_transformers = []

    for (
        name,
        transformer,
        columns,
    ) in preprocessor.transformers:

        if isinstance(
            columns,
            (
                list,
                tuple,
                np.ndarray,
                pd.Index,
            ),
        ):
            new_columns = [
                column
                for column in columns
                if column not in remove_set
            ]

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

    preprocessor.transformers = (
        new_transformers
    )

    return model


# ============================================================
# Numeric Transformer에 Europe Feature 추가
# ============================================================

def add_numeric_features_to_model(
    model,
    features_to_add,
):
    _, preprocessor = (
        find_column_transformer(
            model
        )
    )

    # 기존 numeric transformer를 찾기 위한
    # 기준 feature들
    numeric_reference_features = {
        "age_at_transfer",
        "height",
        "matches",
        "started",
        "goals",
        "assists",
        "minutes",
        "rating",
        "goals_per90",
        "assists_per90",
        "goal_contributions_per90",
        "starts_ratio",
        "minutes_per_match",
        "age_squared",
    }

    best_index = None
    best_score = 0

    # numeric feature가 가장 많이 들어있는
    # transformer를 찾는다.
    for index, (
        name,
        transformer,
        columns,
    ) in enumerate(
        preprocessor.transformers
    ):

        if not isinstance(
            columns,
            (
                list,
                tuple,
                np.ndarray,
                pd.Index,
            ),
        ):
            continue

        score = len(
            set(columns)
            & numeric_reference_features
        )

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        raise ValueError(
            "numeric transformer를 찾지 못했습니다."
        )

    new_transformers = []

    for index, (
        name,
        transformer,
        columns,
    ) in enumerate(
        preprocessor.transformers
    ):

        if (
            index == best_index
            and isinstance(
                columns,
                (
                    list,
                    tuple,
                    np.ndarray,
                    pd.Index,
                ),
            )
        ):
            new_columns = list(
                columns
            )

            for feature in (
                features_to_add
            ):
                if (
                    feature
                    not in new_columns
                ):
                    new_columns.append(
                        feature
                    )

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

    preprocessor.transformers = (
        new_transformers
    )

    return model


# ============================================================
# Model Variant 생성
# ============================================================

def build_model(
    base_model,
    removed_features,
    european=False,
):
    model = (
        remove_features_from_model(
            base_model,
            removed_features,
        )
    )

    if european:
        model = (
            add_numeric_features_to_model(
                model,
                EUROPEAN_FEATURES,
            )
        )

    return model


# ============================================================
# 학습
# ============================================================

def train_variant(
    base_model,
    train_df,
    base_features,
    removed_features,
    european=False,
):

    features = [
        feature
        for feature in base_features
        if feature not in removed_features
    ]

    if european:
        features = (
            features
            + EUROPEAN_FEATURES
        )

    model = build_model(
        base_model,
        removed_features,
        european=european,
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        TARGET
    ]

    sample_weight = (
        make_weight(
            y_train
        )
    )

    estimator_step = (
        get_estimator_step_name(
            model
        )
    )

    fit_params = {
        (
            f"{estimator_step}"
            "__sample_weight"
        ):
            sample_weight
    }

    model.fit(
        X_train,
        np.log1p(
            y_train
        ),
        **fit_params,
    )

    return (
        model,
        features,
    )


# ============================================================
# 예측
# ============================================================

def predict_model(
    model,
    df,
    features,
):
    log_prediction = model.predict(
        df[features]
    )

    prediction = np.expm1(
        log_prediction
    )

    # 음수 이적료 방지
    prediction = np.maximum(
        prediction,
        0,
    )

    return prediction


# ============================================================
# 평가
# ============================================================

def calculate_metrics(
    y_true,
    prediction,
):
    y_true = np.asarray(
        y_true
    )

    prediction = np.asarray(
        prediction
    )

    mae = mean_absolute_error(
        y_true,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            prediction,
        )
    )

    r2 = r2_score(
        y_true,
        prediction,
    )

    return {
        "mae_m":
            mae / 1_000_000,

        "rmse_m":
            rmse / 1_000_000,

        "r2":
            r2,
    }


def calculate_segment_mae(
    y_true,
    prediction,
    threshold,
):
    y_true = np.asarray(
        y_true
    )

    prediction = np.asarray(
        prediction
    )

    mask = (
        y_true
        >= threshold
    )

    if mask.sum() == 0:
        return (
            np.nan,
            0,
        )

    mae = mean_absolute_error(
        y_true[mask],
        prediction[mask],
    )

    return (
        mae / 1_000_000,
        int(mask.sum()),
    )


def calculate_top10_mae(
    y_true,
    prediction,
):
    y_true = np.asarray(
        y_true
    )

    prediction = np.asarray(
        prediction
    )

    count = min(
        10,
        len(y_true),
    )

    indexes = np.argsort(
        y_true
    )[-count:]

    mae = mean_absolute_error(
        y_true[indexes],
        prediction[indexes],
    )

    return (
        mae / 1_000_000
    )


# ============================================================
# 결과 출력
# ============================================================

def print_metrics(
    name,
    y_true,
    prediction,
):
    metrics = calculate_metrics(
        y_true,
        prediction,
    )

    mae_30, count_30 = (
        calculate_segment_mae(
            y_true,
            prediction,
            30_000_000,
        )
    )

    mae_50, count_50 = (
        calculate_segment_mae(
            y_true,
            prediction,
            50_000_000,
        )   
    )

    top10_mae = (
        calculate_top10_mae(
            y_true,
            prediction,
        )
    )

    print()
    print(name)
    print("-" * 60)

    print(
        f"MAE       : "
        f"{metrics['mae_m']:.3f}M"
    )

    print(
        f"RMSE      : "
        f"{metrics['rmse_m']:.3f}M"
    )

    print(
        f"R²        : "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"30M+ MAE  : "
        f"{mae_30:.3f}M "
        f"({count_30}명)"
    )

    print(
        f"50M+ MAE  : "
        f"{mae_50:.3f}M "
        f"({count_50}명)"
    )

    print(
        f"Top10 MAE : "
        f"{top10_mae:.3f}M"
    )

    return {
        **metrics,

        "mae_30m_plus":
            mae_30,

        "count_30m_plus":
            count_30,

        "mae_50m_plus":
            mae_50,

        "count_50m_plus":
            count_50,

        "top10_mae_m":
            top10_mae,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "European Competition Feature Evaluation"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"데이터 없음: "
            f"{DATA_FILE}"
        )

    if not BASE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"모델 없음: "
            f"{BASE_MODEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. 데이터
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    print(
        "전체 rows:",
        len(df),
    )

    df["transfer_date"] = (
        pd.to_datetime(
            df["transfer_date"],
            errors="coerce",
        )
    )

    df = add_v11_features(
        df
    )

    # 유럽대항전 feature 숫자형 + 결측 0
    for feature in (
        EUROPEAN_FEATURES
    ):
        if feature not in df.columns:
            raise ValueError(
                f"유럽대항전 feature 없음: "
                f"{feature}"
            )

        df[feature] = (
            pd.to_numeric(
                df[feature],
                errors="coerce",
            )
            .fillna(0)
        )

    # Target 정상 행만
    df = df[
        df[TARGET].notna()
    ].copy()

    df = df[
        df[TARGET] >= 0
    ].copy()

    # --------------------------------------------------------
    # 3. Base model
    # --------------------------------------------------------

    base_model = joblib.load(
        BASE_MODEL_FILE
    )

    base_features = list(
        base_model.feature_names_in_
    )

    if (
        "value_at_transfer"
        in base_features
    ):
        raise ValueError(
            "Market value 모델입니다."
        )

    print(
        "✓ no-market v1.1 base model 확인"
    )

    # --------------------------------------------------------
    # 4. Time Split
    # --------------------------------------------------------

    transfer_year = (
        df["transfer_date"]
        .dt.year
    )

    train_df = df[
        transfer_year
        <= TRAIN_END_YEAR
    ].copy()

    test_df = df[
        transfer_year
        == TEST_YEAR
    ].copy()

    print()
    print(
        "Train rows:",
        len(train_df),
    )

    print(
        "Test rows :",
        len(test_df),
    )

    print(
        "Train years:",
        int(
            train_df[
                "transfer_date"
            ].dt.year.min()
        ),
        "~",
        int(
            train_df[
                "transfer_date"
            ].dt.year.max()
        ),
    )

    print(
        "Test year :",
        TEST_YEAR,
    )

    if (
        len(train_df) == 0
        or len(test_df) == 0
    ):
        raise ValueError(
            "Train/Test split 결과가 비어 있습니다."
        )

    # --------------------------------------------------------
    # 5. 기존 v1.2 재현
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Baseline v1.2 학습")
    print("=" * 72)

    baseline_c, baseline_features_c = (
        train_variant(
            base_model,
            train_df,
            base_features,
            C_REMOVE_FEATURES,
            european=False,
        )
    )

    baseline_d, baseline_features_d = (
        train_variant(
            base_model,
            train_df,
            base_features,
            D_REMOVE_FEATURES,
            european=False,
        )
    )

    baseline_pred_c = predict_model(
        baseline_c,
        test_df,
        baseline_features_c,
    )

    baseline_pred_d = predict_model(
        baseline_d,
        test_df,
        baseline_features_d,
    )

    baseline_pred = (
        ALPHA_C
        * baseline_pred_c
        + ALPHA_D
        * baseline_pred_d
    )

    # --------------------------------------------------------
    # 6. Europe 모델
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "v1.2 + European Features 학습"
    )
    print("=" * 72)

    europe_c, europe_features_c = (
        train_variant(
            base_model,
            train_df,
            base_features,
            C_REMOVE_FEATURES,
            european=True,
        )
    )

    europe_d, europe_features_d = (
        train_variant(
            base_model,
            train_df,
            base_features,
            D_REMOVE_FEATURES,
            european=True,
        )
    )

    europe_pred_c = predict_model(
        europe_c,
        test_df,
        europe_features_c,
    )

    europe_pred_d = predict_model(
        europe_d,
        test_df,
        europe_features_d,
    )

    europe_pred = (
        ALPHA_C
        * europe_pred_c
        + ALPHA_D
        * europe_pred_d
    )

    # --------------------------------------------------------
    # 7. 전체 성능
    # --------------------------------------------------------

    y_test = (
        test_df[TARGET]
        .to_numpy()
    )

    print()
    print("=" * 72)
    print("2025 Test 결과")
    print("=" * 72)

    baseline_metrics = print_metrics(
        "Baseline v1.2",
        y_test,
        baseline_pred,
    )

    europe_metrics = print_metrics(
        "v1.2 + Europe",
        y_test,
        europe_pred,
    )

    # --------------------------------------------------------
    # 8. 개선량
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("성능 변화")
    print("=" * 72)

    mae_delta = (
        europe_metrics["mae_m"]
        - baseline_metrics["mae_m"]
    )

    rmse_delta = (
        europe_metrics["rmse_m"]
        - baseline_metrics["rmse_m"]
    )

    r2_delta = (
        europe_metrics["r2"]
        - baseline_metrics["r2"]
    )

    high_delta = (
        europe_metrics[
            "mae_50m_plus"
        ]
        - baseline_metrics[
            "mae_50m_plus"
        ]
    )

    print(
        f"MAE 변화      : "
        f"{mae_delta:+.3f}M"
    )

    print(
        f"RMSE 변화     : "
        f"{rmse_delta:+.3f}M"
    )

    print(
        f"R² 변화       : "
        f"{r2_delta:+.4f}"
    )

    print(
        f"50M+ MAE 변화 : "
        f"{high_delta:+.3f}M"
    )

    # --------------------------------------------------------
    # 9. 실제 유럽대항전 참가자만 따로 평가
    # --------------------------------------------------------

    europe_appearances = (
        test_df[
            "ucl_appearances"
        ]
        + test_df[
            "uel_appearances"
        ]
        + test_df[
            "uecl_appearances"
        ]
    )

    played_mask = (
        europe_appearances
        > 0
    ).to_numpy()

    print()
    print("=" * 72)
    print(
        "2025 유럽대항전 참가 선수만"
    )
    print("=" * 72)

    print(
        "대상:",
        int(
            played_mask.sum()
        ),
        "명",
    )

    if played_mask.sum() > 0:

        print_metrics(
            "Baseline v1.2",
            y_test[
                played_mask
            ],
            baseline_pred[
                played_mask
            ],
        )

        print_metrics(
            "v1.2 + Europe",
            y_test[
                played_mask
            ],
            europe_pred[
                played_mask
            ],
        )

    # --------------------------------------------------------
    # 10. 결과 CSV 저장
    # --------------------------------------------------------

    output = pd.DataFrame(
        {
            "player_id":
                test_df[
                    "player_id"
                ].values,

            "player_name":
                test_df[
                    "player_name"
                ].values,

            "transfer_fee":
                y_test,

            "baseline_prediction":
                baseline_pred,

            "europe_prediction":
                europe_pred,

            "prediction_change":
                (
                    europe_pred
                    - baseline_pred
                ),

            "played_europe":
                played_mask.astype(int),

            "ucl_appearances":
                test_df[
                    "ucl_appearances"
                ].values,

            "ucl_goals":
                test_df[
                    "ucl_goals"
                ].values,

            "uel_appearances":
                test_df[
                    "uel_appearances"
                ].values,

            "uel_goals":
                test_df[
                    "uel_goals"
                ].values,

            "uecl_appearances":
                test_df[
                    "uecl_appearances"
                ].values,

            "uecl_goals":
                test_df[
                    "uecl_goals"
                ].values,
        }
    )

    output[
        "absolute_error_baseline"
    ] = np.abs(
        output["transfer_fee"]
        - output[
            "baseline_prediction"
        ]
    )

    output[
        "absolute_error_europe"
    ] = np.abs(
        output["transfer_fee"]
        - output[
            "europe_prediction"
        ]
    )

    output[
        "error_improvement"
    ] = (
        output[
            "absolute_error_baseline"
        ]
        - output[
            "absolute_error_europe"
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "결과 저장:",
        OUTPUT_FILE,
    )

    # --------------------------------------------------------
    # 11. 최종 판단
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("판단")
    print("=" * 72)

    if (
        europe_metrics["mae_m"]
        < baseline_metrics["mae_m"]
        and europe_metrics["r2"]
        > baseline_metrics["r2"]
    ):
        print(
            "✓ 전체 기준에서 유럽대항전 "
            "feature가 개선됨"
        )

        print(
            "→ v1.3 후보로 검토 가능"
        )

    else:
        print(
            "△ 전체 기준에서 명확한 "
            "개선은 확인되지 않음"
        )

        print(
            "→ feature 구성/대회 처리 방식 "
            "추가 실험 권장"
        )


if __name__ == "__main__":
    main()