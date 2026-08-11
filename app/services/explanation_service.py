import pandas as pd
import xgboost as xgb

from app.schemas.response import ExplanationItem
from app.services.model_loader import get_model


# ============================================================
# Feature Settings
# ============================================================

CATEGORICAL_FEATURES = {
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
}


FEATURE_DISPLAY_NAMES = {
    "age_at_transfer": "나이",
    "height": "키",
    "matches": "출전 경기 수",
    "started": "선발 경기 수",
    "goals": "득점",
    "assists": "도움",
    "minutes": "출전 시간",
    "rating": "평점",
    "is_same_league": "동일 리그 이적 여부",
    "is_top5_destination": "5대 리그 목적지 여부",
    "goals_per90": "90분당 득점",
    "assists_per90": "90분당 도움",
    "goal_contributions_per90": "90분당 공격포인트",
    "starts_ratio": "선발 비율",
    "minutes_per_match": "경기당 출전 시간",
    "age_squared": "나이 제곱",
    "from_league_id": "현재 리그",
    "to_league_id": "목적 리그",
    "main_position": "주 포지션",
    "foot": "주발",
}


# ============================================================
# Feature Name
# ============================================================

def clean_feature_name(
    feature_name: str,
) -> str:
    """
    ColumnTransformer가 생성한 prefix를 제거한다.

    예:
    numeric__rating
        -> rating

    categorical__to_league_id_GB1
        -> to_league_id_GB1
    """

    if "__" in feature_name:
        return feature_name.split(
            "__",
            1,
        )[1]

    return feature_name


def get_grouped_feature_name(
    feature_name: str,
) -> str:
    """
    One-Hot Encoding된 categorical feature를
    원래 feature 단위로 묶는다.

    예:
    to_league_id_GB1
        -> to_league_id
    """

    for categorical_feature in CATEGORICAL_FEATURES:

        prefix = (
            f"{categorical_feature}_"
        )

        if feature_name.startswith(prefix):
            return categorical_feature

    return feature_name


# ============================================================
# SHAP Explanation
# ============================================================

def explain_prediction(
    prediction_input: pd.DataFrame,
    top_n: int = 10,
) -> list[ExplanationItem]:
    """
    XGBoost native pred_contribs를 이용해
    예측에 영향을 준 feature를 계산한다.

    impact 값은 유로 단위가 아니라
    log1p(target) 공간의 SHAP 값이다.
    """

    pipeline = get_model()

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    xgb_model = pipeline.named_steps[
        "model"
    ]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    transformed_input = (
        preprocessor.transform(
            prediction_input
        )
    )

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    # --------------------------------------------------------
    # XGBoost Native SHAP
    # --------------------------------------------------------

    booster = xgb_model.get_booster()

    dmatrix = xgb.DMatrix(
        transformed_input
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
    )[0]

    # 마지막 값은 bias(base value)이므로 제외
    feature_contributions = (
        contributions[:-1]
    )

    if (
        len(transformed_feature_names)
        != len(feature_contributions)
    ):
        raise RuntimeError(
            "SHAP feature 개수와 "
            "전처리 feature 개수가 일치하지 않습니다."
        )

    # --------------------------------------------------------
    # Group One-Hot Features
    # --------------------------------------------------------

    grouped_impacts: dict[str, float] = {}

    for feature_name, impact in zip(
        transformed_feature_names,
        feature_contributions,
    ):

        cleaned_name = clean_feature_name(
            str(feature_name)
        )

        grouped_name = (
            get_grouped_feature_name(
                cleaned_name
            )
        )

        grouped_impacts[grouped_name] = (
            grouped_impacts.get(
                grouped_name,
                0.0,
            )
            + float(impact)
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sorted_features = sorted(
        grouped_impacts.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    sorted_features = sorted_features[
        :top_n
    ]

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    explanation = []

    for feature_name, impact in sorted_features:

        direction = (
            "increase"
            if impact >= 0
            else "decrease"
        )

        explanation.append(
            ExplanationItem(
                feature=FEATURE_DISPLAY_NAMES.get(
                    feature_name,
                    feature_name,
                ),
                feature_name=feature_name,
                impact=float(impact),
                direction=direction,
            )
        )

    return explanation