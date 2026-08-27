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
    # --------------------------------------------------------
    # Player / League
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # v1.3 Europe
    # --------------------------------------------------------
    "ucl_appearances": "챔피언스리그 출전",
    "ucl_starts": "챔피언스리그 선발",
    "ucl_goals": "챔피언스리그 득점",
    "ucl_assists": "챔피언스리그 도움",

    "uel_appearances": "유로파리그 출전",
    "uel_starts": "유로파리그 선발",
    "uel_goals": "유로파리그 득점",
    "uel_assists": "유로파리그 도움",

    "uecl_appearances": "컨퍼런스리그 출전",
    "uecl_starts": "컨퍼런스리그 선발",
    "uecl_goals": "컨퍼런스리그 득점",
    "uecl_assists": "컨퍼런스리그 도움",
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
# Single Model SHAP
# ============================================================

def get_model_contributions(
    pipeline,
    prediction_input: pd.DataFrame,
) -> dict[str, float]:
    """
    하나의 XGBoost Pipeline에 대해
    native SHAP(pred_contribs)을 계산한다.

    반환값:
    {
        "rating": 0.21,
        "age_at_transfer": -0.15,
        ...
    }

    값은 log1p(target) 공간의 SHAP contribution.
    """

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

    # 마지막 값은 bias(base value)
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

    return grouped_impacts


# ============================================================
# v1.3 Ensemble SHAP Explanation
# ============================================================

def explain_prediction(
    prediction_input: pd.DataFrame,
    top_n: int = 10,
) -> list[ExplanationItem]:
    """
    v1.3 Ensemble:

        Model C * alpha_c
        +
        Model D * alpha_d

    각 모델의 native SHAP contribution을 계산한 뒤
    ensemble weight를 적용하여 feature 단위로 합산한다.

    impact 값은 실제 유로 금액이 아니라
    각 모델의 log1p(target) 공간 SHAP 값을
    ensemble weight로 결합한 설명용 값이다.
    """

    bundle = get_model()

    # ========================================================
    # Model / Ensemble 정보
    # ========================================================

    model_c = bundle[
        "model_c"
    ]

    model_d = bundle[
        "model_d"
    ]

    alpha_c = float(
        bundle["alpha_c"]
    )

    alpha_d = float(
        bundle["alpha_d"]
    )

    features_c = bundle[
        "features_c"
    ]

    features_d = bundle[
        "features_d"
    ]

    # ========================================================
    # Model C SHAP
    # ========================================================

    input_c = prediction_input[
        features_c
    ]

    contributions_c = (
        get_model_contributions(
            pipeline=model_c,
            prediction_input=input_c,
        )
    )

    # ========================================================
    # Model D SHAP
    # ========================================================

    input_d = prediction_input[
        features_d
    ]

    contributions_d = (
        get_model_contributions(
            pipeline=model_d,
            prediction_input=input_d,
        )
    )

    # ========================================================
    # Ensemble SHAP
    # ========================================================

    ensemble_impacts: dict[str, float] = {}

    # --------------------------------------------------------
    # Model C
    # --------------------------------------------------------

    for feature_name, impact in (
        contributions_c.items()
    ):

        ensemble_impacts[feature_name] = (
            ensemble_impacts.get(
                feature_name,
                0.0,
            )
            + alpha_c * impact
        )

    # --------------------------------------------------------
    # Model D
    # --------------------------------------------------------

    for feature_name, impact in (
        contributions_d.items()
    ):

        ensemble_impacts[feature_name] = (
            ensemble_impacts.get(
                feature_name,
                0.0,
            )
            + alpha_d * impact
        )

    # ========================================================
    # Sort by absolute impact
    # ========================================================

    sorted_features = sorted(
        ensemble_impacts.items(),
        key=lambda item: abs(
            item[1]
        ),
        reverse=True,
    )

    sorted_features = (
        sorted_features[:top_n]
    )

    # ========================================================
    # Response
    # ========================================================

    explanation = []

    for feature_name, impact in (
        sorted_features
    ):

        direction = (
            "increase"
            if impact >= 0
            else "decrease"
        )

        explanation.append(
            ExplanationItem(
                feature=(
                    FEATURE_DISPLAY_NAMES.get(
                        feature_name,
                        feature_name,
                    )
                ),
                feature_name=feature_name,
                impact=float(impact),
                direction=direction,
            )
        )

    return explanation