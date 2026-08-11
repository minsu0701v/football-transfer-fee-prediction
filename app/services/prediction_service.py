import numpy as np
import pandas as pd

from fastapi import HTTPException

from app.config import (
    MODEL_FEATURES,
    PREDICTION_DATE,
    TOP5_LEAGUE_IDS,
)

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse

from app.services.explanation_service import explain_prediction
from app.services.model_loader import get_model
from app.services.player_service import get_player

from app.utils.common import (
    calculate_age,
    nullable_value,
)


# ============================================================
# Safe Divide
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    0으로 나누거나 결측치가 있는 경우 NaN 반환
    """

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator == 0:
        return np.nan

    return numerator / denominator


# ============================================================
# Build Prediction Input
# ============================================================

def build_prediction_input(
    player: pd.Series,
    request: PredictionRequest,
) -> pd.DataFrame:

    current_league_id = player.get(
        "current_league_id"
    )

    if pd.isna(current_league_id):
        from_league_id = np.nan
    else:
        from_league_id = str(
            current_league_id
        ).strip()

    to_league_id = request.to_league_id.strip()

    age_at_transfer = calculate_age(
        player.get("date_of_birth"),
        PREDICTION_DATE,
    )

    height = pd.to_numeric(
        player.get("height"),
        errors="coerce",
    )

    matches = pd.to_numeric(
        player.get("matches"),
        errors="coerce",
    )

    started = pd.to_numeric(
        player.get("started"),
        errors="coerce",
    )

    goals = pd.to_numeric(
        player.get("goals"),
        errors="coerce",
    )

    assists = pd.to_numeric(
        player.get("assists"),
        errors="coerce",
    )

    minutes = pd.to_numeric(
        player.get("minutes"),
        errors="coerce",
    )

    rating = pd.to_numeric(
        player.get("rating"),
        errors="coerce",
    )

    # ========================================================
    # v1.1 Feature Engineering
    # ========================================================

    goals_per90 = (
        safe_divide(
            goals,
            minutes,
        )
        * 90
    )

    assists_per90 = (
        safe_divide(
            assists,
            minutes,
        )
        * 90
    )

    goal_contributions_per90 = (
        safe_divide(
            goals + assists,
            minutes,
        )
        * 90
    )

    starts_ratio = safe_divide(
        started,
        matches,
    )

    minutes_per_match = safe_divide(
        minutes,
        matches,
    )

    age_squared = (
        age_at_transfer ** 2
        if pd.notna(age_at_transfer)
        else np.nan
    )

    # ========================================================
    # Model Input
    # ========================================================

    row = {
        "age_at_transfer": age_at_transfer,
        "height": height,
        "matches": matches,
        "started": started,
        "goals": goals,
        "assists": assists,
        "minutes": minutes,
        "rating": rating,

        "is_same_league": int(
            pd.notna(from_league_id)
            and from_league_id == to_league_id
        ),

        "is_top5_destination": int(
            to_league_id
            in TOP5_LEAGUE_IDS
        ),

        "goals_per90": goals_per90,
        "assists_per90": assists_per90,
        "goal_contributions_per90": (
            goal_contributions_per90
        ),
        "starts_ratio": starts_ratio,
        "minutes_per_match": minutes_per_match,
        "age_squared": age_squared,

        "from_league_id": from_league_id,
        "to_league_id": to_league_id,

        "main_position": player.get(
            "main_position"
        ),

        "foot": player.get(
            "foot"
        ),
    }

    return pd.DataFrame(
        [row]
    )[MODEL_FEATURES]


# ============================================================
# Predict Only
# ============================================================

def predict_fee(
    prediction_input: pd.DataFrame,
) -> float:

    bundle = get_model()

    # ========================================================
    # v1.2 Ensemble 정보
    # ========================================================

    model_c = bundle["model_c"]
    model_d = bundle["model_d"]

    alpha_c = bundle["alpha_c"]
    alpha_d = bundle["alpha_d"]

    features_c = bundle["features_c"]
    features_d = bundle["features_d"]

    # ========================================================
    # Model C Input / Prediction
    # ========================================================

    input_c = prediction_input[
        features_c
    ]

    pred_c_log = model_c.predict(
        input_c
    )[0]

    pred_c = np.expm1(
        pred_c_log
    )

    pred_c = max(
        float(pred_c),
        0,
    )

    # ========================================================
    # Model D Input / Prediction
    # ========================================================

    input_d = prediction_input[
        features_d
    ]

    pred_d_log = model_d.predict(
        input_d
    )[0]

    pred_d = np.expm1(
        pred_d_log
    )

    pred_d = max(
        float(pred_d),
        0,
    )

    # ========================================================
    # v1.2 Final Ensemble
    # ========================================================

    predicted_fee = (
        alpha_c * pred_c
        + alpha_d * pred_d
    )

    print(
        f"Model C: {pred_c / 1_000_000:.2f}M | "
        f"Model D: {pred_d / 1_000_000:.2f}M | "
        f"Final: {predicted_fee / 1_000_000:.2f}M"
    )

    return float(
        predicted_fee
    )


# ============================================================
# Prediction Service
# ============================================================

def predict_transfer_fee(
    request: PredictionRequest,
) -> PredictionResponse:

    player = get_player(
        request.player_id
    )

    if player is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "예측 가능한 선수를 "
                "찾을 수 없습니다."
            ),
        )

    # --------------------------------------------------------
    # Prediction Input
    # --------------------------------------------------------

    prediction_input = build_prediction_input(
        player=player,
        request=request,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_fee = predict_fee(
        prediction_input
    )

    # v1.2 Ensemble SHAP은 별도 구현 예정
    explanation = []

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    age_at_transfer = float(
        prediction_input.iloc[0][
            "age_at_transfer"
        ]
    )

    return PredictionResponse(
        player_id=int(
            player["player_id"]
        ),

        player_name=str(
            player["player_name"]
        ),

        current_club_name=nullable_value(
            player.get(
                "current_club_name"
            )
        ),

        current_league_name=nullable_value(
            player.get(
                "current_league_name"
            )
        ),

        to_league_id=request.to_league_id,

        predicted_transfer_fee=predicted_fee,

        predicted_transfer_fee_million=(
            predicted_fee
            / 1_000_000
        ),

        age_at_transfer=age_at_transfer,

        explanation=explanation,
    )