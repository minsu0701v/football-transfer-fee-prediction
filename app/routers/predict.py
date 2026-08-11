from fastapi import APIRouter

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.services.prediction_service import (
    predict_transfer_fee,
)


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# ============================================================
# Transfer Fee Prediction
# ============================================================

@router.post(
    "",
    response_model=PredictionResponse,
)
def predict_api(
    request: PredictionRequest,
):
    return predict_transfer_fee(
        request
    )