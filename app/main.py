from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    players,
    predict,
    teams,
)
from app.services.model_loader import load_model
from app.services.player_service import (
    load_prediction_dataset,
    load_teams,
)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 시작 시 모델과 데이터를 메모리에 미리 로드한다.
    """

    model = load_model()
    prediction_df = load_prediction_dataset()
    teams_df = load_teams()

    print()
    print("=" * 60)
    print("Football Transfer Fee Prediction API")
    print("=" * 60)

    print(
        f"모델 로드 완료 : "
        f"{model is not None}"
    )

    print(
        f"예측 선수 로드 : "
        f"{len(prediction_df):,}명"
    )

    print(
        f"팀 목록 로드   : "
        f"{len(teams_df):,}개"
    )

    print("=" * 60)

    yield


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Football Transfer Fee Prediction API",
    description=(
        "축구 선수의 프로필과 경기 데이터를 기반으로 "
        "예상 이적료를 예측하는 API"
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routers
# ============================================================

app.include_router(
    players.router
)

app.include_router(
    teams.router
)

app.include_router(
    predict.router
)


# ============================================================
# Root
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "message": (
            "Football Transfer Fee Prediction API"
        ),
        "version": "1.2.0",
    }


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    prediction_df = load_prediction_dataset()
    teams_df = load_teams()

    return {
        "status": "ok",
        "model_loaded": True,
        "player_count": len(prediction_df),
        "team_count": len(teams_df),
    }