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
    get_prediction_player_count,
    get_team_count,
)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 시작 시 모델을 로드하고
    PostgreSQL 데이터 상태를 확인한다.
    """

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    player_count = (
        get_prediction_player_count()
    )

    team_count = (
        get_team_count()
    )

    # --------------------------------------------------------
    # Startup Log
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "Football Transfer Fee Prediction API"
    )
    print("=" * 60)

    print(
        f"모델 로드 완료 : "
        f"{model is not None}"
    )

    print(
        f"예측 선수 수   : "
        f"{player_count:,}명"
    )

    print(
        f"팀 목록 수     : "
        f"{team_count:,}개"
    )

    print(
        "데이터 소스    : PostgreSQL"
    )

    print("=" * 60)

    yield


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title=(
        "Football Transfer Fee "
        "Prediction API"
    ),
    description=(
        "축구 선수의 프로필과 경기 데이터를 기반으로 "
        "예상 이적료를 예측하는 API"
    ),
    version="1.3.0",
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
            "Football Transfer Fee "
            "Prediction API"
        ),
        "version": "1.3.0",
        "data_source": "PostgreSQL",
    }


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():

    player_count = (
        get_prediction_player_count()
    )

    team_count = (
        get_team_count()
    )

    return {
        "status": "ok",
        "model_loaded": True,
        "database": "PostgreSQL",
        "player_count": player_count,
        "team_count": team_count,
    }