from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transfer_fee_model_v1.joblib"
)

PREDICTION_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_dataset.csv"
)

TRAINING_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset.csv"
)


# ============================================================
# 예측 설정
# ============================================================

PREDICTION_DATE = pd.Timestamp("2026-07-01")

TOP5_LEAGUE_IDS = {
    "GB1",
    "ES1",
    "L1",
    "IT1",
    "FR1",
}

MODEL_FEATURES = [
    "value_at_transfer",
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
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]


# ============================================================
# 전역 저장소
# ============================================================

model = None
prediction_df: pd.DataFrame | None = None
teams_df: pd.DataFrame | None = None


# ============================================================
# Pydantic 스키마
# ============================================================

class PredictionRequest(BaseModel):
    player_id: int
    to_league_id: str = Field(
        min_length=1,
        examples=["ES1"],
    )
    value_at_transfer: float = Field(
        gt=0,
        examples=[30_000_000],
        description="현재 시장가, 유로 단위",
    )

    # v1에서는 사용하지 않지만 v1.1을 위해 미리 받는다.
    to_team_id: str | None = Field(
        default=None,
        examples=["131"],
    )


class PlayerSearchResult(BaseModel):
    player_id: int
    player_name: str
    player_image_url: str | None
    current_club_id: str | None
    current_club_name: str | None
    current_league_id: str | None
    current_league_name: str | None
    main_position: str | None
    season_name: str | None
    matches: float | None
    goals: float | None
    assists: float | None
    rating: float | None


class PredictionResponse(BaseModel):
    player_id: int
    player_name: str
    current_club_name: str | None
    current_league_name: str | None
    to_league_id: str
    value_at_transfer: float
    predicted_transfer_fee: float
    predicted_transfer_fee_million: float
    age_at_transfer: float


# ============================================================
# 데이터 변환 함수
# ============================================================

def nullable_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def calculate_age(
    date_of_birth: Any,
    prediction_date: pd.Timestamp,
) -> float:
    birth_date = pd.to_datetime(
        date_of_birth,
        errors="coerce",
    )

    if pd.isna(birth_date):
        return np.nan

    return (
        prediction_date - birth_date
    ).days / 365.25


def load_teams() -> pd.DataFrame:
    if not TRAINING_DATA_FILE.exists():
        return pd.DataFrame(
            columns=[
                "team_id",
                "team_name",
                "league_id",
                "league_name",
            ]
        )

    training_df = pd.read_csv(
        TRAINING_DATA_FILE,
        low_memory=False,
    )

    from_teams = training_df[
        [
            "from_team_id",
            "from_team_name",
            "from_league_id",
            "from_league_name",
        ]
    ].copy()

    from_teams.columns = [
        "team_id",
        "team_name",
        "league_id",
        "league_name",
    ]

    to_teams = training_df[
        [
            "to_team_id",
            "to_team_name",
            "to_league_id",
            "to_league_name",
        ]
    ].copy()

    to_teams.columns = [
        "team_id",
        "team_name",
        "league_id",
        "league_name",
    ]

    result = pd.concat(
        [from_teams, to_teams],
        ignore_index=True,
    )

    result = result.dropna(
        subset=["team_id", "team_name"]
    )

    result["team_id"] = (
        result["team_id"]
        .astype(str)
        .str.strip()
    )

    result = result.drop_duplicates(
        subset=["team_id"],
        keep="last",
    )

    return result.sort_values(
        "team_name"
    ).reset_index(drop=True)


def build_prediction_input(
    player: pd.Series,
    request: PredictionRequest,
) -> pd.DataFrame:
    from_league_id = str(
        player["current_league_id"]
    ).strip()

    row = {
        "value_at_transfer": request.value_at_transfer,
        "age_at_transfer": calculate_age(
            player["date_of_birth"],
            PREDICTION_DATE,
        ),
        "height": pd.to_numeric(
            player["height"],
            errors="coerce",
        ),
        "matches": pd.to_numeric(
            player["matches"],
            errors="coerce",
        ),
        "started": pd.to_numeric(
            player["started"],
            errors="coerce",
        ),
        "goals": pd.to_numeric(
            player["goals"],
            errors="coerce",
        ),
        "assists": pd.to_numeric(
            player["assists"],
            errors="coerce",
        ),
        "minutes": pd.to_numeric(
            player["minutes"],
            errors="coerce",
        ),
        "rating": pd.to_numeric(
            player["rating"],
            errors="coerce",
        ),
        "is_same_league": int(
            from_league_id == request.to_league_id
        ),
        "is_top5_destination": int(
            request.to_league_id in TOP5_LEAGUE_IDS
        ),
        "from_league_id": from_league_id,
        "to_league_id": request.to_league_id,
        "main_position": player["main_position"],
        "foot": player["foot"],
    }

    return pd.DataFrame([row])[MODEL_FEATURES]


# ============================================================
# 앱 시작 시 파일 로드
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global prediction_df
    global teams_df

    if not MODEL_FILE.exists():
        raise RuntimeError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_FILE}"
        )

    if not PREDICTION_DATA_FILE.exists():
        raise RuntimeError(
            "prediction_dataset.csv를 찾을 수 없습니다: "
            f"{PREDICTION_DATA_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    prediction_df = pd.read_csv(
        PREDICTION_DATA_FILE,
        low_memory=False,
    )

    prediction_df["player_id"] = pd.to_numeric(
        prediction_df["player_id"],
        errors="coerce",
    )

    prediction_df = prediction_df.dropna(
        subset=["player_id"]
    ).copy()

    prediction_df["player_id"] = (
        prediction_df["player_id"].astype(int)
    )

    teams_df = load_teams()

    print(
        f"모델 로드 완료: {MODEL_FILE.name}"
    )
    print(
        f"예측 선수 로드: {len(prediction_df):,}명"
    )
    print(
        f"팀 목록 로드  : {len(teams_df):,}개"
    )

    yield


app = FastAPI(
    title="Football Transfer Fee Prediction API",
    version="1.0.0",
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
# 엔드포인트
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Football Transfer Fee Prediction API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "player_count": (
            len(prediction_df)
            if prediction_df is not None
            else 0
        ),
    }


@app.get(
    "/players/search",
    response_model=list[PlayerSearchResult],
)
def search_players(
    q: str = Query(
        min_length=1,
        description="선수 이름 검색어",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
):
    if prediction_df is None:
        raise HTTPException(
            status_code=503,
            detail="선수 데이터가 로드되지 않았습니다.",
        )

    results = prediction_df[
        prediction_df["player_name"]
        .astype(str)
        .str.contains(
            q,
            case=False,
            na=False,
            regex=False,
        )
    ].head(limit)

    response = []

    for _, row in results.iterrows():
        response.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row["player_name"]),
                "player_image_url": nullable_value(
                    row.get("player_image_url")
                ),
                "current_club_id": (
                    str(row.get("current_club_id"))
                    if pd.notna(row.get("current_club_id"))
                    else None
                ),
                "current_club_name": nullable_value(
                    row.get("current_club_name")
                ),
                "current_league_id": (
                    str(row.get("current_league_id"))
                    if pd.notna(row.get("current_league_id"))
                    else None
                ),
                "current_league_name": nullable_value(
                    row.get("current_league_name")
                ),
                "main_position": nullable_value(
                    row.get("main_position")
                ),
                "season_name": nullable_value(
                    row.get("season_name")
                ),
                "matches": nullable_value(
                    row.get("matches")
                ),
                "goals": nullable_value(
                    row.get("goals")
                ),
                "assists": nullable_value(
                    row.get("assists")
                ),
                "rating": nullable_value(
                    row.get("rating")
                ),
            }
        )

    return response


@app.get("/players/{player_id}")
def get_player(player_id: int):
    if prediction_df is None:
        raise HTTPException(
            status_code=503,
            detail="선수 데이터가 로드되지 않았습니다.",
        )

    results = prediction_df[
        prediction_df["player_id"] == player_id
    ]

    if results.empty:
        raise HTTPException(
            status_code=404,
            detail="선수를 찾을 수 없습니다.",
        )

    player = results.iloc[0]

    return {
        column: nullable_value(value)
        for column, value in player.to_dict().items()
    }


@app.get("/teams")
def get_teams(
    league_id: str | None = None,
    q: str | None = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
):
    if teams_df is None:
        raise HTTPException(
            status_code=503,
            detail="팀 데이터가 로드되지 않았습니다.",
        )

    results = teams_df.copy()

    if league_id:
        results = results[
            results["league_id"].astype(str)
            == league_id
        ]

    if q:
        results = results[
            results["team_name"]
            .astype(str)
            .str.contains(
                q,
                case=False,
                na=False,
                regex=False,
            )
        ]

    results = results.head(limit)

    return [
        {
            column: nullable_value(value)
            for column, value in row.to_dict().items()
        }
        for _, row in results.iterrows()
    ]


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_transfer_fee(
    request: PredictionRequest,
):
    if model is None or prediction_df is None:
        raise HTTPException(
            status_code=503,
            detail="모델 또는 데이터가 로드되지 않았습니다.",
        )

    results = prediction_df[
        prediction_df["player_id"]
        == request.player_id
    ]

    if results.empty:
        raise HTTPException(
            status_code=404,
            detail="예측 가능한 선수를 찾을 수 없습니다.",
        )

    player = results.iloc[0]

    prediction_input = build_prediction_input(
        player=player,
        request=request,
    )

    predicted_log = model.predict(
        prediction_input
    )[0]

    predicted_fee = float(
        max(
            np.expm1(predicted_log),
            0,
        )
    )

    age = float(
        prediction_input.iloc[0][
            "age_at_transfer"
        ]
    )

    return {
        "player_id": int(player["player_id"]),
        "player_name": str(player["player_name"]),
        "current_club_name": nullable_value(
            player.get("current_club_name")
        ),
        "current_league_name": nullable_value(
            player.get("current_league_name")
        ),
        "to_league_id": request.to_league_id,
        "value_at_transfer": request.value_at_transfer,
        "predicted_transfer_fee": predicted_fee,
        "predicted_transfer_fee_million": (
            predicted_fee / 1_000_000
        ),
        "age_at_transfer": age,
    }