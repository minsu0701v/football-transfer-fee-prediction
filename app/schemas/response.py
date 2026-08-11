from pydantic import BaseModel, Field


class PlayerSearchResult(BaseModel):
    player_id: int
    player_name: str

    player_image_url: str | None = None

    current_club_id: str | None = None
    current_club_name: str | None = None

    current_league_id: str | None = None
    current_league_name: str | None = None

    main_position: str | None = None

    season_name: str | None = None

    matches: float | None = None
    goals: float | None = None
    assists: float | None = None

    rating: float | None = None
    started: int | None = None
    minutes: float | None = None


class ExplanationItem(BaseModel):
    feature: str
    feature_name: str

    impact: float

    direction: str


class PredictionResponse(BaseModel):
    player_id: int

    player_name: str

    current_club_name: str | None = None

    current_league_name: str | None = None

    to_league_id: str

    predicted_transfer_fee: float

    predicted_transfer_fee_million: float

    age_at_transfer: float

    explanation: list[ExplanationItem] = Field(
        default_factory=list
    )