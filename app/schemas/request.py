from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    player_id: int = Field(
        ...,
        description="Prediction Dataset Player ID",
    )

    to_league_id: str = Field(
        ...,
        min_length=1,
        description="Destination League ID",
        examples=["GB1"],
    )

    to_team_id: str | None = Field(
        default=None,
        description="Destination Team ID",
        examples=["131"],
    )