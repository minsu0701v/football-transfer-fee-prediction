from fastapi import APIRouter, HTTPException, Query

from app.schemas.response import PlayerSearchResult
from app.services.player_service import (
    get_player,
    search_players,
)
from app.utils.common import nullable_value


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/players",
    tags=["Players"],
)


# ============================================================
# Player Search
# ============================================================

@router.get(
    "/search",
    response_model=list[PlayerSearchResult],
)
def search_players_api(
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
    return search_players(
        keyword=q,
        limit=limit,
    )


# ============================================================
# Player Detail
# ============================================================

@router.get("/{player_id}")
def get_player_api(
    player_id: int,
):
    player = get_player(
        player_id
    )

    if player is None:
        raise HTTPException(
            status_code=404,
            detail="선수를 찾을 수 없습니다.",
        )

    return {
        column: nullable_value(value)
        for column, value
        in player.to_dict().items()
    }