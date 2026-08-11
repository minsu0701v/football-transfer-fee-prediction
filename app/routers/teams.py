from fastapi import APIRouter, Query

from app.services.player_service import search_teams


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/teams",
    tags=["Teams"],
)


# ============================================================
# Team Search
# ============================================================

@router.get("")
def get_teams_api(
    league_id: str | None = None,
    q: str | None = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
):
    return search_teams(
        league_id=league_id,
        keyword=q,
        limit=limit,
    )