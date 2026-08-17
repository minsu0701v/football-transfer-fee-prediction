import pandas as pd

from sqlalchemy import text

from app.database import get_engine
from app.schemas.response import PlayerSearchResult
from app.utils.common import nullable_value


# ============================================================
# Search Players
# ============================================================

def search_players(
    keyword: str,
    limit: int = 10,
) -> list[PlayerSearchResult]:

    query = text(
        """
        SELECT
            player_id,
            player_name,
            player_image_url,
            current_club_id,
            current_club_name,
            current_league_id,
            current_league_name,
            main_position,
            season_name,
            matches,
            started,
            minutes,
            goals,
            assists,
            rating
        FROM prediction_players
        WHERE player_name ILIKE :keyword
        ORDER BY player_name
        LIMIT :limit
        """
    )

    results = pd.read_sql_query(
        query,
        con=get_engine(),
        params={
            "keyword": f"%{keyword}%",
            "limit": limit,
        },
    )

    response = []

    for _, row in results.iterrows():

        current_club_id = (
            str(
                row.get(
                    "current_club_id"
                )
            ).strip()
            if pd.notna(
                row.get(
                    "current_club_id"
                )
            )
            else None
        )

        current_league_id = (
            str(
                row.get(
                    "current_league_id"
                )
            ).strip()
            if pd.notna(
                row.get(
                    "current_league_id"
                )
            )
            else None
        )

        response.append(
            PlayerSearchResult(
                player_id=int(
                    row["player_id"]
                ),

                player_name=str(
                    row["player_name"]
                ),

                player_image_url=nullable_value(
                    row.get(
                        "player_image_url"
                    )
                ),

                current_club_id=(
                    current_club_id
                ),

                current_club_name=nullable_value(
                    row.get(
                        "current_club_name"
                    )
                ),

                current_league_id=(
                    current_league_id
                ),

                current_league_name=nullable_value(
                    row.get(
                        "current_league_name"
                    )
                ),

                main_position=nullable_value(
                    row.get(
                        "main_position"
                    )
                ),

                season_name=nullable_value(
                    row.get(
                        "season_name"
                    )
                ),

                matches=nullable_value(
                    row.get(
                        "matches"
                    )
                ),

                started=nullable_value(
                    row.get(
                        "started"
                    )
                ),

                minutes=nullable_value(
                    row.get(
                        "minutes"
                    )
                ),

                goals=nullable_value(
                    row.get(
                        "goals"
                    )
                ),

                assists=nullable_value(
                    row.get(
                        "assists"
                    )
                ),

                rating=nullable_value(
                    row.get(
                        "rating"
                    )
                ),
            )
        )

    return response


# ============================================================
# Get Player
# ============================================================

def get_player(
    player_id: int,
) -> pd.Series | None:

    query = text(
        """
        SELECT *
        FROM prediction_players
        WHERE player_id = :player_id
        LIMIT 1
        """
    )

    result = pd.read_sql_query(
        query,
        con=get_engine(),
        params={
            "player_id": player_id,
        },
    )

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# Search Teams
# ============================================================

def search_teams(
    league_id: str | None = None,
    keyword: str | None = None,
    limit: int = 100,
) -> list[dict]:

    query = text(
        """
        WITH teams AS (

            SELECT
                from_team_id::text AS team_id,
                from_team_name AS team_name,
                from_league_id AS league_id,
                from_league_name AS league_name,
                1 AS priority
            FROM training_data
            WHERE
                from_team_id IS NOT NULL
                AND from_team_name IS NOT NULL

            UNION ALL

            SELECT
                to_team_id::text AS team_id,
                to_team_name AS team_name,
                to_league_id AS league_id,
                to_league_name AS league_name,
                2 AS priority
            FROM training_data
            WHERE
                to_team_id IS NOT NULL
                AND to_team_name IS NOT NULL
        ),

        deduplicated AS (

            SELECT DISTINCT ON (team_id)
                team_id,
                team_name,
                league_id,
                league_name
            FROM teams
            ORDER BY
                team_id,
                priority DESC
        )

        SELECT
            team_id,
            team_name,
            league_id,
            league_name
        FROM deduplicated
        WHERE
            (
                :league_id IS NULL
                OR league_id = :league_id
            )
            AND (
                :keyword IS NULL
                OR team_name ILIKE :keyword
            )
        ORDER BY team_name
        LIMIT :limit
        """
    )

    teams = pd.read_sql_query(
        query,
        con=get_engine(),
        params={
            "league_id": league_id,

            "keyword": (
                f"%{keyword}%"
                if keyword
                else None
            ),

            "limit": limit,
        },
    )

    return [
        {
            column: nullable_value(
                value
            )
            for column, value
            in row.to_dict().items()
        }
        for _, row in teams.iterrows()
    ]


# ============================================================
# Database Counts
# ============================================================

def get_prediction_player_count() -> int:
    """
    prediction_players 테이블의
    전체 선수 수를 반환한다.
    """

    query = text(
        """
        SELECT COUNT(*)
        FROM prediction_players
        """
    )

    with get_engine().connect() as connection:

        result = connection.execute(
            query
        )

        count = result.scalar_one()

    return int(count)


def get_team_count() -> int:
    """
    training_data의 출발/도착 팀을 합친 뒤
    team_id 기준 중복 제거한 팀 수를 반환한다.

    기존 load_teams()의
    drop_duplicates(team_id, keep="last")
    구조와 같은 개념이다.
    """

    query = text(
        """
        WITH teams AS (

            SELECT
                from_team_id::text AS team_id,
                1 AS priority
            FROM training_data
            WHERE
                from_team_id IS NOT NULL
                AND from_team_name IS NOT NULL

            UNION ALL

            SELECT
                to_team_id::text AS team_id,
                2 AS priority
            FROM training_data
            WHERE
                to_team_id IS NOT NULL
                AND to_team_name IS NOT NULL
        ),

        deduplicated AS (

            SELECT DISTINCT ON (team_id)
                team_id
            FROM teams
            ORDER BY
                team_id,
                priority DESC
        )

        SELECT COUNT(*)
        FROM deduplicated
        """
    )

    with get_engine().connect() as connection:

        result = connection.execute(
            query
        )

        count = result.scalar_one()

    return int(count)