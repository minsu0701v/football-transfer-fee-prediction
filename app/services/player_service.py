import pandas as pd

from app.config import (
    PREDICTION_DATA_FILE,
    TRAINING_DATA_FILE,
)
from app.schemas.response import PlayerSearchResult
from app.utils.common import nullable_value


# ============================================================
# Cache
# ============================================================

_prediction_df: pd.DataFrame | None = None
_teams_df: pd.DataFrame | None = None


# ============================================================
# Prediction Dataset
# ============================================================

def load_prediction_dataset() -> pd.DataFrame:
    global _prediction_df

    if _prediction_df is not None:
        return _prediction_df

    if not PREDICTION_DATA_FILE.exists():
        raise FileNotFoundError(
            "prediction_dataset.csv를 찾을 수 없습니다: "
            f"{PREDICTION_DATA_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_DATA_FILE,
        low_memory=False,
    )

    df["player_id"] = pd.to_numeric(
        df["player_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["player_id"]
    ).copy()

    df["player_id"] = (
        df["player_id"]
        .astype(int)
    )

    _prediction_df = df

    return _prediction_df


# ============================================================
# Team Dataset
# ============================================================

def load_teams() -> pd.DataFrame:
    global _teams_df

    if _teams_df is not None:
        return _teams_df

    if not TRAINING_DATA_FILE.exists():
        _teams_df = pd.DataFrame(
            columns=[
                "team_id",
                "team_name",
                "league_id",
                "league_name",
            ]
        )

        return _teams_df

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

    teams = pd.concat(
        [
            from_teams,
            to_teams,
        ],
        ignore_index=True,
    )

    teams = teams.dropna(
        subset=[
            "team_id",
            "team_name",
        ]
    )

    teams["team_id"] = (
        teams["team_id"]
        .astype(str)
        .str.strip()
    )

    teams["league_id"] = (
        teams["league_id"]
        .astype(str)
        .str.strip()
    )

    teams = teams.drop_duplicates(
        subset=["team_id"],
        keep="last",
    )

    _teams_df = (
        teams
        .sort_values("team_name")
        .reset_index(drop=True)
    )

    return _teams_df


# ============================================================
# Getter
# ============================================================

def get_prediction_dataset() -> pd.DataFrame:
    return load_prediction_dataset()


def get_team_dataset() -> pd.DataFrame:
    return load_teams()


# ============================================================
# Search Players
# ============================================================

def search_players(
    keyword: str,
    limit: int = 10,
) -> list[PlayerSearchResult]:

    df = get_prediction_dataset()

    results = df[
        df["player_name"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )
    ].head(limit)

    response = []

    for _, row in results.iterrows():

        current_club_id = (
            str(row.get("current_club_id")).strip()
            if pd.notna(row.get("current_club_id"))
            else None
        )

        current_league_id = (
            str(row.get("current_league_id")).strip()
            if pd.notna(row.get("current_league_id"))
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
                    row.get("player_image_url")
                ),
                current_club_id=current_club_id,
                current_club_name=nullable_value(
                    row.get("current_club_name")
                ),
                current_league_id=current_league_id,
                current_league_name=nullable_value(
                    row.get("current_league_name")
                ),
                main_position=nullable_value(
                    row.get("main_position")
                ),
                season_name=nullable_value(
                    row.get("season_name")
                ),

                matches=nullable_value(
                    row.get("matches")
                ),
                started=nullable_value(
                    row.get("started")
                ),
                minutes=nullable_value(
                    row.get("minutes")
                ),
                goals=nullable_value(
                    row.get("goals")
                ),
                assists=nullable_value(
                    row.get("assists")
                ),
                rating=nullable_value(
                    row.get("rating")
                ),
            )
        )

    return response


# ============================================================
# Player
# ============================================================

def get_player(
    player_id: int,
) -> pd.Series | None:

    df = get_prediction_dataset()

    result = df[
        df["player_id"] == player_id
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# Teams
# ============================================================

def search_teams(
    league_id: str | None = None,
    keyword: str | None = None,
    limit: int = 100,
) -> list[dict]:

    teams = get_team_dataset()

    if league_id:
        teams = teams[
            teams["league_id"]
            .astype(str)
            == league_id
        ]

    if keyword:
        teams = teams[
            teams["team_name"]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                na=False,
                regex=False,
            )
        ]

    teams = teams.head(limit)

    return [
        {
            column: nullable_value(value)
            for column, value
            in row.to_dict().items()
        }
        for _, row in teams.iterrows()
    ]