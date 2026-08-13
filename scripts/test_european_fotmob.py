import re
import unicodedata

import pandas as pd
import requests


PERFORMANCE_FILE = "data/raw/player_performances.csv"
MAPPING_FILE = "data/processed/player_mapping.csv"

PLAYER_ID = 557149
SEASON_NAME = "24/25"
COMPETITION_ID = "CL"

PLAYER_DATA_URL = (
    "https://www.fotmob.com/api/data/playerData"
)

PLAYER_STATS_URL = (
    "https://www.fotmob.com/api/data/playerStats"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


STAT_NAMES = {
    "minutes": {
        "minutes played",
        "minutes",
        "출전 시간",
        "출전시간",
    },
    "rating": {
        "rating",
        "fotmob rating",
        "average rating",
        "평점",
    },
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value).strip().lower(),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(
        r"[^a-z0-9가-힣]",
        "",
        text,
    )


def convert_season(value):
    start, end = (
        str(value)
        .strip()
        .split("/")
    )

    return (
        f"20{int(start):02d}/"
        f"20{int(end):02d}"
    )


def get_json(url, params):
    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def find_entry_id(
    player_data,
    target_season,
    target_competition,
):
    target_competition = normalize_text(
        target_competition
    )

    for season_index, season in enumerate(
        player_data.get(
            "statSeasons",
            [],
        )
    ):
        season_name = (
            season.get("season")
            or season.get("seasonName")
            or season.get("name")
        )

        if str(season_name) != target_season:
            continue

        tournaments = (
            season.get("tournaments")
            or season.get("competitions")
            or season.get("items")
            or []
        )

        for tournament_index, tournament in enumerate(
            tournaments
        ):
            competition_name = (
                tournament.get("name")
                or tournament.get("tournamentName")
                or tournament.get("competitionName")
                or tournament.get("leagueName")
            )

            if (
                normalize_text(competition_name)
                != target_competition
            ):
                continue

            entry_id = (
                tournament.get("entryId")
                or tournament.get("seasonId")
                or (
                    f"{season_index}-"
                    f"{tournament_index}"
                )
            )

            return (
                str(entry_id),
                competition_name,
            )

    return None, None


def parse_number(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    text = (
        str(value)
        .strip()
        .replace(",", "")
    )

    if not text or text == "-":
        return None

    try:
        return float(text)

    except ValueError:
        return None


def find_stat(data, target_names):
    targets = {
        normalize_text(name)
        for name in target_names
    }

    def walk(value):
        if isinstance(value, dict):
            label = (
                value.get("title")
                or value.get("label")
                or value.get("name")
                or value.get("statName")
                or value.get("key")
            )

            if (
                label
                and normalize_text(label)
                in targets
            ):
                stat_value = (
                    value.get("statValue")
                    if "statValue" in value
                    else value.get("value")
                )

                if stat_value is None:
                    stat_value = value.get(
                        "total"
                    )

                if isinstance(
                    stat_value,
                    dict,
                ):
                    stat_value = (
                        stat_value.get("value")
                        or stat_value.get(
                            "total"
                        )
                    )

                return parse_number(
                    stat_value
                )

            for child in value.values():
                result = walk(child)

                if result is not None:
                    return result

        elif isinstance(value, list):
            for child in value:
                result = walk(child)

                if result is not None:
                    return result

        return None

    return walk(data)


def main():
    performances = pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    # ----------------------------------------
    # 1. 원본 유럽대항전 기록 찾기
    # ----------------------------------------

    target = performances[
        (performances["player_id"] == PLAYER_ID)
        & (
            performances["season_name"]
            == SEASON_NAME
        )
        & (
            performances["competition_id"]
            == COMPETITION_ID
        )
    ].copy()

    if target.empty:
        raise ValueError(
            "원본 유럽대항전 기록 없음"
        )

    row = target.iloc[0]

    print("\n================================")
    print("원본 데이터")
    print("================================")

    print(
        "player_id:",
        row["player_id"],
    )

    print(
        "team:",
        row["team_name"],
    )

    print(
        "season:",
        row["season_name"],
    )

    print(
        "competition:",
        row["competition_name"],
    )

    print(
        "matches:",
        row["nb_in_group"],
    )

    print(
        "started:",
        row["nb_on_pitch"],
    )

    print(
        "goals:",
        row["goals"],
    )

    print(
        "assists:",
        row["assists"],
    )

    # ----------------------------------------
    # 2. FotMob ID 찾기
    # ----------------------------------------

    player_mapping = mapping[
        mapping["player_id"]
        == PLAYER_ID
    ]

    if player_mapping.empty:
        raise ValueError(
            "FotMob 매핑 없음"
        )

    fotmob_id = int(
        player_mapping.iloc[0][
            "fotmob_id"
        ]
    )

    print(
        "\nFotMob ID:",
        fotmob_id,
    )

    # ----------------------------------------
    # 3. FotMob 시즌 / 대회 찾기
    # ----------------------------------------

    season = convert_season(
        SEASON_NAME
    )

    competition = row[
        "competition_name"
    ]

    player_data = get_json(
        PLAYER_DATA_URL,
        {
            "id": fotmob_id,
        },
    )

    entry_id, fotmob_competition = (
        find_entry_id(
            player_data,
            season,
            competition,
        )
    )

    if entry_id is None:
        raise ValueError(
            f"FotMob에서 "
            f"{season} {competition} "
            f"기록을 찾지 못함"
        )

    print(
        "FotMob competition:",
        fotmob_competition,
    )

    print(
        "entry_id:",
        entry_id,
    )

    # ----------------------------------------
    # 4. FotMob stats
    # ----------------------------------------

    stats_data = get_json(
        PLAYER_STATS_URL,
        {
            "playerId": fotmob_id,
            "seasonId": entry_id,
        },
    )

    minutes = find_stat(
        stats_data,
        STAT_NAMES["minutes"],
    )

    rating = find_stat(
        stats_data,
        STAT_NAMES["rating"],
    )

    # ----------------------------------------
    # 5. 최종 결과
    # ----------------------------------------

    result = {
        "player_id": int(
            row["player_id"]
        ),
        "fotmob_id": fotmob_id,
        "season_name": row[
            "season_name"
        ],
        "competition_id": row[
            "competition_id"
        ],
        "competition_name": row[
            "competition_name"
        ],
        "team_name": row[
            "team_name"
        ],
        "matches": row[
            "nb_in_group"
        ],
        "started": row[
            "nb_on_pitch"
        ],
        "goals": row[
            "goals"
        ],
        "assists": row[
            "assists"
        ],
        "minutes": minutes,
        "rating": rating,
    }

    print("\n================================")
    print("최종 매핑 결과")
    print("================================")

    for key, value in result.items():
        print(
            f"{key:20}: {value}"
        )

    output_file = (
        "data/processed/"
        "test_european_stats.csv"
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        output_file,
        index=False,
    )

    print(
        "\n저장 완료:",
        output_file,
    )
    for key, value in result.items():
        print(
            f"{key:20}: {value}"
        )


if __name__ == "__main__":
    main()