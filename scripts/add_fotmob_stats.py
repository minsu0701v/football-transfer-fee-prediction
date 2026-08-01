import re
import time
import unicodedata

import pandas as pd
import requests


PERFORMANCE_FILE = (
    "data/processed/target_player_performances.csv"
)
MAPPING_FILE = "data/processed/player_mapping.csv"

OUTPUT_FILE = (
    "data/processed/player_performance_processed.csv"
)
FAIL_FILE = (
    "data/processed/fotmob_stats_failures.csv"
)

TEST_LIMIT = None
REQUEST_DELAY = 1

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


FINAL_COLUMNS = [
    "player_id",
    "season_name",
    "competition_id",
    "competition_name",
    "team_id",
    "team_name",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
]


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
    """
    23/24 -> 2023/2024
    24/25 -> 2024/2025
    """
    if pd.isna(value):
        return None

    try:
        start, end = (
            str(value)
            .strip()
            .split("/")
        )

        start = int(start)
        end = int(end)

        start_year = (
            2000 + start
            if start < 50
            else 1900 + start
        )

        end_year = (
            2000 + end
            if end < 50
            else 1900 + end
        )

        return f"{start_year}/{end_year}"

    except (
        ValueError,
        AttributeError,
    ):
        return None


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

    stat_seasons = player_data.get(
        "statSeasons",
        [],
    )

    for season_index, season in enumerate(
        stat_seasons
    ):
        season_name = (
            season.get("season")
            or season.get("seasonName")
            or season.get("name")
        )

        if (
            str(season_name)
            != str(target_season)
        ):
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
                or tournament.get(
                    "tournamentName"
                )
                or tournament.get(
                    "competitionName"
                )
                or tournament.get(
                    "leagueName"
                )
            )

            if (
                normalize_text(
                    competition_name
                )
                != target_competition
            ):
                continue

            entry_id = (
                tournament.get("entryId")
                or tournament.get(
                    "seasonId"
                )
                or (
                    f"{season_index}-"
                    f"{tournament_index}"
                )
            )

            return str(entry_id)

    return None


def parse_number(value):
    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        if pd.isna(value):
            return None

        return value

    text = (
        str(value)
        .strip()
        .replace(",", "")
    )

    if (
        not text
        or text == "-"
    ):
        return None

    try:
        return float(text)

    except ValueError:
        return None


def find_stat(
    data,
    target_names,
):
    normalized_targets = {
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
                in normalized_targets
            ):
                if "statValue" in value:
                    stat_value = value.get(
                        "statValue"
                    )
                else:
                    stat_value = value.get(
                        "value"
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
                        stat_value.get(
                            "value"
                        )
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


def create_result_row(
    row,
    minutes,
    rating,
):
    """
    기존 CSV에서 경기, 선발, 골, 도움을 가져오고
    FotMob에서 minutes와 rating만 추가한다.
    """
    return {
        "player_id": row["player_id"],
        "season_name": row["season_name"],
        "competition_id": row[
            "competition_id"
        ],
        "competition_name": row[
            "competition_name"
        ],
        "team_id": row["team_id"],
        "team_name": row["team_name"],

        # 기존 target_player_performances.csv
        "matches": row["nb_in_group"],
        "started": row["nb_on_pitch"],
        "goals": row["goals"],
        "assists": row["assists"],

        # FotMob에서 추가
        "minutes": minutes,
        "rating": rating,
    }


def main():
    performances = pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    required_columns = [
        "player_id",
        "season_name",
        "competition_id",
        "competition_name",
        "team_id",
        "team_name",
        "nb_in_group",
        "nb_on_pitch",
        "goals",
        "assists",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in performances.columns
    ]

    if missing_columns:
        raise KeyError(
            "performance 파일에 필요한 "
            "컬럼이 없습니다: "
            f"{missing_columns}"
        )

    data = performances.merge(
        mapping[
            [
                "player_id",
                "fotmob_id",
            ]
        ],
        on="player_id",
        how="inner",
    )

    data = data.dropna(
        subset=["fotmob_id"]
    ).copy()

    if TEST_LIMIT is not None:
        data = (
            data
            .head(TEST_LIMIT)
            .copy()
        )

    results = []
    failures = []

    total = len(data)

    for number, (_, row) in enumerate(
        data.iterrows(),
        start=1,
    ):
        player_id = row["player_id"]
        fotmob_id = int(
            row["fotmob_id"]
        )

        season = convert_season(
            row["season_name"]
        )

        competition = row[
            "competition_name"
        ]

        print("\n" + "=" * 50)
        print(
            f"[{number}/{total}] "
            f"player_id: {player_id}"
        )
        print(
            f"fotmob_id: {fotmob_id}"
        )
        print(f"season: {season}")
        print(
            f"competition: {competition}"
        )

        try:
            player_data = get_json(
                PLAYER_DATA_URL,
                {
                    "id": fotmob_id,
                },
            )

            entry_id = find_entry_id(
                player_data,
                season,
                competition,
            )

            if entry_id is None:
                raise ValueError(
                    "해당 시즌 리그 "
                    "entry_id 없음"
                )

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

            result = create_result_row(
                row,
                minutes,
                rating,
            )

            results.append(result)

            print(
                f"matches: "
                f"{result['matches']}"
            )
            print(
                f"started: "
                f"{result['started']}"
            )
            print(
                f"goals: "
                f"{result['goals']}"
            )
            print(
                f"assists: "
                f"{result['assists']}"
            )
            print(
                f"minutes: {minutes}"
            )
            print(
                f"rating: {rating}"
            )

            if (
                minutes is None
                or rating is None
            ):
                print(
                    "주의: FotMob 통계 일부 결측"
                )

        except Exception as error:
            failure = {
                "player_id": row[
                    "player_id"
                ],
                "fotmob_id": row[
                    "fotmob_id"
                ],
                "season_name": row[
                    "season_name"
                ],
                "competition_id": row[
                    "competition_id"
                ],
                "competition_name": row[
                    "competition_name"
                ],
                "team_id": row[
                    "team_id"
                ],
                "team_name": row[
                    "team_name"
                ],
                "failure_reason": str(
                    error
                ),
            }

            failures.append(failure)

            print(f"실패: {error}")

        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(
        results
    ).reindex(
        columns=FINAL_COLUMNS
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    failure_columns = [
        "player_id",
        "fotmob_id",
        "season_name",
        "competition_id",
        "competition_name",
        "team_id",
        "team_name",
        "failure_reason",
    ]

    failure_df = pd.DataFrame(
        failures
    ).reindex(
        columns=failure_columns
    )

    failure_df.to_csv(
        FAIL_FILE,
        index=False,
    )

    missing_minutes = (
        result_df["minutes"]
        .isna()
        .sum()
    )

    missing_rating = (
        result_df["rating"]
        .isna()
        .sum()
    )

    print("\n작업 완료")
    print(f"전체: {total}")
    print(
        f"성공 저장: {len(results)}"
    )
    print(
        f"API/entry_id 실패: "
        f"{len(failures)}"
    )
    print(
        f"minutes 결측: "
        f"{missing_minutes}"
    )
    print(
        f"rating 결측: "
        f"{missing_rating}"
    )
    print(
        f"결과 파일: {OUTPUT_FILE}"
    )
    print(
        f"실패 파일: {FAIL_FILE}"
    )


if __name__ == "__main__":
    main()