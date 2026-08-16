import os
import re
import time
import unicodedata

import pandas as pd
import requests


TARGET_FILE = (
    "data/processed/"
    "target_european_performances.csv"
)

MAPPING_FILE = (
    "data/processed/"
    "player_mapping.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "european_competition_stats.csv"
)

FAILURE_FILE = (
    "data/processed/"
    "european_fotmob_stats_failures.csv"
)


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


COMPETITION_ALIASES = {
    "UCL": {
        "Champions League",
    },
    "UEL": {
        "Europa League",
    },
    "UECL": {
        "Conference League",
        "Europa Conference League",
        "UEFA Conference League",
    },
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


CHECKPOINT_INTERVAL = 20
REQUEST_DELAY = 0.3
MAX_RETRIES = 3


session = requests.Session()
session.headers.update(HEADERS)


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
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = session.get(
                url,
                params=params,
                timeout=20,
            )

            if response.status_code == 429:
                wait = (
                    int(
                        response.headers.get(
                            "Retry-After",
                            5,
                        )
                    )
                )

                print(
                    f"429 Too Many Requests "
                    f"- {wait}초 대기"
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            time.sleep(
                REQUEST_DELAY
            )

            return response.json()

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                wait = attempt * 2

                print(
                    f"API 요청 실패 "
                    f"({attempt}/{MAX_RETRIES}) "
                    f"- {wait}초 후 재시도"
                )

                time.sleep(wait)

    raise RuntimeError(
        f"API 요청 최종 실패: "
        f"{last_error}"
    )


def find_entry_id(
    player_data,
    target_season,
    european_competition,
):
    aliases = COMPETITION_ALIASES[
        european_competition
    ]

    normalized_aliases = {
        normalize_text(alias)
        for alias in aliases
    }

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

        for (
            tournament_index,
            tournament,
        ) in enumerate(tournaments):
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
                not in normalized_aliases
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

            return (
                str(entry_id),
                competition_name,
            )

    return None, None


def parse_number(value):
    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
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
    targets = {
        normalize_text(name)
        for name in target_names
    }

    def walk(value):
        if isinstance(
            value,
            dict,
        ):
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
                    else value.get(
                        "value"
                    )
                )

                if stat_value is None:
                    stat_value = (
                        value.get("total")
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

            for child in (
                value.values()
            ):
                result = walk(child)

                if result is not None:
                    return result

        elif isinstance(
            value,
            list,
        ):
            for child in value:
                result = walk(child)

                if result is not None:
                    return result

        return None

    return walk(data)


def unique_join(series):
    values = (
        series
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return " | ".join(values)


def prepare_targets(df):
    numeric_columns = [
        "nb_in_group",
        "nb_on_pitch",
        "goals",
        "assists",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # 같은 선수 / 시즌 / 대회에서
    # 겨울 이적 등으로 팀이 여러 개인 경우
    # 먼저 하나로 합친다.
    grouped = (
        df.groupby(
            [
                "player_id",
                "season_name",
                "competition_id",
                "european_competition",
            ],
            as_index=False,
        )
        .agg(
            competition_name=(
                "competition_name",
                "first",
            ),
            team_ids=(
                "team_id",
                unique_join,
            ),
            team_names=(
                "team_name",
                unique_join,
            ),
            source_rows=(
                "team_id",
                "size",
            ),
            matches=(
                "nb_in_group",
                lambda x: x.sum(
                    min_count=1
                ),
            ),
            started=(
                "nb_on_pitch",
                lambda x: x.sum(
                    min_count=1
                ),
            ),
            goals=(
                "goals",
                lambda x: x.sum(
                    min_count=1
                ),
            ),
            assists=(
                "assists",
                lambda x: x.sum(
                    min_count=1
                ),
            ),
        )
    )

    return grouped


def make_key(row):
    return (
        int(row["player_id"]),
        str(row["season_name"]),
        str(row["competition_id"]),
    )


def save_checkpoint(
    results,
    failures,
):
    if results:
        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False,
        )

    if failures:
        pd.DataFrame(
            failures
        ).to_csv(
            FAILURE_FILE,
            index=False,
        )


def main():
    targets = pd.read_csv(
        TARGET_FILE,
        low_memory=False,
    )

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    targets = prepare_targets(
        targets
    )

    print(
        "\n================================"
    )
    print(
        "European FotMob Stats"
    )
    print(
        "================================"
    )

    print(
        "집계 전 rows:",
        pd.read_csv(
            TARGET_FILE,
            low_memory=False,
        ).shape[0],
    )

    print(
        "집계 후 rows:",
        len(targets),
    )

    print(
        "고유 선수:",
        targets[
            "player_id"
        ].nunique(),
    )

    print(
        "\n대회별:"
    )

    print(
        targets[
            "european_competition"
        ]
        .value_counts()
        .to_string()
    )

    # ----------------------------------------
    # FotMob mapping 정리
    # ----------------------------------------

    mapping = mapping[
        [
            "player_id",
            "fotmob_id",
        ]
    ].copy()

    mapping = mapping.dropna(
        subset=[
            "player_id",
            "fotmob_id",
        ]
    )

    mapping[
        "player_id"
    ] = (
        mapping[
            "player_id"
        ]
        .astype(int)
    )

    mapping[
        "fotmob_id"
    ] = (
        mapping[
            "fotmob_id"
        ]
        .astype(int)
    )

    mapping = mapping.drop_duplicates(
        subset=[
            "player_id",
        ],
        keep="first",
    )

    fotmob_map = dict(
        zip(
            mapping["player_id"],
            mapping["fotmob_id"],
        )
    )

    # ----------------------------------------
    # 기존 결과 있으면 이어서 실행
    # ----------------------------------------

    results = []
    failures = []
    completed_keys = set()

    if os.path.exists(
        OUTPUT_FILE
    ):
        existing = pd.read_csv(
            OUTPUT_FILE,
            low_memory=False,
        )

        results = (
            existing
            .to_dict(
                orient="records"
            )
        )

        for _, row in (
            existing.iterrows()
        ):
            completed_keys.add(
                (
                    int(
                        row["player_id"]
                    ),
                    str(
                        row["season_name"]
                    ),
                    str(
                        row[
                            "competition_id"
                        ]
                    ),
                )
            )

        print(
            "\n기존 완료 rows:",
            len(
                completed_keys
            ),
        )

    if os.path.exists(
        FAILURE_FILE
    ):
        existing_failures = (
            pd.read_csv(
                FAILURE_FILE,
                low_memory=False,
            )
        )

        failures = (
            existing_failures
            .to_dict(
                orient="records"
            )
        )

    # playerData는 같은 선수에 대해
    # 여러 번 호출할 필요 없음
    player_data_cache = {}

    total = len(targets)

    # ----------------------------------------
    # 전체 수집
    # ----------------------------------------

    for index, row in (
        targets.iterrows()
    ):
        key = make_key(row)

        if key in completed_keys:
            continue

        player_id = int(
            row["player_id"]
        )

        season_name = str(
            row["season_name"]
        )

        competition_id = str(
            row["competition_id"]
        )

        european_competition = str(
            row[
                "european_competition"
            ]
        )

        print(
            f"\n[{index + 1}/{total}] "
            f"player={player_id} "
            f"{season_name} "
            f"{european_competition}"
        )

        fotmob_id = (
            fotmob_map.get(
                player_id
            )
        )

        if fotmob_id is None:
            print(
                "  ✗ FotMob mapping 없음"
            )

            failures.append(
                {
                    "player_id":
                        player_id,
                    "season_name":
                        season_name,
                    "competition_id":
                        competition_id,
                    "european_competition":
                        european_competition,
                    "reason":
                        "fotmob_mapping_missing",
                }
            )

            continue

        try:
            # --------------------------------
            # playerData
            # --------------------------------

            if (
                fotmob_id
                not in player_data_cache
            ):
                player_data_cache[
                    fotmob_id
                ] = get_json(
                    PLAYER_DATA_URL,
                    {
                        "id":
                            fotmob_id,
                    },
                )

            player_data = (
                player_data_cache[
                    fotmob_id
                ]
            )

            fotmob_season = (
                convert_season(
                    season_name
                )
            )

            (
                entry_id,
                fotmob_competition,
            ) = find_entry_id(
                player_data,
                fotmob_season,
                european_competition,
            )

            if entry_id is None:
                raise ValueError(
                    "season_competition_not_found"
                )

            # --------------------------------
            # playerStats
            # --------------------------------

            stats_data = get_json(
                PLAYER_STATS_URL,
                {
                    "playerId":
                        fotmob_id,
                    "seasonId":
                        entry_id,
                },
            )

            minutes = find_stat(
                stats_data,
                STAT_NAMES[
                    "minutes"
                ],
            )

            rating = find_stat(
                stats_data,
                STAT_NAMES[
                    "rating"
                ],
            )

            if (
                minutes is None
                and rating is None
            ):
                raise ValueError(
                    "minutes_rating_not_found"
                )

            result = {
                "player_id":
                    player_id,
                "fotmob_id":
                    fotmob_id,
                "season_name":
                    season_name,
                "competition_id":
                    competition_id,
                "competition_name":
                    row[
                        "competition_name"
                    ],
                "european_competition":
                    european_competition,
                "team_ids":
                    row[
                        "team_ids"
                    ],
                "team_names":
                    row[
                        "team_names"
                    ],
                "source_rows":
                    row[
                        "source_rows"
                    ],
                "matches":
                    row[
                        "matches"
                    ],
                "started":
                    row[
                        "started"
                    ],
                "goals":
                    row[
                        "goals"
                    ],
                "assists":
                    row[
                        "assists"
                    ],
                "minutes":
                    minutes,
                "rating":
                    rating,
                "fotmob_competition":
                    fotmob_competition,
                "entry_id":
                    entry_id,
            }

            results.append(
                result
            )

            completed_keys.add(
                key
            )

            print(
                f"  ✓ "
                f"minutes={minutes}, "
                f"rating={rating}, "
                f"entry={entry_id}"
            )

        except Exception as exc:
            print(
                "  ✗ 실패:",
                exc,
            )

            failures.append(
                {
                    "player_id":
                        player_id,
                    "fotmob_id":
                        fotmob_id,
                    "season_name":
                        season_name,
                    "competition_id":
                        competition_id,
                    "european_competition":
                        european_competition,
                    "reason":
                        str(exc),
                }
            )

        # ------------------------------------
        # checkpoint
        # ------------------------------------

        if (
            (index + 1)
            % CHECKPOINT_INTERVAL
            == 0
        ):
            save_checkpoint(
                results,
                failures,
            )

            print(
                "\n--- checkpoint 저장 ---"
            )

    # ----------------------------------------
    # 최종 저장
    # ----------------------------------------

    save_checkpoint(
        results,
        failures,
    )

    print(
        "\n================================"
    )

    print(
        "수집 완료"
    )

    print(
        "================================"
    )

    print(
        "성공:",
        len(results),
    )

    print(
        "실패:",
        len(failures),
    )

    print(
        "성공 파일:",
        OUTPUT_FILE,
    )

    print(
        "실패 파일:",
        FAILURE_FILE,
    )


if __name__ == "__main__":
    main()