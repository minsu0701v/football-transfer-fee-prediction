import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CANDIDATE_FILE = Path(
    "data/processed/prediction_candidates.csv"
)

MAPPING_FILE = Path(
    "data/processed/prediction_all_player_mapping.csv"
)

OUTPUT_FILE = Path(
    "data/processed/prediction_dataset.csv"
)

FAIL_FILE = Path(
    "outputs/data_validation/"
    "prediction_fotmob_stats_failures.csv"
)

TARGET_SEASON = "24/25"

TEST_LIMIT = None
REQUEST_DELAY = 1
SAVE_INTERVAL = 10

PLAYER_DATA_URL = (
    "https://www.fotmob.com/api/data/playerData"
)

PLAYER_STATS_URL = (
    "https://www.fotmob.com/api/data/playerStats"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


FINAL_COLUMNS = [
    "player_id",
    "player_name",
    "player_image_url",
    "date_of_birth",
    "height",
    "citizenship",
    "main_position",
    "foot",
    "current_club_id",
    "current_club_name",
    "current_league_id",
    "current_league_name",
    "season_name",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "fotmob_id",
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


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


SESSION = create_session()


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value).strip().lower(),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(
        r"[^a-z0-9가-힣]",
        "",
        text,
    )


def convert_season(value):
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
    response = SESSION.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def find_entry_id(
    player_data,
    target_season,
    target_competition,
):
    target_competition_normalized = (
        normalize_text(target_competition)
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

        if str(season_name) != str(target_season):
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
                != target_competition_normalized
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

    if not text or text == "-":
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
                        or stat_value.get("total")
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
    return {
        "player_id": int(row["player_id"]),
        "player_name": row["player_name"],
        "player_image_url": row[
            "player_image_url"
        ],
        "date_of_birth": row[
            "date_of_birth"
        ],
        "height": row["height"],
        "citizenship": row[
            "citizenship"
        ],
        "main_position": row[
            "main_position"
        ],
        "foot": row["foot"],
        "current_club_id": row[
            "current_club_id"
        ],
        "current_club_name": row[
            "current_club_name"
        ],
        "current_league_id": row[
            "current_league_id"
        ],
        "current_league_name": row[
            "current_league_name"
        ],
        "season_name": row[
            "season_name"
        ],

        # 원본 성과 데이터
        "matches": row[
            "raw_squad_appearances"
        ],
        "started": row[
            "raw_appearances"
        ],
        "goals": row[
            "raw_goals"
        ],
        "assists": row[
            "raw_assists"
        ],

        # FotMob
        "minutes": minutes,
        "rating": rating,
        "fotmob_id": int(
            row["fotmob_id"]
        ),
    }


def empty_result_dataframe():
    return pd.DataFrame(
        columns=FINAL_COLUMNS
    )


def empty_failure_dataframe():
    return pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "fotmob_id",
            "season_name",
            "current_league_id",
            "current_league_name",
            "failure_reason",
        ]
    )


def load_existing_results():
    if not OUTPUT_FILE.exists():
        return empty_result_dataframe()

    results = pd.read_csv(
        OUTPUT_FILE,
        low_memory=False,
    )

    for column in FINAL_COLUMNS:
        if column not in results.columns:
            results[column] = pd.NA

    results["player_id"] = pd.to_numeric(
        results["player_id"],
        errors="coerce",
    )

    results = results.dropna(
        subset=["player_id"]
    )

    results["player_id"] = (
        results["player_id"].astype(int)
    )

    return (
        results[FINAL_COLUMNS]
        .drop_duplicates(
            subset=["player_id"],
            keep="last",
        )
    )


def load_existing_failures():
    if not FAIL_FILE.exists():
        return empty_failure_dataframe()

    failures = pd.read_csv(
        FAIL_FILE,
        low_memory=False,
    )

    failure_columns = (
        empty_failure_dataframe()
        .columns
        .tolist()
    )

    for column in failure_columns:
        if column not in failures.columns:
            failures[column] = pd.NA

    failures["player_id"] = pd.to_numeric(
        failures["player_id"],
        errors="coerce",
    )

    failures = failures.dropna(
        subset=["player_id"]
    )

    failures["player_id"] = (
        failures["player_id"].astype(int)
    )

    return failures[failure_columns]


def save_results(
    result_rows,
    failure_rows,
):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        result_rows
    )

    if results.empty:
        results = empty_result_dataframe()
    else:
        results = (
            results
            .reindex(columns=FINAL_COLUMNS)
            .drop_duplicates(
                subset=["player_id"],
                keep="last",
            )
            .sort_values("player_id")
        )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failures = pd.DataFrame(
        failure_rows
    )

    if failures.empty:
        failures = empty_failure_dataframe()
    else:
        failure_columns = (
            empty_failure_dataframe()
            .columns
            .tolist()
        )

        failures = (
            failures
            .reindex(columns=failure_columns)
            .drop_duplicates(
                subset=["player_id"],
                keep="last",
            )
            .sort_values("player_id")
        )

    completed_ids = set(
        results["player_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    failures = failures[
        ~failures["player_id"].isin(
            completed_ids
        )
    ]

    failures.to_csv(
        FAIL_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"후보 파일이 없습니다: "
            f"{CANDIDATE_FILE}"
        )

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"통합 매핑 파일이 없습니다: "
            f"{MAPPING_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        low_memory=False,
    )

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    data = candidates.drop(
        columns=["fotmob_id"],
        errors="ignore",
    ).merge(
        mapping[
            [
                "player_id",
                "fotmob_id",
            ]
        ],
        on="player_id",
        how="inner",
        validate="one_to_one",
    )

    data = data.dropna(
        subset=["fotmob_id"]
    ).copy()

    data["fotmob_id"] = pd.to_numeric(
        data["fotmob_id"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["fotmob_id"]
    )

    data["fotmob_id"] = (
        data["fotmob_id"].astype(int)
    )

    existing_results = load_existing_results()
    existing_failures = load_existing_failures()

    result_rows = (
        existing_results
        .to_dict("records")
    )

    failure_rows = (
        existing_failures
        .to_dict("records")
    )

    completed_ids = set(
        existing_results["player_id"]
        .astype(int)
        .tolist()
    )

    remaining_data = data[
        ~data["player_id"].isin(
            completed_ids
        )
    ].copy()

    if TEST_LIMIT is not None:
        remaining_data = (
            remaining_data.head(TEST_LIMIT)
        )

    total = len(remaining_data)

    print("=" * 70)
    print("Prediction 선수 FotMob 통계 수집")
    print("=" * 70)
    print(f"전체 매핑 선수 : {len(data):,}")
    print(f"기존 성공 완료 : {len(completed_ids):,}")
    print(f"이번 실행 대상 : {total:,}")
    print(f"기존 실패 기록 : {len(existing_failures):,}")
    print(f"결과 파일      : {OUTPUT_FILE}")
    print(f"실패 파일      : {FAIL_FILE}")
    print("-" * 70)

    try:
        for number, (_, row) in enumerate(
            remaining_data.iterrows(),
            start=1,
        ):
            player_id = int(
                row["player_id"]
            )

            fotmob_id = int(
                row["fotmob_id"]
            )

            season = convert_season(
                row["season_name"]
            )

            competition = row[
                "current_league_name"
            ]

            print(
                f"[{number:,}/{total:,}] "
                f"{row['player_name']} "
                f"({player_id})"
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
                        "해당 시즌 리그 entry_id 없음"
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

                result_rows.append(
                    result
                )

                print(
                    f"  성공: minutes={minutes}, "
                    f"rating={rating}"
                )

                if (
                    minutes is None
                    or rating is None
                ):
                    print(
                        "  주의: FotMob 통계 일부 결측"
                    )

            except Exception as error:
                failure_rows.append(
                    {
                        "player_id": player_id,
                        "player_name": row[
                            "player_name"
                        ],
                        "fotmob_id": fotmob_id,
                        "season_name": row[
                            "season_name"
                        ],
                        "current_league_id": row[
                            "current_league_id"
                        ],
                        "current_league_name": (
                            competition
                        ),
                        "failure_reason": str(
                            error
                        ),
                    }
                )

                print(f"  실패: {error}")

            if number % SAVE_INTERVAL == 0:
                save_results(
                    result_rows,
                    failure_rows,
                )

                print(
                    f"  중간 저장 완료 "
                    f"({number:,}/{total:,})"
                )

            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print()
        print(
            "사용자 중단 감지: "
            "현재 결과를 저장합니다."
        )

        save_results(
            result_rows,
            failure_rows,
        )

        print("중간 결과 저장 완료")
        print(
            "다시 실행하면 성공한 선수는 "
            "자동으로 건너뜁니다."
        )

        return

    save_results(
        result_rows,
        failure_rows,
    )

    final_results = load_existing_results()
    final_failures = load_existing_failures()

    missing_minutes = int(
        final_results["minutes"]
        .isna()
        .sum()
    )

    missing_rating = int(
        final_results["rating"]
        .isna()
        .sum()
    )

    print()
    print("=" * 70)
    print("Prediction Dataset 생성 완료")
    print("=" * 70)
    print(f"전체 매핑 선수 : {len(data):,}")
    print(f"성공 저장      : {len(final_results):,}")
    print(f"실패 목록      : {len(final_failures):,}")
    print(f"minutes 결측   : {missing_minutes:,}")
    print(f"rating 결측    : {missing_rating:,}")
    print(f"결과 파일      : {OUTPUT_FILE}")
    print(f"실패 파일      : {FAIL_FILE}")


if __name__ == "__main__":
    main()