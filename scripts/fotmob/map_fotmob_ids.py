import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests


PROFILE_FILE = "data/raw/player_profiles.csv"
TRANSFER_FILE = "data/processed/top5_transfers.csv"
MAPPING_FILE = "data/processed/player_mapping.csv"
FAIL_FILE = "data/processed/fotmob_mapping_failures.csv"

REQUEST_DELAY = 1
SAVE_INTERVAL = 10
TEST_LIMIT = None

SEARCH_URL = "https://www.fotmob.com/api/data/search/suggest"
PLAYER_DATA_URL = "https://www.fotmob.com/api/data/playerData"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


def clean_name(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        str(value).strip(),
    )


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

    return re.sub(r"[^a-z0-9가-힣]", "", text)


def normalize_date(value):
    date = pd.to_datetime(value, errors="coerce")

    if pd.isna(date):
        return None

    return date.strftime("%Y-%m-%d")


def get_json(url, params):
    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()

    return response.json()


def search_players(player_name):
    data = get_json(
        SEARCH_URL,
        {
            "hits": 50,
            "lang": "ko,en",
            "term": player_name,
        },
    )

    players = {}

    for group in data:
        for item in group.get("suggestions", []):
            if item.get("type") != "player":
                continue

            fotmob_id = item.get("id")

            if not fotmob_id:
                continue

            players[int(fotmob_id)] = {
                "fotmob_id": int(fotmob_id),
                "fotmob_name": item.get("name"),
                "fotmob_team": item.get("teamName"),
            }

    return list(players.values())


def get_player_data(fotmob_id):
    return get_json(
        PLAYER_DATA_URL,
        {"id": fotmob_id},
    )


def get_birth_date(player_data):
    value = (
        player_data
        .get("birthDate", {})
        .get("utcTime")
    )

    if not value:
        value = (
            player_data
            .get("meta", {})
            .get("personJSONLD", {})
            .get("birthDate")
        )

    return normalize_date(value)


def find_fotmob_player(player_name, profile_birth):
    candidates = search_players(player_name)

    for candidate in candidates:
        if (
            normalize_text(player_name)
            != normalize_text(candidate["fotmob_name"])
        ):
            continue

        try:
            player_data = get_player_data(
                candidate["fotmob_id"]
            )
        except requests.RequestException:
            continue

        candidate_birth = get_birth_date(player_data)

        if (
            profile_birth is not None
            and profile_birth == candidate_birth
        ):
            candidate["birth_date"] = candidate_birth
            return candidate

        time.sleep(REQUEST_DELAY)

    return None


def load_existing_mapping():
    if not Path(MAPPING_FILE).exists():
        return pd.DataFrame(
            columns=[
                "player_id",
                "fotmob_id",
                "fotmob_name",
                "birth_date",
            ]
        )

    mapping = pd.read_csv(MAPPING_FILE)

    for column in [
        "player_id",
        "fotmob_id",
        "fotmob_name",
        "birth_date",
    ]:
        if column not in mapping.columns:
            mapping[column] = pd.NA

    return mapping[
        [
            "player_id",
            "fotmob_id",
            "fotmob_name",
            "birth_date",
        ]
    ]


def save_results(mapping_rows, failure_rows):
    mapping = pd.DataFrame(mapping_rows)

    if not mapping.empty:
        mapping = (
            mapping
            .drop_duplicates(
                subset=["player_id"],
                keep="last",
            )
            .sort_values("player_id")
        )

        mapping.to_csv(
            MAPPING_FILE,
            index=False,
        )

    failures = pd.DataFrame(failure_rows)

    if not failures.empty:
        failures = (
            failures
            .drop_duplicates(
                subset=["player_id"],
                keep="last",
            )
            .sort_values("player_id")
        )

        failures.to_csv(
            FAIL_FILE,
            index=False,
        )


def main():
    profiles = pd.read_csv(
        PROFILE_FILE,
        low_memory=False,
    )

    transfers = pd.read_csv(
        TRANSFER_FILE,
        low_memory=False,
    )

    required_profile_columns = {
        "player_id",
        "player_name",
        "date_of_birth",
    }

    missing = (
        required_profile_columns
        - set(profiles.columns)
    )

    if missing:
        raise KeyError(
            f"{PROFILE_FILE} 컬럼 부족: "
            f"{sorted(missing)}"
        )

    target_ids = (
        transfers["player_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
    )

    targets = (
        profiles[
            profiles["player_id"].isin(target_ids)
        ][
            [
                "player_id",
                "player_name",
                "date_of_birth",
            ]
        ]
        .drop_duplicates(subset=["player_id"])
        .copy()
    )

    targets["player_name"] = (
        targets["player_name"]
        .apply(clean_name)
    )

    targets["birth_date"] = (
        targets["date_of_birth"]
        .apply(normalize_date)
    )

    existing = load_existing_mapping()

    mapping_rows = existing.to_dict("records")

    completed_ids = set(
        existing.loc[
            existing["fotmob_id"].notna(),
            "player_id",
        ]
        .astype(int)
        .tolist()
    )

    targets = targets[
        ~targets["player_id"].isin(completed_ids)
    ]

    if TEST_LIMIT is not None:
        targets = targets.head(TEST_LIMIT)

    failure_rows = []

    total = len(targets)

    print(f"전체 대상 선수: {len(target_ids)}")
    print(f"기존 매핑 완료: {len(completed_ids)}")
    print(f"이번 실행 대상: {total}")
    print("-" * 50)

    for index, row in enumerate(
        targets.itertuples(index=False),
        start=1,
    ):
        player_id = int(row.player_id)
        player_name = row.player_name
        birth_date = row.birth_date

        print(
            f"[{index}/{total}] "
            f"{player_name} ({player_id})"
        )

        try:
            matched = find_fotmob_player(
                player_name,
                birth_date,
            )

            if matched:
                mapping_rows.append(
                    {
                        "player_id": player_id,
                        "fotmob_id": matched[
                            "fotmob_id"
                        ],
                        "fotmob_name": matched[
                            "fotmob_name"
                        ],
                        "birth_date": matched[
                            "birth_date"
                        ],
                    }
                )

                print(
                    "  성공:",
                    matched["fotmob_id"],
                    matched["fotmob_name"],
                )

            else:
                failure_rows.append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "birth_date": birth_date,
                        "reason": "일치 후보 없음",
                    }
                )

                print("  실패: 일치 후보 없음")

        except requests.RequestException as error:
            failure_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "birth_date": birth_date,
                    "reason": str(error),
                }
            )

            print(f"  요청 오류: {error}")

        if index % SAVE_INTERVAL == 0:
            save_results(
                mapping_rows,
                failure_rows,
            )
            print("  중간 저장 완료")

        time.sleep(REQUEST_DELAY)

    save_results(
        mapping_rows,
        failure_rows,
    )

    final_mapping = pd.read_csv(MAPPING_FILE)

    print("-" * 50)
    print("매핑 작업 완료")
    print(
        "FotMob ID 보유:",
        final_mapping["fotmob_id"].notna().sum(),
    )
    print(
        "FotMob ID 미보유:",
        final_mapping["fotmob_id"].isna().sum(),
    )


if __name__ == "__main__":
    main()