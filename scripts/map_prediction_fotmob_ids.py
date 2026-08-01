import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TARGET_FILE = Path(
    "data/processed/prediction_mapping_targets.csv"
)

MAPPING_FILE = Path(
    "data/processed/prediction_player_mapping.csv"
)

FAIL_FILE = Path(
    "outputs/data_validation/"
    "prediction_fotmob_mapping_failures.csv"
)

REQUEST_DELAY = 1
SAVE_INTERVAL = 10
TEST_LIMIT = None

SEARCH_URL = (
    "https://www.fotmob.com/api/data/search/suggest"
)
PLAYER_DATA_URL = (
    "https://www.fotmob.com/api/data/playerData"
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
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(
        r"[^a-z0-9가-힣]",
        "",
        text,
    )


def normalize_date(value):
    date = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(date):
        return None

    return date.strftime("%Y-%m-%d")


def get_json(url, params):
    response = SESSION.get(
        url,
        params=params,
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

    if not isinstance(data, list):
        return []

    for group in data:
        if not isinstance(group, dict):
            continue

        suggestions = group.get(
            "suggestions",
            [],
        )

        for item in suggestions:
            if item.get("type") != "player":
                continue

            fotmob_id = item.get("id")

            if not fotmob_id:
                continue

            try:
                fotmob_id = int(fotmob_id)
            except (TypeError, ValueError):
                continue

            players[fotmob_id] = {
                "fotmob_id": fotmob_id,
                "fotmob_name": item.get("name"),
                "fotmob_team": item.get("teamName"),
            }

    return list(players.values())


def get_player_data(fotmob_id):
    return get_json(
        PLAYER_DATA_URL,
        {
            "id": fotmob_id,
        },
    )


def get_birth_date(player_data):
    birth_date = (
        player_data
        .get("birthDate", {})
        .get("utcTime")
    )

    if not birth_date:
        birth_date = (
            player_data
            .get("meta", {})
            .get("personJSONLD", {})
            .get("birthDate")
        )

    return normalize_date(birth_date)


def find_fotmob_player(
    player_name,
    profile_birth,
):
    candidates = search_players(
        player_name,
    )

    normalized_player_name = normalize_text(
        player_name,
    )

    same_name_candidates = [
        candidate
        for candidate in candidates
        if normalize_text(
            candidate["fotmob_name"]
        ) == normalized_player_name
    ]

    if not same_name_candidates:
        return None, "동일 이름 후보 없음"

    for candidate in same_name_candidates:
        try:
            player_data = get_player_data(
                candidate["fotmob_id"]
            )
        except requests.RequestException:
            continue

        candidate_birth = get_birth_date(
            player_data
        )

        # 생년월일이 있는 경우 반드시 일치해야 함
        if profile_birth is not None:
            if profile_birth != candidate_birth:
                time.sleep(REQUEST_DELAY)
                continue

        candidate["birth_date"] = candidate_birth

        return candidate, None

    return None, "이름 일치, 생년월일 불일치"


def empty_mapping_dataframe():
    return pd.DataFrame(
        columns=[
            "player_id",
            "fotmob_id",
            "fotmob_name",
            "birth_date",
        ]
    )


def empty_failure_dataframe():
    return pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "birth_date",
            "current_club_name",
            "current_league_name",
            "reason",
        ]
    )


def load_existing_mapping():
    if not MAPPING_FILE.exists():
        return empty_mapping_dataframe()

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    required_columns = [
        "player_id",
        "fotmob_id",
        "fotmob_name",
        "birth_date",
    ]

    for column in required_columns:
        if column not in mapping.columns:
            mapping[column] = pd.NA

    mapping["player_id"] = pd.to_numeric(
        mapping["player_id"],
        errors="coerce",
    )

    mapping["fotmob_id"] = pd.to_numeric(
        mapping["fotmob_id"],
        errors="coerce",
    )

    mapping = mapping.dropna(
        subset=[
            "player_id",
            "fotmob_id",
        ]
    )

    mapping["player_id"] = (
        mapping["player_id"].astype(int)
    )

    mapping["fotmob_id"] = (
        mapping["fotmob_id"].astype(int)
    )

    return (
        mapping[required_columns]
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

    required_columns = [
        "player_id",
        "player_name",
        "birth_date",
        "current_club_name",
        "current_league_name",
        "reason",
    ]

    for column in required_columns:
        if column not in failures.columns:
            failures[column] = pd.NA

    failures["player_id"] = pd.to_numeric(
        failures["player_id"],
        errors="coerce",
    )

    failures = failures.dropna(
        subset=["player_id"],
    )

    failures["player_id"] = (
        failures["player_id"].astype(int)
    )

    return failures[required_columns]


def save_results(
    mapping_rows,
    failure_rows,
):
    MAPPING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.DataFrame(
        mapping_rows
    )

    if mapping.empty:
        mapping = empty_mapping_dataframe()
    else:
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
        encoding="utf-8-sig",
    )

    failures = pd.DataFrame(
        failure_rows
    )

    if failures.empty:
        failures = empty_failure_dataframe()
    else:
        # 같은 선수를 재시도한 경우 최신 실패 사유 유지
        failures = (
            failures
            .drop_duplicates(
                subset=["player_id"],
                keep="last",
            )
            .sort_values("player_id")
        )

    # 성공한 선수는 과거 실패 목록에서 제거
    if not mapping.empty:
        completed_ids = set(
            mapping["player_id"]
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


def validate_targets(targets):
    required_columns = {
        "player_id",
        "player_name",
        "date_of_birth",
    }

    missing_columns = (
        required_columns
        - set(targets.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{TARGET_FILE} 컬럼 부족: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = int(
        targets["player_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "신규 매핑 대상 파일에 "
            f"player_id 중복이 {duplicate_count}건 있습니다."
        )


def main():
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"신규 매핑 대상 파일이 없습니다: "
            f"{TARGET_FILE}"
        )

    targets = pd.read_csv(
        TARGET_FILE,
        low_memory=False,
    )

    validate_targets(targets)

    targets["player_id"] = pd.to_numeric(
        targets["player_id"],
        errors="coerce",
    )

    targets = targets.dropna(
        subset=[
            "player_id",
            "player_name",
        ]
    ).copy()

    targets["player_id"] = (
        targets["player_id"].astype(int)
    )

    targets["player_name"] = (
        targets["player_name"]
        .apply(clean_name)
    )

    targets["birth_date"] = (
        targets["date_of_birth"]
        .apply(normalize_date)
    )

    existing_mapping = load_existing_mapping()
    existing_failures = load_existing_failures()

    mapping_rows = (
        existing_mapping.to_dict("records")
    )

    failure_rows = (
        existing_failures.to_dict("records")
    )

    completed_ids = set(
        existing_mapping["player_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    remaining_targets = targets[
        ~targets["player_id"].isin(
            completed_ids
        )
    ].copy()

    if TEST_LIMIT is not None:
        remaining_targets = (
            remaining_targets.head(TEST_LIMIT)
        )

    total_original = len(targets)
    total_completed = len(completed_ids)
    total_remaining = len(remaining_targets)

    print("=" * 70)
    print("Prediction 선수 FotMob ID 매핑")
    print("=" * 70)
    print(f"전체 신규 대상 : {total_original:,}")
    print(f"기존 성공 완료 : {total_completed:,}")
    print(f"이번 실행 대상 : {total_remaining:,}")
    print(f"기존 실패 기록 : {len(existing_failures):,}")
    print(f"성공 저장 파일 : {MAPPING_FILE}")
    print(f"실패 저장 파일 : {FAIL_FILE}")
    print("-" * 70)

    if total_remaining == 0:
        print("새로 처리할 선수가 없습니다.")
        return

    for index, row in enumerate(
        remaining_targets.itertuples(
            index=False
        ),
        start=1,
    ):
        player_id = int(row.player_id)
        player_name = row.player_name
        birth_date = row.birth_date

        current_club_name = getattr(
            row,
            "current_club_name",
            None,
        )

        current_league_name = getattr(
            row,
            "current_league_name",
            None,
        )

        print(
            f"[{index:,}/{total_remaining:,}] "
            f"{player_name} ({player_id})"
        )

        try:
            matched, failure_reason = (
                find_fotmob_player(
                    player_name,
                    birth_date,
                )
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
                        "current_club_name": (
                            current_club_name
                        ),
                        "current_league_name": (
                            current_league_name
                        ),
                        "reason": failure_reason,
                    }
                )

                print(
                    f"  실패: {failure_reason}"
                )

        except requests.RequestException as error:
            failure_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "birth_date": birth_date,
                    "current_club_name": (
                        current_club_name
                    ),
                    "current_league_name": (
                        current_league_name
                    ),
                    "reason": (
                        f"요청 오류: {error}"
                    ),
                }
            )

            print(f"  요청 오류: {error}")

        except (ValueError, TypeError) as error:
            failure_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "birth_date": birth_date,
                    "current_club_name": (
                        current_club_name
                    ),
                    "current_league_name": (
                        current_league_name
                    ),
                    "reason": (
                        f"데이터 처리 오류: {error}"
                    ),
                }
            )

            print(
                f"  데이터 처리 오류: {error}"
            )

        if index % SAVE_INTERVAL == 0:
            save_results(
                mapping_rows,
                failure_rows,
            )

            print(
                f"  중간 저장 완료 "
                f"({index:,}/{total_remaining:,})"
            )

        time.sleep(REQUEST_DELAY)

    save_results(
        mapping_rows,
        failure_rows,
    )

    final_mapping = load_existing_mapping()
    final_failures = load_existing_failures()

    mapped_target_count = int(
        targets["player_id"].isin(
            final_mapping["player_id"]
        ).sum()
    )

    unmapped_target_count = (
        len(targets)
        - mapped_target_count
    )

    mapping_rate = (
        mapped_target_count
        / len(targets)
        * 100
        if len(targets) > 0
        else 0
    )

    print()
    print("=" * 70)
    print("Prediction FotMob ID 매핑 작업 완료")
    print("=" * 70)
    print(
        f"신규 대상 전체    : "
        f"{len(targets):,}"
    )
    print(
        f"신규 매핑 성공    : "
        f"{mapped_target_count:,}"
    )
    print(
        f"신규 매핑 미완료  : "
        f"{unmapped_target_count:,}"
    )
    print(
        f"신규 매핑 성공률  : "
        f"{mapping_rate:.2f}%"
    )
    print(
        f"현재 실패 목록    : "
        f"{len(final_failures):,}"
    )
    print(f"성공 저장 파일    : {MAPPING_FILE}")
    print(f"실패 저장 파일    : {FAIL_FILE}")


if __name__ == "__main__":
    main()