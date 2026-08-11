from pathlib import Path
import re

import pandas as pd


PROFILE_FILE = Path("data/raw/player_profiles.csv")
PERFORMANCE_FILE = Path("data/raw/player_performances.csv")

OUTPUT_FILE = Path(
    "data/processed/prediction_candidates.csv"
)

PROFILE_FAILURE_FILE = Path(
    "outputs/data_validation/prediction_profile_failures.csv"
)

TARGET_SEASON = "24/25"

TOP5_COMPETITIONS = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
}

PERFORMANCE_COLUMNS = [
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
    "minutes_played",
]

PROFILE_COLUMNS = [
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
]


def clean_player_name(value):
    if pd.isna(value):
        return value

    return re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        str(value),
    ).strip()


def load_top5_performances():
    filtered_chunks = []

    print(
        f"{TARGET_SEASON}시즌 "
        "5대 리그 기록을 검색합니다."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            PERFORMANCE_FILE,
            usecols=PERFORMANCE_COLUMNS,
            chunksize=200_000,
            low_memory=False,
        ),
        start=1,
    ):
        filtered = chunk[
            (chunk["season_name"] == TARGET_SEASON)
            & (
                chunk["competition_id"].isin(
                    TOP5_COMPETITIONS
                )
            )
        ].copy()

        if not filtered.empty:
            filtered_chunks.append(filtered)

        print(
            f"청크 {chunk_number}: "
            f"5대 리그 {len(filtered):,}행"
        )

    if not filtered_chunks:
        raise ValueError(
            f"{TARGET_SEASON}시즌 "
            "5대 리그 기록을 찾지 못했습니다."
        )

    performances = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

    numeric_columns = [
        "nb_in_group",
        "nb_on_pitch",
        "goals",
        "assists",
        "minutes_played",
    ]

    for column in numeric_columns:
        performances[column] = pd.to_numeric(
            performances[column],
            errors="coerce",
        )

    return performances


def select_one_row_per_player(
    performances: pd.DataFrame,
    profiles: pd.DataFrame,
):
    """
    같은 시즌에 여러 팀에서 뛴 선수는 한 행만 선택한다.

    우선순위:
    1. 프로필의 current_club_id와 team_id가 일치하는 기록
    2. 출전 시간이 가장 많은 기록
    3. 출전 경기 수가 가장 많은 기록
    """

    current_clubs = profiles[
        [
            "player_id",
            "current_club_id",
        ]
    ].copy()

    performances = performances.merge(
        current_clubs,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    performances["is_current_club"] = (
        performances["team_id"]
        == performances["current_club_id"]
    ).astype(int)

    performances = performances.sort_values(
        by=[
            "player_id",
            "is_current_club",
            "minutes_played",
            "nb_on_pitch",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
        na_position="last",
    )

    selected = performances.drop_duplicates(
        subset=["player_id"],
        keep="first",
    ).copy()

    return selected


def main():
    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"프로필 파일이 없습니다: {PROFILE_FILE}"
        )

    if not PERFORMANCE_FILE.exists():
        raise FileNotFoundError(
            f"성과 파일이 없습니다: {PERFORMANCE_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROFILE_FAILURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles = pd.read_csv(
        PROFILE_FILE,
        usecols=PROFILE_COLUMNS,
        low_memory=False,
    )

    profiles["player_name"] = (
        profiles["player_name"]
        .apply(clean_player_name)
    )

    profiles["date_of_birth"] = pd.to_datetime(
        profiles["date_of_birth"],
        errors="coerce",
    )

    profiles["height"] = pd.to_numeric(
        profiles["height"],
        errors="coerce",
    )

    performances = load_top5_performances()

    print()
    print("=" * 70)
    print("필터링 결과")
    print("=" * 70)
    print(
        f"{TARGET_SEASON} 5대 리그 성과 행: "
        f"{len(performances):,}"
    )
    print(
        f"고유 선수 수: "
        f"{performances['player_id'].nunique():,}"
    )

    selected_performances = select_one_row_per_player(
        performances=performances,
        profiles=profiles,
    )

    prediction_candidates = selected_performances.merge(
        profiles,
        on="player_id",
        how="left",
        suffixes=("", "_profile"),
        validate="one_to_one",
    )

    prediction_candidates = prediction_candidates.rename(
        columns={
            "competition_id": "current_league_id",
            "competition_name": "current_league_name",
            "team_id": "performance_team_id",
            "team_name": "performance_team_name",
            "nb_in_group": "raw_squad_appearances",
            "nb_on_pitch": "raw_appearances",
            "minutes_played": "raw_minutes",
            "goals": "raw_goals",
            "assists": "raw_assists",
        }
    )

    # 프로필 미매칭 선수 별도 저장
    missing_profiles = prediction_candidates[
        prediction_candidates["player_name"].isna()
    ].copy()

    missing_profiles.to_csv(
        PROFILE_FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # 프로필이 없는 선수는 최종 후보군에서 제외
    prediction_candidates = prediction_candidates[
        prediction_candidates["player_name"].notna()
    ].copy()

    # 이후 FotMob 매핑 및 기록 수집에서 채울 컬럼
    prediction_candidates["fotmob_id"] = pd.NA
    prediction_candidates["matches"] = pd.NA
    prediction_candidates["started"] = pd.NA
    prediction_candidates["minutes"] = pd.NA
    prediction_candidates["rating"] = pd.NA
    prediction_candidates["fotmob_goals"] = pd.NA
    prediction_candidates["fotmob_assists"] = pd.NA

    final_columns = [
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
        "performance_team_id",
        "performance_team_name",
        "season_name",
        "raw_squad_appearances",
        "raw_appearances",
        "raw_minutes",
        "raw_goals",
        "raw_assists",
        "fotmob_id",
        "matches",
        "started",
        "minutes",
        "rating",
        "fotmob_goals",
        "fotmob_assists",
    ]

    prediction_candidates = prediction_candidates[
        final_columns
    ].sort_values(
        by=[
            "current_league_id",
            "current_club_name",
            "player_name",
        ],
        na_position="last",
    )

    prediction_candidates.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_count = int(
        prediction_candidates["player_id"]
        .duplicated()
        .sum()
    )

    missing_profile_count = int(
        prediction_candidates["player_name"]
        .isna()
        .sum()
    )

    print()
    print("=" * 70)
    print("Prediction Candidate Dataset 생성 완료")
    print("=" * 70)
    print(
        f"최종 선수 수       : "
        f"{len(prediction_candidates):,}"
    )
    print(
        f"player_id 중복     : "
        f"{duplicate_count:,}"
    )
    print(
        f"최종 프로필 결측   : "
        f"{missing_profile_count:,}"
    )
    print(
        f"프로필 미매칭 제외 : "
        f"{len(missing_profiles):,}"
    )

    print()
    print("리그별 선수 수")

    print(
        prediction_candidates[
            "current_league_name"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(f"저장 파일     : {OUTPUT_FILE}")
    print(f"실패 목록 저장: {PROFILE_FAILURE_FILE}")

    if (
        duplicate_count == 0
        and missing_profile_count == 0
    ):
        print("PREDICTION_CANDIDATES_READY=True")
    else:
        print("PREDICTION_CANDIDATES_READY=False")


if __name__ == "__main__":
    main()