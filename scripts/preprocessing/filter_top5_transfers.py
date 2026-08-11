import pandas as pd

TRANSFER_FILE = "data/raw/transfer_history.csv"
PERFORMANCE_FILE = "data/raw/player_performances.csv"
OUTPUT_FILE = "data/processed/top5_transfers.csv"

TOP5_LEAGUES = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
}


def previous_season(season):
    if pd.isna(season):
        return None

    try:
        start, end = str(season).strip().split("/")
        return f"{(int(start) - 1) % 100:02d}/{(int(end) - 1) % 100:02d}"

    except (ValueError, AttributeError):
        return None


def make_team_league_lookup(performances):
    """
    팀별·시즌별 대표 리그를 추정한다.

    한 팀이 리그, 컵, 유럽대항전 등에 동시에 참가하므로
    해당 시즌에 선수 출전 수(nb_on_pitch)의 합이 가장 큰 대회를
    대표 리그로 사용한다.
    """
    required_columns = {
        "team_id",
        "season_name",
        "competition_id",
        "competition_name",
        "nb_on_pitch",
    }

    missing_columns = required_columns - set(performances.columns)

    if missing_columns:
        raise KeyError(
            "도착 리그 추정에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    team_competitions = performances[
        [
            "team_id",
            "season_name",
            "competition_id",
            "competition_name",
            "nb_on_pitch",
        ]
    ].copy()

    team_competitions["team_id"] = pd.to_numeric(
        team_competitions["team_id"],
        errors="coerce",
    )

    team_competitions["nb_on_pitch"] = pd.to_numeric(
        team_competitions["nb_on_pitch"],
        errors="coerce",
    ).fillna(0)

    team_competitions = (
        team_competitions
        .dropna(
            subset=[
                "team_id",
                "season_name",
                "competition_id",
            ]
        )
        .groupby(
            [
                "team_id",
                "season_name",
                "competition_id",
                "competition_name",
            ],
            as_index=False,
        )
        .agg(total_appearances=("nb_on_pitch", "sum"))
    )

    representative_competitions = (
        team_competitions
        .sort_values(
            [
                "team_id",
                "season_name",
                "total_appearances",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "team_id",
                "season_name",
            ]
        )
        .rename(
            columns={
                "team_id": "to_team_id",
                "season_name": "season_name_transfer",
                "competition_id": "to_league_id",
                "competition_name": "to_league_name",
            }
        )
    )

    return representative_competitions[
        [
            "to_team_id",
            "season_name_transfer",
            "to_league_id",
            "to_league_name",
        ]
    ]


def main():
    transfers = pd.read_csv(
        TRANSFER_FILE,
        low_memory=False,
    )

    performances = pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce",
    )

    # 유상 완전이적만
    transfers = transfers[
        (transfers["transfer_type"] == "Transfer")
        & (transfers["transfer_fee"] > 0)
    ].copy()

    # 2020~2025년 여름 이적만
    transfers = transfers[
        transfers["transfer_date"].dt.year.between(2020, 2025)
        & transfers["transfer_date"].dt.month.between(6, 9)
    ].copy()

    # 이적 시즌의 직전 시즌 계산
    transfers["previous_season"] = transfers[
        "season_name"
    ].apply(previous_season)

    # ID 자료형 통일
    transfers["from_team_id"] = pd.to_numeric(
        transfers["from_team_id"],
        errors="coerce",
    )

    transfers["to_team_id"] = pd.to_numeric(
        transfers["to_team_id"],
        errors="coerce",
    )

    performances["team_id"] = pd.to_numeric(
        performances["team_id"],
        errors="coerce",
    )

    # 직전 시즌 5대리그 기록만 선택
    top5_performances = performances[
        performances["competition_id"].isin(TOP5_LEAGUES)
    ][
        [
            "player_id",
            "season_name",
            "team_id",
            "competition_id",
            "team_name",
        ]
    ].copy()

    top5_performances = top5_performances.rename(
        columns={
            "competition_id": "from_league_id",
        }
    )

    top5_performances["from_league_name"] = (
        top5_performances["from_league_id"]
        .map(TOP5_LEAGUES)
    )

    # 같은 선수의 같은 시즌·팀·리그 중복 제거
    top5_performances = top5_performances.drop_duplicates(
        subset=[
            "player_id",
            "season_name",
            "team_id",
            "from_league_id",
        ]
    )

    # 선수 + 직전 시즌 + 이적 전 팀 기준으로 매칭
    result = transfers.merge(
        top5_performances,
        left_on=[
            "player_id",
            "previous_season",
            "from_team_id",
        ],
        right_on=[
            "player_id",
            "season_name",
            "team_id",
        ],
        how="inner",
        suffixes=("_transfer", "_performance"),
    )

    # 도착 팀의 이적 시즌 대표 리그 조회표 생성
    to_league_lookup = make_team_league_lookup(
        performances
    )

    # 이적 시즌 + 도착 팀을 기준으로 도착 리그 매칭
    result = result.merge(
        to_league_lookup,
        on=[
            "to_team_id",
            "season_name_transfer",
        ],
        how="left",
    )

    # 동일 이적 건 중복 제거
    result = result.drop_duplicates(
        subset=[
            "player_id",
            "transfer_date",
            "from_team_id",
            "to_team_id",
        ]
    )

    result = result.sort_values(
        [
            "transfer_date",
            "transfer_fee",
        ],
        ascending=[
            True,
            False,
        ],
    )

    # 병합 확인용 및 중복 컬럼 제거
    result = result.drop(
        columns=[
            "season_name_performance",
            "team_id",
            "team_name",
            "transfer_type",
        ],
        errors="ignore",
    )

    # 이적 시즌 컬럼명 정리
    result = result.rename(
        columns={
            "season_name_transfer": "season_name",
        }
    )

    # 최종 CSV 컬럼
    final_columns = [
        "player_id",
        "season_name",
        "transfer_date",
        "previous_season",
        "from_team_id",
        "from_team_name",
        "from_league_id",
        "from_league_name",
        "to_team_id",
        "to_team_name",
        "to_league_id",
        "to_league_name",
        "value_at_transfer",
        "transfer_fee",
    ]

    result = result[final_columns]

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"5대리그 출발 이적 수: {len(result):,}")
    print(f"저장 완료: {OUTPUT_FILE}")

    print("\n미리보기")
    print("-" * 50)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()