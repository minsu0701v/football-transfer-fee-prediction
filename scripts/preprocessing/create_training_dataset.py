from pathlib import Path

import pandas as pd


TRANSFER_FILE = Path("data/processed/top5_transfers.csv")
PERFORMANCE_FILE = Path("data/processed/player_performance_processed.csv")
PROFILE_FILE = Path("data/processed/player_profile_processed.csv")

OUTPUT_FILE = Path("data/processed/training_dataset.csv")


TOP5_LEAGUES = {"GB1", "ES1", "L1", "IT1", "FR1"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transfers = pd.read_csv(TRANSFER_FILE)
    performances = pd.read_csv(PERFORMANCE_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    print("[원본 데이터]")
    print(f"이적 데이터: {transfers.shape}")
    print(f"성과 데이터: {performances.shape}")
    print(f"프로필 데이터: {profiles.shape}")

    return transfers, performances, profiles


def validate_columns(
    transfers: pd.DataFrame,
    performances: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    transfer_columns = {
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
    }

    performance_columns = {
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
    }

    profile_columns = {
        "player_id",
        "player_name",
        "date_of_birth",
        "height",
        "citizenship",
        "main_position",
        "foot",
        "fotmob_id",
    }

    datasets = [
        ("top5_transfers.csv", transfers, transfer_columns),
        (
            "player_performance_processed.csv",
            performances,
            performance_columns,
        ),
        ("player_profile.csv", profiles, profile_columns),
    ]

    for file_name, dataframe, required_columns in datasets:
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"{file_name}에 필요한 컬럼이 없습니다: "
                f"{sorted(missing_columns)}"
            )


def prepare_data(
    transfers: pd.DataFrame,
    performances: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transfers = transfers.copy()
    performances = performances.copy()
    profiles = profiles.copy()

    transfers["player_id"] = pd.to_numeric(
        transfers["player_id"],
        errors="coerce",
    ).astype("Int64")

    performances["player_id"] = pd.to_numeric(
        performances["player_id"],
        errors="coerce",
    ).astype("Int64")

    profiles["player_id"] = pd.to_numeric(
        profiles["player_id"],
        errors="coerce",
    ).astype("Int64")

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce",
    )

    profiles["date_of_birth"] = pd.to_datetime(
        profiles["date_of_birth"],
        errors="coerce",
    )

    numeric_columns = [
        "matches",
        "started",
        "goals",
        "assists",
        "minutes",
        "rating",
    ]

    for column in numeric_columns:
        performances[column] = pd.to_numeric(
            performances[column],
            errors="coerce",
        )

    for column in ["height", "fotmob_id"]:
        profiles[column] = pd.to_numeric(
            profiles[column],
            errors="coerce",
        )

    for column in ["value_at_transfer", "transfer_fee"]:
        transfers[column] = pd.to_numeric(
            transfers[column],
            errors="coerce",
        )

    return transfers, performances, profiles


def merge_data(
    transfers: pd.DataFrame,
    performances: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    transfer_performance = transfers.merge(
        performances,
        how="inner",
        left_on=["player_id", "previous_season","from_team_id"],
        right_on=["player_id", "season_name","team_id"],
        suffixes=("_transfer", "_performance"),
        validate="many_to_many",
    )

    print("\n[이적 + 성과 병합]")
    print(f"병합 전 이적 행 수: {len(transfers):,}")
    print(f"병합 후 행 수: {len(transfer_performance):,}")

    training_data = transfer_performance.merge(
        profiles,
        how="inner",
        on="player_id",
        validate="many_to_one",
    )

    print("\n[프로필 병합]")
    print(f"프로필 병합 후 행 수: {len(training_data):,}")

    return training_data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    age_days = (
        data["transfer_date"] - data["date_of_birth"]
    ).dt.days

    data["age_at_transfer"] = age_days / 365.25

    data["is_same_league"] = (
        data["from_league_id"] == data["to_league_id"]
    ).astype(int)

    data["is_top5_destination"] = (
        data["to_league_id"].isin(TOP5_LEAGUES)
    ).astype(int)

    return data


def select_columns(data: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "player_id",
        "player_name",
        "season_name_transfer",
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
        "date_of_birth",
        "height",
        "citizenship",
        "main_position",
        "foot",
        "fotmob_id",
        "age_at_transfer",
        "is_same_league",
        "is_top5_destination",
        "transfer_fee",
    ]

    return data[selected_columns].copy()


def print_summary(data: pd.DataFrame) -> None:
    print("\n[최종 학습 데이터]")
    print(f"행 수: {len(data):,}")
    print(f"컬럼 수: {len(data.columns)}")

    print("\n[컬럼 목록]")
    for column in data.columns:
        print(f"- {column}")

    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\n[결측치]")
    if missing.empty:
        print("결측치 없음")
    else:
        print(missing.to_string())

    duplicate_count = data.duplicated().sum()

    print("\n[완전 중복 행]")
    print(f"{duplicate_count:,}건")

    print("\n[transfer_fee 요약]")
    print(data["transfer_fee"].describe())

    print("\n[샘플]")
    print(data.head().to_string(index=False))


def main() -> None:
    transfers, performances, profiles = load_data()

    validate_columns(
        transfers,
        performances,
        profiles,
    )

    transfers, performances, profiles = prepare_data(
        transfers,
        performances,
        profiles,
    )

    training_data = merge_data(
        transfers,
        performances,
        profiles,
    )

    training_data = create_features(training_data)
    training_data = select_columns(training_data)

    training_data = training_data.sort_values(
        by=["transfer_date", "player_id"],
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(training_data)

    print(f"\n저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()