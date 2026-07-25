import pandas as pd

PERFORMANCE_FILE = "data/raw/player_performances.csv"
TRANSFER_FILE = "data/processed/top5_transfers.csv"
OUTPUT_FILE = "data/processed/target_player_performances.csv"

TOP5_LEAGUES = {"GB1", "ES1", "L1", "IT1", "FR1"}


def main():
    performances = pd.read_csv(PERFORMANCE_FILE, low_memory=False)
    transfers = pd.read_csv(TRANSFER_FILE, low_memory=False)

    target_pairs = (
        transfers[
            [
                "player_id",
                "previous_season",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    performances = performances[
        performances["competition_id"].isin(TOP5_LEAGUES)
    ].copy()

    result = performances.merge(
        target_pairs,
        left_on=[
            "player_id",
            "season_name",
        ],
        right_on=[
            "player_id",
            "previous_season",
        ],
        how="inner",
    )

    result = result.drop(columns=["previous_season"])

    result.to_csv(OUTPUT_FILE, index=False)

    print("기존 기록 수:", len(performances))
    print("필터링 후 기록 수:", len(result))
    print("대상 선수 수:", result["player_id"].nunique())
    print("저장 파일:", OUTPUT_FILE)

    print("\n대회별 기록 수")
    print(result["competition_id"].value_counts())


if __name__ == "__main__":
    main()