from pathlib import Path

import pandas as pd

DATA_FILE = Path("data/processed/training_dataset.csv")


def main():
    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("Training Dataset")
    print("=" * 60)

    print(f"전체 행 수 : {len(df):,}")

    transfer_key = [
        "player_id",
        "transfer_date",
        "from_team_id",
        "to_team_id",
    ]

    unique_transfers = (
        df[transfer_key]
        .drop_duplicates()
    )

    print(f"고유 이적 건수 : {len(unique_transfers):,}")

    duplicated = df.duplicated(
        subset=transfer_key,
        keep=False,
    )

    duplicated_df = (
        df.loc[duplicated]
        .sort_values(transfer_key)
    )

    print(f"\n중복된 이적 행 : {len(duplicated_df):,}")

    if len(duplicated_df) > 0:

        counts = (
            duplicated_df
            .groupby(transfer_key)
            .size()
            .reset_index(name="rows")
            .sort_values(
                "rows",
                ascending=False,
            )
        )

        print("\n중복 이적 TOP 20")
        print(counts.head(20))

        print("\n예시")

        print(
            duplicated_df[
                transfer_key
                + [
                    "team_name",
                    "competition_name",
                    "matches",
                    "minutes",
                    "rating",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    else:
        print("\n중복 없음")


if __name__ == "__main__":
    main()