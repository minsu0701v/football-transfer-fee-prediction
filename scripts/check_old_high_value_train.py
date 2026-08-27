from pathlib import Path

import pandas as pd


# ============================================================
# 설정
# ============================================================

DATA_FILE = Path(
    "data/processed/training_dataset_european.csv"
)

TRAIN_START_YEAR = 2020
TRAIN_END_YEAR = 2024

AGE_THRESHOLD = 27

TARGET = "transfer_fee"


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print("TRAIN 2020~2024 - AGE 27+ HIGH VALUE CHECK")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 데이터 로드
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"데이터 없음: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df["age_at_transfer"] = pd.to_numeric(
        df["age_at_transfer"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    # --------------------------------------------------------
    # 2. Train 2020~2024
    # --------------------------------------------------------

    train_df = df[
        df["transfer_date"]
        .dt.year
        .between(
            TRAIN_START_YEAR,
            TRAIN_END_YEAR,
        )
    ].copy()

    train_df = train_df[
        train_df["age_at_transfer"].notna()
        & train_df[TARGET].notna()
    ].copy()

    print()
    print(
        f"Train rows: {len(train_df)}"
    )

    # --------------------------------------------------------
    # 3. Age >= 27
    # --------------------------------------------------------

    age27 = train_df[
        train_df["age_at_transfer"]
        >= AGE_THRESHOLD
    ].copy()

    age27_30 = age27[
        age27[TARGET]
        >= 30_000_000
    ].copy()

    age27_50 = age27[
        age27[TARGET]
        >= 50_000_000
    ].copy()

    age27_70 = age27[
        age27[TARGET]
        >= 70_000_000
    ].copy()

    # --------------------------------------------------------
    # 4. Count
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("COUNT")
    print("=" * 72)

    print(
        f"Age >= 27 전체       : {len(age27)}명"
    )

    print(
        f"Age >= 27 & 30M+    : {len(age27_30)}명"
    )

    print(
        f"Age >= 27 & 50M+    : {len(age27_50)}명"
    )

    print(
        f"Age >= 27 & 70M+    : {len(age27_70)}명"
    )

    # --------------------------------------------------------
    # 5. 고액 선수 상세
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("AGE >= 27 & 30M+ PLAYERS")
    print("=" * 72)

    if len(age27_30) == 0:

        print("해당 선수 없음")

    else:

        result = (
            age27_30
            .sort_values(
                TARGET,
                ascending=False,
            )
            .copy()
        )

        result["fee_m"] = (
            result[TARGET]
            / 1_000_000
        )

        result["age"] = (
            result["age_at_transfer"]
            .round(2)
        )

        result["year"] = (
            result["transfer_date"]
            .dt.year
        )

        # 구간 표시
        result["30M+"] = (
            result[TARGET]
            >= 30_000_000
        )

        result["50M+"] = (
            result[TARGET]
            >= 50_000_000
        )

        result["70M+"] = (
            result[TARGET]
            >= 70_000_000
        )

        columns = []

        for column in [
            "player_name",
            "year",
            "age",
            "fee_m",
            "from_team_name",
            "to_team_name",
            "from_club_name",
            "to_club_name",
            "30M+",
            "50M+",
            "70M+",
        ]:

            if column in result.columns:
                columns.append(column)

        print(
            result[
                columns
            ].to_string(
                index=False,
            )
        )

    # --------------------------------------------------------
    # 6. 50M+ 별도
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("AGE >= 27 & 50M+")
    print("=" * 72)

    if len(age27_50) == 0:

        print("해당 선수 없음")

    else:

        temp = (
            age27_50
            .sort_values(
                TARGET,
                ascending=False,
            )
            .copy()
        )

        temp["fee_m"] = (
            temp[TARGET]
            / 1_000_000
        )

        print(
            temp[
                [
                    "player_name",
                    "age_at_transfer",
                    "fee_m",
                ]
            ]
            .round(2)
            .to_string(
                index=False,
            )
        )

    # --------------------------------------------------------
    # 7. 70M+ 별도
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("AGE >= 27 & 70M+")
    print("=" * 72)

    if len(age27_70) == 0:

        print("해당 선수 없음")

    else:

        temp = (
            age27_70
            .sort_values(
                TARGET,
                ascending=False,
            )
            .copy()
        )

        temp["fee_m"] = (
            temp[TARGET]
            / 1_000_000
        )

        print(
            temp[
                [
                    "player_name",
                    "age_at_transfer",
                    "fee_m",
                ]
            ]
            .round(2)
            .to_string(
                index=False,
            )
        )

    print()
    print("=" * 72)
    print("CHECK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()