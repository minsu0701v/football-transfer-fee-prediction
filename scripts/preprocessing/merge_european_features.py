import pandas as pd


TRAINING_FILE = (
    "data/processed/"
    "training_dataset.csv"
)

EUROPEAN_FILE = (
    "data/processed/"
    "european_features.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "training_dataset_european.csv"
)

def main():
    training = pd.read_csv(
        TRAINING_FILE,
        low_memory=False,
    )

    european = pd.read_csv(
        EUROPEAN_FILE,
        low_memory=False,
    )

    print(
        "training rows:",
        len(training),
    )

    print(
        "european rows:",
        len(european),
    )

    # ----------------------------------------
    # merge
    # ----------------------------------------

    merged = training.merge(
        european,
        left_on=[
            "player_id",
            "previous_season",
        ],
        right_on=[
            "player_id",
            "season_name",
        ],
        how="left",
        validate="many_to_one",
    )

    # european_features의 season_name은
    # previous_season과 같은 정보이므로 제거
    merged = merged.drop(
        columns=[
            "season_name",
        ]
    )

    # ----------------------------------------
    # 유럽대항전 feature
    # ----------------------------------------

    european_columns = [
        "ucl_appearances",
        "ucl_starts",
        "ucl_goals",
        "ucl_assists",
        "has_ucl",

        "uel_appearances",
        "uel_starts",
        "uel_goals",
        "uel_assists",
        "has_uel",

        "uecl_appearances",
        "uecl_starts",
        "uecl_goals",
        "uecl_assists",
        "has_uecl",
    ]

    # 유럽대항전 기록 없는 선수는 0
    merged[
        european_columns
    ] = (
        merged[
            european_columns
        ]
        .fillna(0)
    )

    # ----------------------------------------
    # 검증
    # ----------------------------------------

    print(
        "\nmerge 후 rows:",
        len(merged),
    )

    if len(merged) != len(training):
        raise ValueError(
            "merge 후 training row 수가 변경됨"
        )

    matched = (
        merged[
            [
                "has_ucl",
                "has_uel",
                "has_uecl",
            ]
        ]
        .sum(axis=1)
        > 0
    )

    print(
        "유럽대항전 실제 출전 샘플:",
        matched.sum(),
    )

    print(
        "유럽대항전 기록 없는 샘플:",
        (~matched).sum(),
    )

    print(
        "\nUCL 참가:",
        int(
            merged[
                "has_ucl"
            ].sum()
        ),
    )

    print(
        "UEL 참가:",
        int(
            merged[
                "has_uel"
            ].sum()
        ),
    )

    print(
        "UECL 참가:",
        int(
            merged[
                "has_uecl"
            ].sum()
        ),
    )

    print(
        "\n신규 feature 결측:"
    )

    print(
        merged[
            european_columns
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ----------------------------------------
    # 저장
    # ----------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n저장 완료:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()