import pandas as pd


PERFORMANCE_FILE = "data/raw/player_performances.csv"
TARGET_FILE = "data/processed/target_player_performances.csv"

OUTPUT_FILE = (
    "data/processed/"
    "target_european_performances.csv"
)


EUROPEAN_COMPETITIONS = {
    "CL": "UCL",
    "EL": "UEL",
    "UCOL": "UECL",
}


def classify_european_competition(competition_id):
    if pd.isna(competition_id):
        return None

    return EUROPEAN_COMPETITIONS.get(
        str(competition_id).strip()
    )

def main():
    # ----------------------------------------
    # 1. 원본 경기 기록
    # ----------------------------------------

    performances = pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )

    # ----------------------------------------
    # 2. 기존 target 선수/시즌
    # ----------------------------------------

    targets = pd.read_csv(
        TARGET_FILE,
        low_memory=False,
    )

    print(
        "전체 performance rows:",
        len(performances),
    )

    print(
        "기존 target rows:",
        len(targets),
    )

    # 우리가 필요한 건
    # "이 선수의 이 시즌"이라는 정보
    target_keys = (
        targets[
            [
                "player_id",
                "season_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    print(
        "고유 player-season:",
        len(target_keys),
    )

    # ----------------------------------------
    # 3. 유럽대항전 분류
    # ----------------------------------------

    performances[
        "european_competition"
    ] = (
        performances[
            "competition_id"
        ]
        .apply(
            classify_european_competition
        )
    )

    european = performances[
        performances[
            "european_competition"
        ].notna()
    ].copy()

    print(
        "전체 유럽대항전 rows:",
        len(european),
    )

    print(
        "\n발견된 대회:"
    )

    print(
        european[
            [
                "competition_id",
                "competition_name",
                "european_competition",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "european_competition",
                "competition_name",
            ]
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------
    # 4. target 선수 + target 시즌만 추출
    # ----------------------------------------

    target_european = european.merge(
        target_keys,
        on=[
            "player_id",
            "season_name",
        ],
        how="inner",
    )

    # 혹시 완전히 동일한 행이 있다면 제거
    target_european = (
        target_european
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ----------------------------------------
    # 5. 결과 확인
    # ----------------------------------------

    print(
        "\n================================"
    )

    print(
        "Target European Performance"
    )

    print(
        "================================"
    )

    print(
        "rows:",
        len(target_european),
    )

    print(
        "players:",
        target_european[
            "player_id"
        ].nunique(),
    )

    print(
        "\n대회별:"
    )

    print(
        target_european[
            "european_competition"
        ]
        .value_counts()
        .to_string()
    )

    # ----------------------------------------
    # 6. 저장
    # ----------------------------------------

    target_european.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n저장 완료:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()