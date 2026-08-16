import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "target_european_performances.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "european_features.csv"
)


COMPETITIONS = [
    "UCL",
    "UEL",
    "UECL",
]


def main():
    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print("원본 rows:", len(df))

    # ----------------------------------------
    # 1. 사용할 수치형 컬럼 정리
    # ----------------------------------------

    numeric_columns = [
        "nb_on_pitch",
        "subed_in",
        "goals",
        "assists",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ----------------------------------------
    # 2. 실제 모델용 통계 정의
    # ----------------------------------------

    df["appearances"] = (
        df["nb_on_pitch"]
        .fillna(0)
    )

    df["starts"] = (
        df["nb_on_pitch"].fillna(0)
        - df["subed_in"].fillna(0)
    )

    # 혹시 데이터 오류로 음수가 생기면 0
    df["starts"] = (
        df["starts"]
        .clip(lower=0)
    )

    df["goals"] = (
        df["goals"]
        .fillna(0)
    )

    df["assists"] = (
        df["assists"]
        .fillna(0)
    )

    # ----------------------------------------
    # 3. 선수 + 시즌 + 대회 기준 집계
    #
    # 같은 시즌 동일 대회에서
    # 두 팀을 뛴 경우도 여기서 합산
    # ----------------------------------------

    grouped = (
        df.groupby(
            [
                "player_id",
                "season_name",
                "european_competition",
            ],
            as_index=False,
        )
        .agg(
            appearances=(
                "appearances",
                "sum",
            ),
            starts=(
                "starts",
                "sum",
            ),
            goals=(
                "goals",
                "sum",
            ),
            assists=(
                "assists",
                "sum",
            ),
        )
    )

    print(
        "대회 단위 집계 rows:",
        len(grouped),
    )

    # ----------------------------------------
    # 4. Wide 형태로 변환
    #
    # UCL / UEL / UECL 각각 별도 feature
    # ----------------------------------------

    value_columns = [
        "appearances",
        "starts",
        "goals",
        "assists",
    ]

    wide = grouped.pivot_table(
        index=[
            "player_id",
            "season_name",
        ],
        columns="european_competition",
        values=value_columns,
        aggfunc="sum",
        fill_value=0,
    )

    # MultiIndex 컬럼
    # ('goals', 'UCL')
    # →
    # ucl_goals
    wide.columns = [
        f"{competition.lower()}_{stat}"
        for stat, competition
        in wide.columns
    ]

    wide = (
        wide
        .reset_index()
    )

    # ----------------------------------------
    # 5. 모든 대회 컬럼 보장
    # ----------------------------------------

    feature_columns = []

    for competition in COMPETITIONS:
        prefix = competition.lower()

        for stat in value_columns:
            column = (
                f"{prefix}_{stat}"
            )

            feature_columns.append(
                column
            )

            if column not in wide.columns:
                wide[column] = 0

    # ----------------------------------------
    # 6. 참가 여부 feature
    # ----------------------------------------

    for competition in COMPETITIONS:
        prefix = competition.lower()

        wide[
            f"has_{prefix}"
        ] = (
            wide[
                f"{prefix}_appearances"
            ] > 0
        ).astype(int)

    # ----------------------------------------
    # 7. 컬럼 순서 정리
    # ----------------------------------------

    ordered_columns = [
        "player_id",
        "season_name",

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

    wide = wide[
        ordered_columns
    ]

    # ----------------------------------------
    # 8. 검증
    # ----------------------------------------

    print(
        "최종 player-season rows:",
        len(wide),
    )

    print(
        "고유 선수:",
        wide["player_id"].nunique(),
    )

    print("\n대회 참가:")
    print(
        "UCL :",
        wide["has_ucl"].sum(),
    )
    print(
        "UEL :",
        wide["has_uel"].sum(),
    )
    print(
        "UECL:",
        wide["has_uecl"].sum(),
    )

    duplicate_count = (
        wide.duplicated(
            subset=[
                "player_id",
                "season_name",
            ]
        )
        .sum()
    )

    print(
        "\nplayer-season 중복:",
        duplicate_count,
    )

    # ----------------------------------------
    # 9. 저장
    # ----------------------------------------

    wide.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n저장 완료:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()