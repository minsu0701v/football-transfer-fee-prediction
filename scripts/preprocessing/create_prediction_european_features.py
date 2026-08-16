from pathlib import Path

import pandas as pd


# ============================================================
# 설정
# ============================================================

PREDICTION_FILE = Path(
    "data/processed/prediction_dataset.csv"
)

PERFORMANCE_FILE = Path(
    "data/raw/player_performances.csv"
)

EUROPEAN_FEATURE_FILE = Path(
    "data/processed/prediction_european_features_24_25.csv"
)

OUTPUT_FILE = Path(
    "data/processed/prediction_dataset_european.csv"
)


TARGET_SEASON = "24/25"


COMPETITION_MAP = {
    "CL": "UCL",
    "EL": "UEL",
    "UCOL": "UECL",
}


EUROPEAN_FEATURES = [
    "ucl_appearances",
    "ucl_starts",
    "ucl_goals",
    "ucl_assists",

    "uel_appearances",
    "uel_starts",
    "uel_goals",
    "uel_assists",

    "uecl_appearances",
    "uecl_starts",
    "uecl_goals",
    "uecl_assists",
]


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "Prediction Dataset European Features - 24/25"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "예측 데이터가 없습니다: "
            f"{PREDICTION_FILE}"
        )

    if not PERFORMANCE_FILE.exists():
        raise FileNotFoundError(
            "원본 performance 데이터가 없습니다: "
            f"{PERFORMANCE_FILE}"
        )

    # --------------------------------------------------------
    # 2. 데이터 로드
    # --------------------------------------------------------

    prediction = pd.read_csv(
        PREDICTION_FILE,
        low_memory=False,
    )

    performances = pd.read_csv(
        PERFORMANCE_FILE,
        low_memory=False,
    )

    print(
        "prediction rows:",
        len(prediction),
    )

    print(
        "prediction players:",
        prediction["player_id"].nunique(),
    )

    print(
        "raw performances rows:",
        len(performances),
    )

    # --------------------------------------------------------
    # 3. ID 자료형 통일
    # --------------------------------------------------------

    prediction["player_id"] = pd.to_numeric(
        prediction["player_id"],
        errors="coerce",
    )

    performances["player_id"] = pd.to_numeric(
        performances["player_id"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # 4. 예측 대상 선수만
    # --------------------------------------------------------

    prediction_player_ids = set(
        prediction[
            "player_id"
        ]
        .dropna()
        .astype(int)
    )

    europe = performances[
        performances["player_id"].isin(
            prediction_player_ids
        )
    ].copy()

    print()
    print(
        "예측 대상 선수 raw rows:",
        len(europe),
    )

    # --------------------------------------------------------
    # 5. 24/25 시즌만
    # --------------------------------------------------------

    europe = europe[
        europe["season_name"]
        == TARGET_SEASON
    ].copy()

    print(
        "24/25 rows:",
        len(europe),
    )

    # --------------------------------------------------------
    # 6. UCL / UEL / UECL 본선만
    # --------------------------------------------------------

    europe = europe[
        europe["competition_id"].isin(
            COMPETITION_MAP.keys()
        )
    ].copy()

    europe["european_competition"] = (
        europe["competition_id"]
        .map(
            COMPETITION_MAP
        )
    )

    print(
        "24/25 UCL/UEL/UECL rows:",
        len(europe),
    )

    print()
    print(
        "대회별 raw rows:"
    )

    print(
        europe[
            "european_competition"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 7. 숫자형 변환
    # --------------------------------------------------------

    numeric_columns = [
        "nb_on_pitch",
        "subed_in",
        "goals",
        "assists",
    ]

    for column in numeric_columns:
        europe[column] = pd.to_numeric(
            europe[column],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # 8. 학습 데이터와 동일한 feature 정의
    #
    # appearances = 실제 경기 출전
    # starts      = 실제 출전 - 교체 투입
    # --------------------------------------------------------

    europe["appearances"] = (
        europe["nb_on_pitch"]
    )

    europe["starts"] = (
        europe["nb_on_pitch"]
        - europe["subed_in"]
    ).clip(
        lower=0
    )

    # --------------------------------------------------------
    # 9. 선수 + 대회 기준 집계
    #
    # 같은 시즌에 팀이 둘 이상이어도 합산
    # --------------------------------------------------------

    grouped = (
        europe
        .groupby(
            [
                "player_id",
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

    print()
    print(
        "선수-대회 집계 rows:",
        len(grouped),
    )

    # --------------------------------------------------------
    # 10. Wide 변환
    # --------------------------------------------------------

    value_columns = [
        "appearances",
        "starts",
        "goals",
        "assists",
    ]

    wide = grouped.pivot_table(
        index="player_id",
        columns="european_competition",
        values=value_columns,
        aggfunc="sum",
        fill_value=0,
    )

    wide.columns = [
        f"{competition.lower()}_{stat}"
        for stat, competition
        in wide.columns
    ]

    wide = wide.reset_index()

    # --------------------------------------------------------
    # 11. 12개 feature 컬럼 모두 보장
    # --------------------------------------------------------

    for feature in EUROPEAN_FEATURES:

        if feature not in wide.columns:
            wide[feature] = 0

    wide = wide[
        [
            "player_id",
            *EUROPEAN_FEATURES,
        ]
    ]

    # --------------------------------------------------------
    # 12. feature 파일 저장
    # --------------------------------------------------------

    EUROPEAN_FEATURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wide.to_csv(
        EUROPEAN_FEATURE_FILE,
        index=False,
    )

    print()
    print(
        "Europe feature 선수:",
        len(wide),
    )

    print(
        "저장:",
        EUROPEAN_FEATURE_FILE,
    )

    # --------------------------------------------------------
    # 13. prediction dataset에 LEFT JOIN
    # --------------------------------------------------------

    before_rows = len(
        prediction
    )

    merged = prediction.merge(
        wide,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    # 유럽대항전 기록이 없는 선수
    # → 0
    merged[
        EUROPEAN_FEATURES
    ] = (
        merged[
            EUROPEAN_FEATURES
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 14. 검증
    # --------------------------------------------------------

    after_rows = len(
        merged
    )

    print()
    print("=" * 72)
    print("Merge 검증")
    print("=" * 72)

    print(
        "merge 전 rows:",
        before_rows,
    )

    print(
        "merge 후 rows:",
        after_rows,
    )

    if before_rows != after_rows:
        raise ValueError(
            "Prediction dataset 행 수가 변경되었습니다."
        )

    # 실제 유럽대항전 출전
    total_appearances = (
        merged["ucl_appearances"]
        + merged["uel_appearances"]
        + merged["uecl_appearances"]
    )

    played_europe = (
        total_appearances
        > 0
    )

    print(
        "유럽대항전 실제 출전:",
        int(
            played_europe.sum()
        ),
    )

    print(
        "유럽대항전 출전 없음:",
        int(
            (~played_europe).sum()
        ),
    )

    print()
    print(
        "UCL 출전:",
        int(
            (
                merged[
                    "ucl_appearances"
                ] > 0
            ).sum()
        ),
    )

    print(
        "UEL 출전:",
        int(
            (
                merged[
                    "uel_appearances"
                ] > 0
            ).sum()
        ),
    )

    print(
        "UECL 출전:",
        int(
            (
                merged[
                    "uecl_appearances"
                ] > 0
            ).sum()
        ),
    )

    print()
    print(
        "신규 feature 결측:"
    )

    print(
        merged[
            EUROPEAN_FEATURES
        ]
        .isna()
        .sum()
        .to_string()
    )

    # --------------------------------------------------------
    # 15. 예시 출력
    # --------------------------------------------------------

    display_columns = [
        column
        for column in [
            "player_id",
            "player_name",

            "ucl_appearances",
            "ucl_starts",
            "ucl_goals",
            "ucl_assists",

            "uel_appearances",
            "uel_starts",
            "uel_goals",
            "uel_assists",

            "uecl_appearances",
            "uecl_starts",
            "uecl_goals",
            "uecl_assists",
        ]
        if column in merged.columns
    ]

    example = (
        merged[
            played_europe
        ]
        .sort_values(
            [
                "ucl_appearances",
                "uel_appearances",
                "uecl_appearances",
            ],
            ascending=False,
        )
        .head(20)
    )

    print()
    print("=" * 72)
    print("24/25 유럽대항전 예시")
    print("=" * 72)

    print(
        example[
            display_columns
        ]
        .to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # 16. 최종 저장
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print(
        "✓ Prediction Europe merge 완료"
    )
    print("=" * 72)

    print(
        "저장 완료:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()