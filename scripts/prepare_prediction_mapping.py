from pathlib import Path

import pandas as pd


CANDIDATE_FILE = Path(
    "data/processed/prediction_candidates.csv"
)

EXISTING_MAPPING_FILE = Path(
    "data/processed/player_mapping.csv"
)

OUTPUT_MAPPED_FILE = Path(
    "data/processed/prediction_candidates_mapped.csv"
)

OUTPUT_TARGET_FILE = Path(
    "data/processed/prediction_mapping_targets.csv"
)


def main():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"후보 파일이 없습니다: {CANDIDATE_FILE}"
        )

    if not EXISTING_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"기존 매핑 파일이 없습니다: "
            f"{EXISTING_MAPPING_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        low_memory=False,
    )

    mappings = pd.read_csv(
        EXISTING_MAPPING_FILE,
        low_memory=False,
    )

    required_candidate_columns = {
        "player_id",
        "player_name",
        "date_of_birth",
    }

    missing_candidate_columns = (
        required_candidate_columns
        - set(candidates.columns)
    )

    if missing_candidate_columns:
        raise ValueError(
            "후보 파일에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_candidate_columns)}"
        )

    required_mapping_columns = {
        "player_id",
        "fotmob_id",
        "fotmob_name",
        "birth_date",
    }

    missing_mapping_columns = (
        required_mapping_columns
        - set(mappings.columns)
    )

    if missing_mapping_columns:
        raise ValueError(
            "매핑 파일에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_mapping_columns)}"
        )

    mappings = mappings[
        [
            "player_id",
            "fotmob_id",
            "fotmob_name",
            "birth_date",
        ]
    ].copy()

    mappings["fotmob_id"] = pd.to_numeric(
        mappings["fotmob_id"],
        errors="coerce",
    )

    mappings = mappings.dropna(
        subset=[
            "player_id",
            "fotmob_id",
        ]
    )

    mappings = mappings.drop_duplicates(
        subset=["player_id"],
        keep="first",
    )

    candidates = candidates.drop(
        columns=[
            "fotmob_id",
        ],
        errors="ignore",
    )

    merged = candidates.merge(
        mappings,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    mapped_count = int(
        merged["fotmob_id"].notna().sum()
    )

    unmapped_count = int(
        merged["fotmob_id"].isna().sum()
    )

    duplicate_count = int(
        merged["player_id"].duplicated().sum()
    )

    mapping_targets = merged[
        merged["fotmob_id"].isna()
    ].copy()

    target_columns = [
        "player_id",
        "player_name",
        "date_of_birth",
        "current_club_id",
        "current_club_name",
        "current_league_id",
        "current_league_name",
        "performance_team_id",
        "performance_team_name",
        "season_name",
    ]

    target_columns = [
        column
        for column in target_columns
        if column in mapping_targets.columns
    ]

    mapping_targets = mapping_targets[
        target_columns
    ].copy()

    merged.to_csv(
        OUTPUT_MAPPED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    mapping_targets.to_csv(
        OUTPUT_TARGET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 70)
    print("Prediction Candidate 기존 매핑 적용 완료")
    print("=" * 70)
    print(f"전체 후보         : {len(merged):,}")
    print(f"기존 매핑 재사용 : {mapped_count:,}")
    print(f"신규 매핑 필요   : {unmapped_count:,}")
    print(f"player_id 중복   : {duplicate_count:,}")
    print()
    print(f"전체 저장 파일   : {OUTPUT_MAPPED_FILE}")
    print(f"신규 매핑 대상   : {OUTPUT_TARGET_FILE}")

    if (
        len(merged) == len(candidates)
        and duplicate_count == 0
        and mapped_count + unmapped_count
        == len(merged)
    ):
        print("PREDICTION_MAPPING_PREPARED=True")
    else:
        print("PREDICTION_MAPPING_PREPARED=False")


if __name__ == "__main__":
    main()