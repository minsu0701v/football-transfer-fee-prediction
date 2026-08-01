from pathlib import Path

import pandas as pd


CANDIDATE_FILE = Path(
    "data/processed/prediction_candidates.csv"
)

EXISTING_MAPPING_FILE = Path(
    "data/processed/player_mapping.csv"
)

NEW_MAPPING_FILE = Path(
    "data/processed/prediction_player_mapping.csv"
)

OUTPUT_FILE = Path(
    "data/processed/prediction_all_player_mapping.csv"
)


def load_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"매핑 파일을 찾을 수 없습니다: {path}"
        )

    mapping = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = [
        "player_id",
        "fotmob_id",
        "fotmob_name",
        "birth_date",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in mapping.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path}에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    mapping = mapping[
        required_columns
    ].copy()

    mapping["player_id"] = pd.to_numeric(
        mapping["player_id"],
        errors="coerce",
    )

    mapping["fotmob_id"] = pd.to_numeric(
        mapping["fotmob_id"],
        errors="coerce",
    )

    mapping = mapping.dropna(
        subset=[
            "player_id",
            "fotmob_id",
        ]
    )

    mapping["player_id"] = (
        mapping["player_id"].astype(int)
    )

    mapping["fotmob_id"] = (
        mapping["fotmob_id"].astype(int)
    )

    return mapping


def main():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"후보 파일을 찾을 수 없습니다: "
            f"{CANDIDATE_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        usecols=["player_id"],
        low_memory=False,
    )

    candidate_ids = set(
        candidates["player_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    existing_mapping = load_mapping(
        EXISTING_MAPPING_FILE
    )

    new_mapping = load_mapping(
        NEW_MAPPING_FILE
    )

    # 기존 매핑 중 Prediction 후보에 포함되는 선수만 사용
    existing_mapping = existing_mapping[
        existing_mapping["player_id"].isin(
            candidate_ids
        )
    ].copy()

    existing_count = len(existing_mapping)
    new_count = len(new_mapping)

    combined = pd.concat(
        [
            existing_mapping,
            new_mapping,
        ],
        ignore_index=True,
    )

    # 신규 매핑 결과를 우선 사용
    combined = combined.drop_duplicates(
        subset=["player_id"],
        keep="last",
    )

    combined = combined.sort_values(
        "player_id"
    ).reset_index(drop=True)

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    mapped_ids = set(
        combined["player_id"].tolist()
    )

    unmapped_count = len(
        candidate_ids - mapped_ids
    )

    duplicate_count = int(
        combined["player_id"]
        .duplicated()
        .sum()
    )

    mapping_rate = (
        len(combined)
        / len(candidate_ids)
        * 100
        if candidate_ids
        else 0
    )

    print("=" * 70)
    print("Prediction 전체 FotMob 매핑 통합 완료")
    print("=" * 70)
    print(f"전체 후보             : {len(candidate_ids):,}")
    print(f"기존 매핑 재사용      : {existing_count:,}")
    print(f"신규 매핑 성공        : {new_count:,}")
    print(f"통합 FotMob ID 보유   : {len(combined):,}")
    print(f"FotMob ID 미보유      : {unmapped_count:,}")
    print(f"전체 매핑률           : {mapping_rate:.2f}%")
    print(f"player_id 중복        : {duplicate_count:,}")
    print(f"저장 파일             : {OUTPUT_FILE}")

    if duplicate_count == 0:
        print("PREDICTION_MAPPING_MERGED=True")
    else:
        print("PREDICTION_MAPPING_MERGED=False")


if __name__ == "__main__":
    main()