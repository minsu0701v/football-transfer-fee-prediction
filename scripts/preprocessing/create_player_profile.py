import pandas as pd

PROFILE_FILE = "data/raw/player_profiles.csv"
MAPPING_FILE = "data/processed/player_mapping.csv"
OUTPUT_FILE = "data/processed/player_profile_processed.csv"

COLUMNS = [
    "player_id",
    "player_name",
    "date_of_birth",
    "height",
    "citizenship",
    "main_position",
    "foot",
    "fotmob_id",
]


def main():
    print("파일 불러오는 중...")

    profiles = pd.read_csv(PROFILE_FILE, low_memory=False)
    mapping = pd.read_csv(MAPPING_FILE)

    profiles["player_name"] = (
        profiles["player_name"]
        .str.replace(r"\s*\(\d+\)$", "", regex=True)
    )

    print(f"원본 선수 수: {len(profiles):,}")
    print(f"매핑 선수 수: {len(mapping):,}")

    result = (
        profiles.merge(
            mapping[["player_id", "fotmob_id"]],
            on="player_id",
            how="inner",
        )[COLUMNS]
        .sort_values("player_id")
        .reset_index(drop=True)
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n완료")
    print("-" * 40)
    print(f"생성된 선수 수: {len(result):,}")
    print(f"컬럼 수: {len(result.columns)}")
    print(f"저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()