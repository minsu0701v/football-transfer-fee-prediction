# scripts/check_training_dataset_integrity.py

from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_FILE = Path("data/processed/training_dataset.csv")
TRANSFER_FILE = Path("data/processed/top5_transfers.csv")

OUTPUT_DIR = Path("outputs/data_validation")
DUPLICATE_FILE = OUTPUT_DIR / "duplicate_transfers.csv"
MISSING_TRANSFER_FILE = OUTPUT_DIR / "missing_transfers_after_merge.csv"
SUSPICIOUS_FILE = OUTPUT_DIR / "suspicious_rows.csv"


# 이적 1건을 구분하는 기준
TRANSFER_KEY_CANDIDATES = [
    "player_id",
    "transfer_date",
    "from_team_id",
    "to_team_id",
]

REQUIRED_COLUMNS = [
    "player_id",
    "transfer_date",
    "transfer_fee",
    "value_at_transfer",
    "matches",
    "minutes",
    "rating",
    "age_at_transfer",
    "main_position",
    "from_league_id",
    "to_league_id",
]


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} 파일을 찾을 수 없습니다: {path}"
        )

    df = pd.read_csv(path)

    print(f"{name}: {len(df):,}행 × {len(df.columns):,}열")

    return df


def get_transfer_key(df: pd.DataFrame) -> list[str]:
    transfer_key = [
        column
        for column in TRANSFER_KEY_CANDIDATES
        if column in df.columns
    ]

    if "player_id" not in transfer_key:
        raise ValueError("player_id 컬럼이 없습니다.")

    if "transfer_date" not in transfer_key:
        raise ValueError("transfer_date 컬럼이 없습니다.")

    print(f"이적 식별 기준: {transfer_key}")

    return transfer_key


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "transfer_date" in result.columns:
        result["transfer_date"] = pd.to_datetime(
            result["transfer_date"],
            errors="coerce",
        )

    numeric_columns = [
        "transfer_fee",
        "value_at_transfer",
        "matches",
        "started",
        "minutes",
        "rating",
        "goals",
        "assists",
        "age_at_transfer",
        "height",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def check_required_columns(df: pd.DataFrame) -> list[str]:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    print()
    print("=" * 72)
    print("1. 필수 컬럼 검사")
    print("=" * 72)

    if missing_columns:
        print(f"[경고] 없는 컬럼: {missing_columns}")
    else:
        print("[통과] 필요한 핵심 컬럼이 모두 존재합니다.")

    return missing_columns


def check_exact_duplicates(df: pd.DataFrame) -> int:
    print()
    print("=" * 72)
    print("2. 완전히 동일한 행 검사")
    print("=" * 72)

    duplicate_count = int(df.duplicated().sum())

    print(f"완전 중복 행 수: {duplicate_count:,}")

    if duplicate_count == 0:
        print("[통과] 완전히 동일한 중복 행이 없습니다.")
    else:
        print("[실패] 완전히 동일한 행이 중복되어 있습니다.")

    return duplicate_count


def check_transfer_duplicates(
    df: pd.DataFrame,
    transfer_key: list[str],
) -> pd.DataFrame:
    print()
    print("=" * 72)
    print("3. 동일 이적 건의 복수 행 검사")
    print("=" * 72)

    counts = (
        df.groupby(
            transfer_key,
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
        .sort_values(
            "row_count",
            ascending=False,
        )
    )

    duplicated_keys = counts[
        counts["row_count"] > 1
    ].copy()

    max_rows_per_transfer = int(counts["row_count"].max())

    print(f"고유 이적 건수       : {len(counts):,}")
    print(f"학습 데이터 행 수    : {len(df):,}")
    print(f"이적 1건당 최대 행 수: {max_rows_per_transfer:,}")
    print(f"복수 행 이적 건수    : {len(duplicated_keys):,}")

    if duplicated_keys.empty:
        print("[통과] 모든 이적이 학습 데이터에서 1행으로 구성됩니다.")
        return duplicated_keys

    duplicate_rows = df.merge(
        duplicated_keys[transfer_key],
        on=transfer_key,
        how="inner",
    ).sort_values(
        transfer_key,
    )

    duplicate_rows.to_csv(
        DUPLICATE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("[실패] 동일한 이적이 여러 행으로 존재합니다.")
    print(f"상세 저장: {DUPLICATE_FILE}")

    display_columns = [
        column
        for column in [
            *transfer_key,
            "player_name",
            "season_name",
            "previous_season",
            "competition_id",
            "competition_name",
            "team_id",
            "team_name",
            "transfer_fee",
        ]
        if column in duplicate_rows.columns
    ]

    print()
    print(
        duplicate_rows[
            display_columns
        ].head(20).to_string(index=False)
    )

    return duplicated_keys


def check_merge_coverage(
    training_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    transfer_key: list[str],
) -> pd.DataFrame:
    print()
    print("=" * 72)
    print("4. 원본 이적 데이터 대비 병합 결과 검사")
    print("=" * 72)

    common_key = [
        column
        for column in transfer_key
        if column in transfer_df.columns
        and column in training_df.columns
    ]

    if len(common_key) < 2:
        print(
            "[경고] 공통 식별 컬럼이 부족해 "
            "원본 대비 병합 누락을 검사하지 못했습니다."
        )
        return pd.DataFrame()

    original_unique = transfer_df[
        common_key
    ].drop_duplicates()

    training_unique = training_df[
        common_key
    ].drop_duplicates()

    comparison = original_unique.merge(
        training_unique.assign(in_training_dataset=True),
        on=common_key,
        how="left",
    )

    missing_transfers = comparison[
        comparison["in_training_dataset"].isna()
    ][common_key].copy()

    original_count = len(original_unique)
    training_count = len(training_unique)
    missing_count = len(missing_transfers)

    coverage_rate = (
        training_count / original_count * 100
        if original_count > 0
        else 0
    )

    print(f"원본 고유 이적 건수 : {original_count:,}")
    print(f"학습 반영 이적 건수 : {training_count:,}")
    print(f"학습 제외 이적 건수 : {missing_count:,}")
    print(f"병합 반영률          : {coverage_rate:.2f}%")

    if missing_transfers.empty:
        print("[통과] 원본 이적이 모두 학습 데이터에 반영됐습니다.")
        return missing_transfers

    missing_detail = transfer_df.merge(
        missing_transfers,
        on=common_key,
        how="inner",
    )

    missing_detail.to_csv(
        MISSING_TRANSFER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[확인 필요] 병합 과정에서 제외된 이적이 있습니다. "
        "Inner merge라면 일부 제외는 정상일 수 있습니다."
    )
    print(f"상세 저장: {MISSING_TRANSFER_FILE}")

    display_columns = [
        column
        for column in [
            "player_id",
            "player_name",
            "transfer_date",
            "from_team_name",
            "to_team_name",
            "previous_season",
            "transfer_fee",
        ]
        if column in missing_detail.columns
    ]

    if display_columns:
        print()
        print(
            missing_detail[
                display_columns
            ].head(20).to_string(index=False)
        )

    return missing_transfers


def check_missing_values(df: pd.DataFrame):
    print()
    print("=" * 72)
    print("5. 핵심 컬럼 결측치 검사")
    print("=" * 72)

    check_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column in df.columns
    ]

    missing_summary = pd.DataFrame(
        {
            "missing_count": df[check_columns].isna().sum(),
            "missing_rate_percent": (
                df[check_columns].isna().mean() * 100
            ),
        }
    ).sort_values(
        "missing_count",
        ascending=False,
    )

    print(
        missing_summary.to_string(
            formatters={
                "missing_rate_percent": (
                    lambda value: f"{value:.2f}%"
                ),
            }
        )
    )

    target_missing = 0

    for column in [
        "player_id",
        "transfer_date",
        "transfer_fee",
    ]:
        if column in df.columns:
            target_missing += int(
                df[column].isna().sum()
            )

    if target_missing == 0:
        print("\n[통과] 식별자·이적일·목표값에는 결측치가 없습니다.")
    else:
        print("\n[실패] 식별자 또는 목표값에 결측치가 있습니다.")

    return target_missing


def check_suspicious_values(df: pd.DataFrame) -> pd.DataFrame:
    print()
    print("=" * 72)
    print("6. 비정상 범위 검사")
    print("=" * 72)

    conditions = []

    def add_condition(
        column: str,
        condition: pd.Series,
        reason: str,
    ):
        if column not in df.columns:
            return

        flagged = df.loc[condition].copy()

        if flagged.empty:
            return

        flagged["suspicious_reason"] = reason
        conditions.append(flagged)

    if "transfer_fee" in df.columns:
        add_condition(
            "transfer_fee",
            df["transfer_fee"] <= 0,
            "transfer_fee <= 0",
        )

    if "value_at_transfer" in df.columns:
        add_condition(
            "value_at_transfer",
            df["value_at_transfer"] < 0,
            "value_at_transfer < 0",
        )

    if "matches" in df.columns:
        add_condition(
            "matches",
            (df["matches"] < 0) | (df["matches"] > 50),
            "matches outside 0~50",
        )

    if "started" in df.columns:
        add_condition(
            "started",
            (df["started"] < 0)
            | (
                df["matches"].notna()
                & (df["started"] > df["matches"])
            ),
            "started < 0 or started > matches",
        )

    if "minutes" in df.columns:
        add_condition(
            "minutes",
            (df["minutes"] < 0) | (df["minutes"] > 5_000),
            "minutes outside 0~5000",
        )

    if "rating" in df.columns:
        add_condition(
            "rating",
            (df["rating"] < 0) | (df["rating"] > 10),
            "rating outside 0~10",
        )

    if "age_at_transfer" in df.columns:
        add_condition(
            "age_at_transfer",
            (df["age_at_transfer"] < 15)
            | (df["age_at_transfer"] > 45),
            "age_at_transfer outside 15~45",
        )

    if "height" in df.columns:
        add_condition(
            "height",
            (df["height"] < 140) | (df["height"] > 220),
            "height outside 140~220",
        )

    if not conditions:
        print("[통과] 지정한 범위를 벗어난 값이 없습니다.")
        return pd.DataFrame()

    suspicious_df = pd.concat(
        conditions,
        ignore_index=True,
    ).drop_duplicates()

    suspicious_df.to_csv(
        SUSPICIOUS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[확인 필요] 비정상 후보 행: {len(suspicious_df):,}건")
    print(f"상세 저장: {SUSPICIOUS_FILE}")

    display_columns = [
        column
        for column in [
            "player_id",
            "player_name",
            "transfer_date",
            "matches",
            "started",
            "minutes",
            "rating",
            "age_at_transfer",
            "height",
            "transfer_fee",
            "value_at_transfer",
            "suspicious_reason",
        ]
        if column in suspicious_df.columns
    ]

    print()
    print(
        suspicious_df[
            display_columns
        ].head(30).to_string(index=False)
    )

    return suspicious_df


def check_target_consistency(
    df: pd.DataFrame,
    transfer_key: list[str],
):
    print()
    print("=" * 72)
    print("7. 동일 이적의 목표값 일관성 검사")
    print("=" * 72)

    if "transfer_fee" not in df.columns:
        print("[경고] transfer_fee가 없어 검사하지 못했습니다.")
        return 0

    target_counts = (
        df.groupby(
            transfer_key,
            dropna=False,
        )["transfer_fee"]
        .nunique(dropna=False)
    )

    inconsistent_count = int(
        (target_counts > 1).sum()
    )

    print(f"목표값이 충돌하는 이적 건수: {inconsistent_count:,}")

    if inconsistent_count == 0:
        print("[통과] 동일 이적 내 transfer_fee가 일관됩니다.")
    else:
        print("[실패] 동일 이적에 서로 다른 transfer_fee가 존재합니다.")

    return inconsistent_count


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Training Dataset 무결성 검사")
    print("=" * 72)

    training_df = normalize_columns(
        load_csv(
            TRAINING_FILE,
            "training_dataset",
        )
    )

    transfer_df = normalize_columns(
        load_csv(
            TRANSFER_FILE,
            "top5_transfers",
        )
    )

    transfer_key = get_transfer_key(training_df)

    missing_columns = check_required_columns(training_df)
    exact_duplicate_count = check_exact_duplicates(training_df)

    duplicated_keys = check_transfer_duplicates(
        training_df,
        transfer_key,
    )

    missing_transfers = check_merge_coverage(
        training_df,
        transfer_df,
        transfer_key,
    )

    target_missing_count = check_missing_values(training_df)

    suspicious_df = check_suspicious_values(training_df)

    inconsistent_target_count = check_target_consistency(
        training_df,
        transfer_key,
    )

    critical_failure_count = sum(
        [
            exact_duplicate_count > 0,
            not duplicated_keys.empty,
            target_missing_count > 0,
            inconsistent_target_count > 0,
        ]
    )

    print()
    print("=" * 72)
    print("최종 판정")
    print("=" * 72)

    if critical_failure_count == 0:
        print("[통과] 모델 학습을 막을 치명적인 병합 오류는 발견되지 않았습니다.")
    else:
        print(
            f"[실패] 치명적 검사 항목 "
            f"{critical_failure_count}개를 수정해야 합니다."
        )

    print()
    print("참고 항목")
    print(f"- 없는 핵심 컬럼 수    : {len(missing_columns):,}")
    print(f"- 원본 대비 제외 이적 : {len(missing_transfers):,}")
    print(f"- 비정상 후보 행       : {len(suspicious_df):,}")

    if critical_failure_count == 0:
        print()
        print("MODEL_FINALIZATION_READY=True")
    else:
        print()
        print("MODEL_FINALIZATION_READY=False")


if __name__ == "__main__":
    main()