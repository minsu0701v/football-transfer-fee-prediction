from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "transfer_fee_model_v1.joblib"
)

PREDICTION_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_dataset.csv"
)


# ============================================================
# 예측 설정
# ============================================================

# 2026년 여름 이적시장 기준일
PREDICTION_DATE = pd.Timestamp("2026-07-01")

TOP5_LEAGUE_IDS = {
    "GB1",  # Premier League
    "ES1",  # LaLiga
    "L1",   # Bundesliga
    "IT1",  # Serie A
    "FR1",  # Ligue 1
}

NUMERIC_FEATURES = [
    "value_at_transfer",
    "age_at_transfer",
    "height",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "is_same_league",
    "is_top5_destination",
]

CATEGORICAL_FEATURES = [
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 출력 함수
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def format_euro(value: float) -> str:
    if value >= 1_000_000:
        return f"€{value / 1_000_000:,.2f}M"

    if value >= 1_000:
        return f"€{value / 1_000:,.0f}K"

    return f"€{value:,.0f}"


# ============================================================
# 데이터 로드
# ============================================================

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "저장된 모델을 찾을 수 없습니다.\n"
            f"확인 경로: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


def load_prediction_data() -> pd.DataFrame:
    if not PREDICTION_DATA_FILE.exists():
        raise FileNotFoundError(
            "예측 데이터셋을 찾을 수 없습니다.\n"
            f"확인 경로: {PREDICTION_DATA_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_DATA_FILE,
        low_memory=False,
    )

    required_columns = [
        "player_id",
        "player_name",
        "date_of_birth",
        "height",
        "main_position",
        "foot",
        "current_club_name",
        "current_league_id",
        "current_league_name",
        "matches",
        "started",
        "goals",
        "assists",
        "minutes",
        "rating",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "prediction_dataset.csv에 필요한 컬럼이 없습니다.\n"
            f"누락 컬럼: {missing_columns}"
        )

    return df


# ============================================================
# 선수 검색
# ============================================================

def search_players(
    df: pd.DataFrame,
    keyword: str,
) -> pd.DataFrame:
    keyword = keyword.strip()

    if not keyword:
        return pd.DataFrame()

    result = df[
        df["player_name"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    return result.head(20)


def select_player(df: pd.DataFrame) -> pd.Series:
    while True:
        keyword = input(
            "\n검색할 선수 이름을 입력하세요: "
        ).strip()

        results = search_players(df, keyword)

        if results.empty:
            print("검색된 선수가 없습니다.")
            continue

        print()
        print("검색 결과")

        display_columns = [
            "player_name",
            "current_club_name",
            "current_league_name",
            "season_name",
            "matches",
            "goals",
            "assists",
            "rating",
        ]

        for number, (_, row) in enumerate(
            results.iterrows(),
            start=1,
        ):
            print(
                f"{number:>2}. "
                f"{row['player_name']} | "
                f"{row['current_club_name']} | "
                f"{row['current_league_name']} | "
                f"{row['season_name']} | "
                f"{row['matches']}경기 "
                f"{row['goals']}골 "
                f"{row['assists']}도움 | "
                f"평점 {row['rating']}"
            )

        selection = input(
            "\n선수 번호를 입력하세요: "
        ).strip()

        try:
            selection_number = int(selection)

        except ValueError:
            print("숫자로 입력하세요.")
            continue

        if not 1 <= selection_number <= len(results):
            print("목록 안의 번호를 입력하세요.")
            continue

        return results.iloc[
            selection_number - 1
        ].copy()


# ============================================================
# 목적지 리그 입력
# ============================================================

def select_destination_league() -> tuple[str, str]:
    leagues = [
        ("GB1", "Premier League"),
        ("ES1", "LaLiga"),
        ("L1", "Bundesliga"),
        ("IT1", "Serie A"),
        ("FR1", "Ligue 1"),
    ]

    print()
    print("예상 이적 리그를 선택하세요.")

    for number, (_, league_name) in enumerate(
        leagues,
        start=1,
    ):
        print(f"{number}. {league_name}")

    while True:
        selection = input(
            "\n리그 번호를 입력하세요: "
        ).strip()

        try:
            selection_number = int(selection)

        except ValueError:
            print("숫자로 입력하세요.")
            continue

        if not 1 <= selection_number <= len(leagues):
            print("1~5 사이 번호를 입력하세요.")
            continue

        return leagues[selection_number - 1]

def input_market_value() -> float:
    print()
    print("현재 선수 시장가를 입력하세요.")
    print("예: 25M, 25000000, 25,000,000")

    while True:
        raw_value = input("현재 시장가: ").strip().upper()

        try:
            cleaned = (
                raw_value
                .replace("€", "")
                .replace(",", "")
                .strip()
            )

            if cleaned.endswith("M"):
                value = float(cleaned[:-1]) * 1_000_000

            elif cleaned.endswith("K"):
                value = float(cleaned[:-1]) * 1_000

            else:
                value = float(cleaned)

        except ValueError:
            print("예: 25M 또는 25000000 형식으로 입력하세요.")
            continue

        if value <= 0:
            print("시장가는 0보다 큰 값이어야 합니다.")
            continue

        return value


# ============================================================
# 예측 입력 생성
# ============================================================

def calculate_age(
    date_of_birth,
    prediction_date: pd.Timestamp,
) -> float:
    birth_date = pd.to_datetime(
        date_of_birth,
        errors="coerce",
    )

    if pd.isna(birth_date):
        return np.nan

    return (
        prediction_date - birth_date
    ).days / 365.25


def build_prediction_input(
    player: pd.Series,
    to_league_id: str,
    market_value: float,
) -> pd.DataFrame:
    from_league_id = str(
        player["current_league_id"]
    ).strip()

    prediction_input = pd.DataFrame(
        [
            {
                "value_at_transfer": market_value,
                "age_at_transfer": calculate_age(
                    player["date_of_birth"],
                    PREDICTION_DATE,
                ),
                "height": pd.to_numeric(
                    player["height"],
                    errors="coerce",
                ),
                "matches": pd.to_numeric(
                    player["matches"],
                    errors="coerce",
                ),
                "started": pd.to_numeric(
                    player["started"],
                    errors="coerce",
                ),
                "goals": pd.to_numeric(
                    player["goals"],
                    errors="coerce",
                ),
                "assists": pd.to_numeric(
                    player["assists"],
                    errors="coerce",
                ),
                "minutes": pd.to_numeric(
                    player["minutes"],
                    errors="coerce",
                ),
                "rating": pd.to_numeric(
                    player["rating"],
                    errors="coerce",
                ),
                "is_same_league": int(
                    from_league_id == to_league_id
                ),
                "is_top5_destination": int(
                    to_league_id in TOP5_LEAGUE_IDS
                ),
                "from_league_id": from_league_id,
                "to_league_id": to_league_id,
                "main_position": player["main_position"],
                "foot": player["foot"],
            }
        ]
    )

    return prediction_input[FEATURES]


# ============================================================
# 예측
# ============================================================

def predict_transfer_fee(
    model,
    prediction_input: pd.DataFrame,
) -> float:
    predicted_log_fee = model.predict(
        prediction_input
    )[0]

    predicted_fee = np.expm1(
        predicted_log_fee
    )

    return float(
        max(predicted_fee, 0)
    )


# ============================================================
# 메인 실행
# ============================================================

def main() -> None:
    print_section("Transfer Fee Prediction V1")

    model = load_model()
    prediction_df = load_prediction_data()

    print(f"예측 가능 선수 수: {len(prediction_df):,}")
    print(
        f"예측 기준일      : "
        f"{PREDICTION_DATE.date()}"
    )

    player = select_player(prediction_df)

    (
        to_league_id,
        to_league_name,
    ) = select_destination_league()

    market_value = input_market_value()

    prediction_input = build_prediction_input(
        player=player,
        to_league_id=to_league_id,
        market_value=market_value,
    )

    predicted_fee = predict_transfer_fee(
        model=model,
        prediction_input=prediction_input,
    )

    print_section("예측 결과")

    print(
        f"선수             : "
        f"{player['player_name']}"
    )
    print(
        f"현재 소속        : "
        f"{player['current_club_name']}"
    )
    print(
        f"현재 리그        : "
        f"{player['current_league_name']}"
    )
    print(
        f"현재 시장가      : "
        f"{format_euro(market_value)}"
    )
    print(
        f"예상 이적 리그   : "
        f"{to_league_name}"
    )
    print(
        f"나이             : "
        f"{prediction_input.iloc[0]['age_at_transfer']:.1f}세"
    )
    print(
        f"최근 시즌 기록   : "
        f"{player['matches']}경기 / "
        f"{player['goals']}골 / "
        f"{player['assists']}도움 / "
        f"평점 {player['rating']}"
    )
    print()
    print(
        f"예상 이적료      : "
        f"{format_euro(predicted_fee)}"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n예측을 종료합니다.")
        sys.exit(0)

    except Exception as error:
        print_section("예측 실행 실패")
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)