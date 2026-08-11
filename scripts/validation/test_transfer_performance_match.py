import pandas as pd

PROFILE_FILE = "data/raw/player_profiles.csv"
TRANSFER_FILE = "data/raw/transfer_history.csv"
PERFORMANCE_FILE = "data/raw/player_performances.csv"


def previous_season(season):
    start, end = season.split("/")

    return (
        f"{(int(start) - 1) % 100:02d}/"
        f"{(int(end) - 1) % 100:02d}"
    )


profiles = pd.read_csv(PROFILE_FILE, low_memory=False)
transfers = pd.read_csv(TRANSFER_FILE, low_memory=False)
performances = pd.read_csv(PERFORMANCE_FILE, low_memory=False)

transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce",
)

for df in [profiles, transfers, performances]:
    df["player_id"] = df["player_id"].astype(str)

name = input("선수 이름 입력: ").strip()

players = profiles[
    profiles["player_name"]
    .astype(str)
    .str.contains(name, case=False, na=False, regex=False)
]

if players.empty:
    print("일치하는 선수를 찾지 못했습니다.")
    raise SystemExit

print("\n검색된 선수")
print(
    players[
        [
            "player_id",
            "player_name",
            "current_club_name",
            "position",
            "date_of_birth",
        ]
    ].to_string(index=False)
)

player_id = input("\n확인할 player_id 입력: ").strip()

transfer_fee = pd.to_numeric(
    transfers["transfer_fee"],
    errors="coerce",
).fillna(0)

player_transfers = transfers[
    (transfers["player_id"] == player_id)
    & (transfer_fee > 0)
    & (transfers["transfer_type"] == "Transfer")
    & transfers["transfer_date"].dt.month.isin([6, 7, 8])
].copy()

if player_transfers.empty:
    print("유상 여름 완전이적 기록이 없습니다.")
    raise SystemExit

player_transfers = player_transfers.reset_index(drop=True)

display_columns = [
    "season_name",
    "transfer_date",
    "from_team_name",
    "to_team_name",
    "value_at_transfer",
    "transfer_fee",
]

display = player_transfers[display_columns].copy()
display.insert(0, "번호", range(1, len(display) + 1))

print("\n유상 여름 완전이적 기록")
print(display.to_string(index=False))

transfer_number = int(
    input("\n확인할 이적 번호 입력: ").strip()
)

selected = player_transfers.iloc[transfer_number - 1]

transfer_season = selected["season_name"]
performance_season = previous_season(transfer_season)

performance = performances[
    (performances["player_id"] == player_id)
    & (performances["season_name"] == performance_season)
].copy()

if performance.empty:
    print("직전 시즌 경기 기록이 없습니다.")
    raise SystemExit

for column in ["nb_on_pitch", "goals", "assists"]:
    performance[column] = pd.to_numeric(
        performance[column],
        errors="coerce",
    ).fillna(0)

print("\n직전 시즌 대회별 기록")
print(
    performance[
        [
            "competition_name",
            "team_name",
            "nb_on_pitch",
            "goals",
            "assists",
        ]
    ].to_string(index=False)
)

result = pd.DataFrame(
    [
        {
            "player_id": player_id,
            "transfer_season": transfer_season,
            "performance_season": performance_season,
            "transfer_date": selected["transfer_date"],
            "from_team_name": selected["from_team_name"],
            "to_team_name": selected["to_team_name"],
            "value_at_transfer": selected["value_at_transfer"],
            "transfer_fee": selected["transfer_fee"],
            "games": performance["nb_on_pitch"].sum(),
            "goals": performance["goals"].sum(),
            "assists": performance["assists"].sum(),
        }
    ]
)

print("\n최종 매칭 결과")
print(result.to_string(index=False))