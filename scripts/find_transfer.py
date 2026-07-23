import pandas as pd

profiles = pd.read_csv("data/raw/player_profiles.csv", low_memory=False)
transfers = pd.read_csv("data/raw/transfer_history.csv", low_memory=False)

name = input("선수 이름 입력: ").strip()

matched_players = profiles[
    profiles["player_name"]
    .astype(str)
    .str.contains(name, case=False, na=False, regex=False)
]

if matched_players.empty:
    print("일치하는 선수를 찾지 못했습니다.")
else:
    player_columns = [
        col
        for col in [
            "player_id",
            "player_name",
            "current_club_name",
            "position",
            "date_of_birth",
        ]
        if col in matched_players.columns
    ]

    print("\n검색된 선수")
    print(matched_players[player_columns].to_string(index=False))

    player_ids = matched_players["player_id"].astype(str)

    transfer_result = transfers[
        transfers["player_id"].astype(str).isin(player_ids)
    ]

    transfer_fee = pd.to_numeric(
        transfer_result["transfer_fee"],
        errors="coerce",
    ).fillna(0)

    transfer_result = transfer_result[
        (transfer_fee > 0)
        & (
            transfer_result["transfer_type"]
            .astype(str)
            .str.strip()
            .eq("Transfer")
        )
    ]

    columns = [
        col
        for col in [
            "player_id",
            "season_name",
            "transfer_date",
            "from_team_name",
            "to_team_name",
            "value_at_transfer",
            "transfer_fee",
        ]
        if col in transfer_result.columns
    ]

    if transfer_result.empty:
        print("\n유상 완전이적 기록이 없습니다.")
    else:
        print("\n유상 완전이적 기록")
        print(transfer_result[columns].to_string(index=False))
