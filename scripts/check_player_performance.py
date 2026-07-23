import pandas as pd

df = pd.read_csv("data/raw/player_performances.csv", low_memory=False)

player_id = int(input("Transfermarkt player_id: ").strip())
season = input("시즌 (예: 24/25): ").strip()

result = df[
    (df["player_id"] == player_id)
    & (df["season_name"] == season)
]

print(f"\n{season} 시즌 기록 수: {len(result)}")

if result.empty:
    print("기록이 없습니다.")
else:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(result.to_string(index=False))















