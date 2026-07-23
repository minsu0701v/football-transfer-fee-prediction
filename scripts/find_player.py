import pandas as pd 

df = pd.read_csv("data/raw/player_profiles.csv", low_memory=False)
name = input("선수 이름 입력: ").strip()

result = df[ df["player_name"] 
            .astype(str) 
            .str.contains(name, case=False, na=False, regex=False) 
            ]

columns = [
    col
    for col in [
        "player_id",
        "player_name",
        "current_club_name",
        "position",
        "date_of_birth",
    ]
    if col in result.columns
]

if result.empty:
    print("일치하는 선수를 찾지 못했습니다.")
else:
    print(result[columns].to_string(index=False))





















