from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Only the feature columns, not the names or positions
for group in ["D", "M", "F"]:
    df = pd.read_csv(ROOT / "data_processed" / f"players_{group}.csv")
    features = df.drop(columns=["player_name", "team_name", "position", "minutesPlayed"])

    corr = features.corr().abs()

    # Look at the upper triangle only, so each pair appears once
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
           if corr.iloc[i, j] > 0.7:
                pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))

    print(f"\n=== Group {group}: {len(features.columns)} features ===")
    if pairs:
        for a, b, r in pairs:
            print(f"{a}  <->  {b}   r = {r}")
    else:
        print("No pairs above 0.7")
    