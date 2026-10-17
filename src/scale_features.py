from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]

for group in ["D", "M", "F"]:
    df = pd.read_csv(ROOT / "data_processed" / f"players_{group}_features.csv")

    ids = df[ID_COLS]
    features = df.drop(columns=ID_COLS)

    # Fit the scaler on this group only, so each position group is scaled separately
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Check: means should be about 0 and standard deviations about 1
    print(f"\nGroup {group}")
    print("  largest |mean|:", round(scaled.mean().abs().max(), 6))
    print("  std range:", round(scaled.std().min(), 3), "to", round(scaled.std().max(), 3))

    out = pd.concat([ids.reset_index(drop=True), scaled], axis=1)
    out.to_csv(ROOT / "data_processed" / f"players_{group}_scaled.csv", index=False)
    print(f"  saved {out.shape[0]} players to players_{group}_scaled.csv")