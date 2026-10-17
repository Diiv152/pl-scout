from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Redundant features to drop, chosen from the 0.7 correlation check
DROP = {
    "D": ["expectedAssists_p90", "shotsOnTarget_p90", "aerialDuelsWon_p90"],
    "M": ["goals_p90", "shotsOnTarget_p90", "expectedAssists_p90", "aerialDuelsWon_p90"],
    "F": ["goals_p90", "shotsOnTarget_p90", "expectedAssists_p90", "totalPasses_p90"],
}

for group, drop_cols in DROP.items():
    df = pd.read_csv(ROOT / "data_processed" / f"players_{group}.csv")

    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        print(f"Group {group}: columns not found, check spelling:", missing)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    out = ROOT / "data_processed" / f"players_{group}_features.csv"
    df.to_csv(out, index=False)
    feature_count = len(df.columns) - 4  # minus player_name, team_name, position, minutesPlayed
    print(f"Group {group}: saved {len(df)} players, {feature_count} features -> {out.name}")