from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data_processed" / "players_clean.csv")

missing = df[df["expectedGoals_p90"].isna()]
print("Missing count:", len(missing))
print("\nBy position:")
print(missing["position"].value_counts())
print("\nMinutes played (min, median, max):",
      missing["minutesPlayed"].min(),
      missing["minutesPlayed"].median(),
      missing["minutesPlayed"].max())
print("\nSample:")
print(missing[["player_name", "team_name", "position", "minutesPlayed"]].head(10))

# Compare: do the same players have a shots count?
print("\nTotal shots_p90 for these players (should be 0 or NaN if they never shot):")
print(missing.get("totalShots_p90", pd.Series(dtype=float)).describe())