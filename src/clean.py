from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data_raw" / "pl_2025-26_gw35.csv")

MIN_MINUTES = 900  # about 10 full matches

# Keep only players who played enough minutes
df = df[df["minutesPlayed"] >= MIN_MINUTES].copy()
print("Players after minutes filter:", len(df))

# Count stats to convert to per-90 values
count_stats = [
    "goals", "assists", "expectedGoals", "expectedAssists",
    "totalShots", "shotsOnTarget", "keyPasses", "totalPasses",
    "successfulDribbles", "tackles", "interceptions",
    "clearances", "ballRecovery", "aerialDuelsWon",
]

# Only use columns that actually exist in the file
missing_cols = [c for c in count_stats if c not in df.columns]
if missing_cols:
    print("Columns not found in file:", missing_cols)
count_stats = [c for c in count_stats if c in df.columns]

for col in count_stats:
    df[col + "_p90"] = df[col] / df["minutesPlayed"] * 90

# Blanks are players with zero shots (checked: all 25 have totalShots_p90 = 0)
df["expectedGoals_p90"] = df["expectedGoals_p90"].fillna(0)

# Fix HTML encoding left in the source data
df["team_name"] = df["team_name"].str.replace("&amp;", "&", regex=False)

# Keep identity columns, per-90 features, and the percentage stats
keep = ["player_name", "team_name", "position", "minutesPlayed"]
keep += [c + "_p90" for c in count_stats]
keep += ["accuratePassesPercentage", "successfulDribblesPercentage",
         "tacklesWonPercentage", "aerialDuelsWonPercentage"]
keep = [c for c in keep if c in df.columns]
clean = df[keep].copy()

# Report missing values before deciding what to do with them
na = clean.isna().sum()
print("\nMissing values per kept column:")
print(na[na > 0] if (na > 0).any() else "none")

out_file = ROOT / "data_processed" / "players_clean.csv"
clean.to_csv(out_file, index=False)
print("\nSaved", clean.shape, "to", out_file)