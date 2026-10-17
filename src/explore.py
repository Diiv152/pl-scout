from pathlib import Path
import pandas as pd

# Path to the file, built from this script's location so it works from any folder
DATA = Path(__file__).resolve().parent.parent / "data_raw" / "fpl_players_2025-26.csv"

df = pd.read_csv(DATA)

print("Rows and columns:", df.shape)
print("\nColumn names:")
print(list(df.columns))
print("\nFirst 3 rows (a few columns):")
print(df[["web_name", "team_name", "position", "minutes", "goals_scored", "assists"]].head(3))
print("\nPlayers with zero minutes:", (df["minutes"] == 0).sum())