from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data_raw" / "fpl_players_2025-26.csv")

print("Data type of minutes:", df["minutes"].dtype)
print("Max minutes:", df["minutes"].max())
print("Players with 900+ minutes:", (df["minutes"] >= 900).sum())
print("\nTop 5 by minutes:")
print(df[["web_name", "minutes"]].sort_values("minutes", ascending=False).head(5))