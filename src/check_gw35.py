from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data_raw" / "pl_2025-26_gw35.csv")

print("Shape:", df.shape)
print("Duplicate names:", df["player_name"].duplicated().sum())
print("Max minutesPlayed:", df["minutesPlayed"].max())
print("Players with 900+ minutes:", (df["minutesPlayed"] >= 900).sum())
print("Positions:", df["position"].value_counts().to_dict())