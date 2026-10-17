from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data_processed" / "players_clean.csv")

print("Total players:", len(df))
print("\nCount per position:")
print(df["position"].value_counts())

# Save one file per position group
for pos in df["position"].unique():
    group = df[df["position"] == pos]
    out = ROOT / "data_processed" / f"players_{pos}.csv"
    group.to_csv(out, index=False)
    print(f"Saved {len(group)} rows to {out.name}")