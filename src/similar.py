from pathlib import Path
import pandas as pd
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]

# Load each group once
GROUPS = {}
for g in ["D", "M", "F"]:
    GROUPS[g] = pd.read_csv(ROOT / "data_processed" / f"players_{g}_scaled.csv")


def find_similar(player_name, top_n=5):
    """Return the top_n most similar players in the same position group."""
    # Find which group the player is in
    group = None
    for g, df in GROUPS.items():
        if player_name in df["player_name"].values:
            group = g
            break
    if group is None:
        raise ValueError(f"'{player_name}' not found. Check spelling, or note the player may be a goalkeeper or below 900 minutes.")

    df = GROUPS[group]
    X = df.drop(columns=ID_COLS)

    # Distance from the query player to everyone in the same group
    idx = df.index[df["player_name"] == player_name][0]
    dists = pairwise_distances(X.iloc[[idx]], X)[0]

    result = df[["player_name", "team_name", "position"]].copy()
    result["distance"] = dists
    result = result[result["player_name"] != player_name]  # drop the player themselves
    return result.nsmallest(top_n, "distance").reset_index(drop=True)


if __name__ == "__main__":
    for name in ["Bukayo Saka", "Cole Palmer", "Erling Haaland"]:
        try:
            print(f"\nPlayers similar to {name}:")
            print(find_similar(name, top_n=5).to_string(index=False))
        except ValueError as e:
            print(e)