from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]

# Primary K per group, from the silhouette and face-validity checks
PRIMARY_K = {"D": 2, "M": 3, "F": 2}

# Weights for the two new factors. Change these and record the choice.
W_ROLE = 1.0      # penalty added if the two players are in different clusters
W_QUALITY = 1.0   # penalty added per standard deviation of rating difference

# Match rating from the raw file (not in the cleaned file)
raw = pd.read_csv(ROOT / "data_raw" / "pl_2025-26_gw35.csv")
if "rating" not in raw.columns:
    raise SystemExit("No 'rating' column in the raw file. Check the column names.")
rating = raw[["player_name", "rating"]].drop_duplicates("player_name")

FEATURES, GROUPS = {}, {}
for g, k in PRIMARY_K.items():
    df = pd.read_csv(ROOT / "data_processed" / f"players_{g}_scaled.csv")
    X = df.drop(columns=ID_COLS)
    FEATURES[g] = X

    df["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    df = df.merge(rating, on="player_name", how="left")

    # Fill missing ratings with the group average, and report how many
    missing = df["rating"].isna().sum()
    if missing:
        print(f"Group {g}: {missing} players had no rating; filled with the group mean")
    df["rating"] = df["rating"].fillna(df["rating"].mean())
    df["rating_z"] = (df["rating"] - df["rating"].mean()) / df["rating"].std()

    GROUPS[g] = df


def find_similar(player_name, top_n=5):
    group = next((g for g, df in GROUPS.items() if player_name in df["player_name"].values), None)
    if group is None:
        raise ValueError(f"'{player_name}' not found. Check spelling, or the player may be a goalkeeper or below 900 minutes.")

    df = GROUPS[group]
    i = df.index[df["player_name"] == player_name][0]

    feature_dist = pairwise_distances(FEATURES[group].iloc[[i]], FEATURES[group])[0]
    role_pen = (df["cluster"].values != df.loc[i, "cluster"]).astype(float)
    quality_gap = abs(df["rating_z"].values - df.loc[i, "rating_z"])
    score = feature_dist + W_ROLE * role_pen + W_QUALITY * quality_gap

    out = df[["player_name", "team_name", "cluster", "rating"]].copy()
    out["feature_dist"] = feature_dist
    out["role_pen"] = role_pen
    out["quality_gap"] = quality_gap
    out["score"] = score

    out = out[out["player_name"] != player_name]
    return out.nsmallest(top_n, "score").reset_index(drop=True).round(2)


if __name__ == "__main__":
    for name in ["Bukayo Saka", "Cole Palmer", "Erling Haaland"]:
        try:
            base = GROUPS[next(g for g, d in GROUPS.items() if name in d["player_name"].values)]
            row = base[base["player_name"] == name].iloc[0]
            print(f"\n{name} (cluster {row['cluster']}, rating {row['rating']:.2f})")
            print(find_similar(name, top_n=5).to_string(index=False))
        except (ValueError, StopIteration) as e:
            print(e)