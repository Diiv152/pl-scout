from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]

for group in ["D", "M", "F"]:
    df = pd.read_csv(ROOT / "data_processed" / f"players_{group}_scaled.csv")
    X = df.drop(columns=ID_COLS)

    for k in [2, 3]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        df["cluster"] = km.fit_predict(X)

        print(f"\n===== Group {group}, K={k} =====")
        print("Cluster sizes:", df["cluster"].value_counts().sort_index().to_dict())

        # Top 8 players in each cluster, ranked by distance to the cluster centre
        centres = km.cluster_centers_
        for c in range(k):
            members = df[df["cluster"] == c].copy()
            dist = ((X.loc[members.index] - centres[c]) ** 2).sum(axis=1) ** 0.5
            top = members.assign(dist=dist).nsmallest(8, "dist")
            print(f"\n  Cluster {c} (most typical players):")
            print("   ", ", ".join(top["player_name"].tolist()))