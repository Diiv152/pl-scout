from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]

for group in ["D", "M", "F"]:
    df = pd.read_csv(ROOT / "data_processed" / f"players_{group}_scaled.csv")
    X = df.drop(columns=ID_COLS)

    print(f"\nGroup {group} ({len(X)} players)")
    print("  K   silhouette   inertia")
    for k in range(2, 9):
        # n_init=20 runs K-means 20 times from different starts and keeps the best
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"  {k}   {sil:.3f}        {km.inertia_:.1f}")