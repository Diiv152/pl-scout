from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]
PRIMARY_K = {"D": 2, "M": 3, "F": 2}
TOP_N = 5
N_PERM = 1000

rng = np.random.default_rng(42)
print("group  K  observed  permutation_mean  p_value")

for g, k in PRIMARY_K.items():
    df = pd.read_csv(ROOT / "data_processed" / f"players_{g}_scaled.csv")
    X = df.drop(columns=ID_COLS)
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)

    # Nearest neighbours by feature distance only (no role or quality terms)
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :TOP_N]

    # Share of top-five neighbours in the same cluster
    observed = (labels[nn] == labels[:, None]).mean()

    # Null: shuffle cluster labels, which keeps cluster sizes but breaks any link to the players
    perm = np.array([
        (rng.permutation(labels)[nn] == rng.permutation(labels)[:, None]).mean()
        for _ in range(N_PERM)
    ])
    p_value = (np.sum(perm >= observed) + 1) / (N_PERM + 1)

    print(f"{g}  {k}  {observed:.3f}    {perm.mean():.3f}            {p_value:.4f}")