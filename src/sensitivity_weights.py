from pathlib import Path
import itertools
import pandas as pd

import similar_v2 as s  # reuses the clusters and scores from similar_v2.py

results = []
for w_role, w_qual in itertools.product([0.0, 1.0, 3.0], [0.0, 1.0, 3.0]):
    s.W_ROLE, s.W_QUALITY = w_role, w_qual
    changed = 0
    total = 0
    for g, df in s.GROUPS.items():
        for name in df["player_name"]:
            top = s.find_similar(name, top_n=5)["player_name"].tolist()
            baseline = s.find_similar(name, top_n=5)  # same call, kept for clarity
            total += 1
            # count players whose top-5 contains anyone from a different cluster
            if (top and any(
                df.loc[df["player_name"] == p, "cluster"].values[0] != df.loc[df["player_name"] == name, "cluster"].values[0]
                for p in top
            )):
                changed += 1
    results.append((w_role, w_qual, changed, total))

print("W_ROLE  W_QUALITY  queries_with_cross_cluster_match  of_total")
for r in results:
    print(f"{r[0]:<7} {r[1]:<10} {r[2]:<32} {r[3]}")