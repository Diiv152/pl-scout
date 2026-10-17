# Data sources

## Dataset

- File: pl_2025-26_gw35.csv
- Description: Premier League 2025-26 player statistics, data up to and including Gameweek 35
- Rows: 526 players, 106 columns
- Original filename: premier_league_complete_stats_until35thGameDayOnSeason2025-26.csv
- Source: [where you got the file, such as a website or dataset name and URL]
- Accessed: [today's date]
- Licence / terms: [what the source says about reuse, or "unknown, to check"]

## Processing decisions

- Players with fewer than 900 minutes played were excluded, leaving 326 players.
- Players with no recorded shots had their expected goals set to 0. This applies to 25 players: 23 goalkeepers and 2 outfield players. Checked: all 25 have zero total shots.
- Team names were cleaned of HTML encoding (for example, "&amp;" was changed to "&").
- Goalkeepers (24 players) were excluded from clustering: the group is too small for stable clusters, and the file lacks goalkeeper-specific measures.
- Players were split into three position groups (defenders, midfielders, forwards) and clustered separately.
- Feature redundancy checked with pairwise correlation (threshold 0.9): no pairs found in defenders, midfielders, or forwards. All 18 features kept per group.
- Redundant features removed where pairwise correlation exceeded 0.7, keeping the less noisy or less redundant measure of each pair (for example, expected goals over goals; key passes over expected assists; total shots over shots on target). Drops per group: defenders 3, midfielders 4, forwards 4.
- The 0.7 threshold and the choice of which feature to keep were judgement calls, and are tested in the sensitivity analysis.
- Standardisation uses scikit-learn's StandardScaler (population standard deviation). Pandas' sample standard deviation reads about 1.004 (defenders, midfielders) and 1.009 (forwards), which is expected and not an error.
- K-means K chosen by silhouette score over K = 2 to 8. Silhouette peaked at K = 2 in all groups but was low (about 0.18 to 0.21), indicating weak cluster structure. K = 2 is the primary setting; K = 3 is reported as a sensitivity check.
- Cluster interpretation (face validity): defenders K=2 (central defenders vs full-backs), midfielders K=3 (attacking, defensive, box-to-box), forwards K=2 (wide attackers vs centre-forwards), with K=3 reported as a sensitivity check. Interpretation was made by the author, not a formal test.
- Similar-player score = standardised feature distance + role penalty (1 if different cluster, 0 if same) + quality penalty (absolute difference in standardised match rating). Weights set to 1.0 each as a starting point, and to be tested in sensitivity analysis.
- Quality proxy: match rating from the GW35 file. Market value was not available in the source file and was not used.
- Missing ratings were filled with the group mean.
- Similar-player test (Saka, Palmer, Haaland) with W_ROLE = W_QUALITY = 1.0: the quality factor reordered the top five toward players of similar rating (e.g. Palmer's list now favours Foden, Summerville, Eze). The role penalty changed no top-five results for these three players, because feature distance already keeps similar styles in the same cluster.
- Role-weight sensitivity (302 players in defender, midfield, and forward groups; goalkeepers excluded): with no role penalty, 120 queries had a cross-cluster match in the top five. Penalty 1.0 reduced this to 31; penalty 3.0 removed all. Quality weight raised cross-cluster matches (12 at quality 0, 31 at 1, 78 at 3, with role penalty 1.0). Default W_ROLE = W_QUALITY = 1.0 chosen as a balance between style and role.
- Neighbour-cluster agreement (top five by feature distance, no role or quality terms): defenders 0.915 vs 0.500 chance (K=2); midfielders 0.777 vs 0.349 chance (K=3); forwards 0.862 vs 0.513 chance (K=2). Permutation test with 1,000 shuffles, p < 0.001 in all groups. Caveat: clusters and neighbours share the same feature space, so this partly reflects the method's internal consistency. An independent check (split features or external labels) is needed to support the claim further.
- Dashboard (src/app.py): club shown as three-letter Premier League code (for example MCI, LIV, ARS), set in TEAM_ABBR. Badges and market values not included.
- Player photos: taken from the Fantasy Premier League API when src/build_photos.py runs. Matching is by name and club. About [N] of 526 players have photos. Players not in the FPL data, including Mohamed Salah, Alisson and Rodri, show initials. Photos are for display only and are not used in the analysis.
- PL Scout Ranking (revised): weighted average of z-scored per-90 stats within each position group, using unscaled values so goals are included. Weights were set by the author (forwards: goals 2, expected goals 1, assists 1, shots 1, key passes 0.5, dribbles 0.5; other groups in src/app.py). Goals were excluded from the earlier radar-based ranking because they were redundant with expected goals for clustering, which made that ranking miss the main striker output. The ranking is a transparent judgement, not a validated measure of quality.