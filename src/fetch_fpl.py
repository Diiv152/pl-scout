import requests
import pandas as pd

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

response = requests.get(URL, timeout=30)
response.raise_for_status()
data = response.json()

players = pd.DataFrame(data["elements"])
teams = pd.DataFrame(data["teams"])
positions = pd.DataFrame(data["element_types"])

# Attach readable team and position names
players["team_name"] = players["team"].map(teams.set_index("id")["name"])
players["position"] = players["element_type"].map(positions.set_index("id")["singular_name"])

players.to_csv("data_raw/fpl_players_2025-26.csv", index=False)

from pathlib import Path

out_dir = Path(__file__).resolve().parent.parent / "data_raw"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "fpl_players_2025-26.csv"
players.to_csv(out_file, index=False)
print("Saved", len(players), "players to", out_file)