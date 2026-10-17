from pathlib import Path
import re
import unicodedata
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"


def norm(s):
    # Lowercase, remove accents and punctuation, so "Martínez" matches "Martinez"
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# Fetch player records directly from the API. Nothing is saved to data_raw.
data = requests.get(API_URL, timeout=30).json()
fpl = pd.DataFrame(data["elements"])
fpl = fpl[fpl["photo"].notna()].copy()
fpl["photo_url"] = fpl["photo"].str.replace(".jpg", "", regex=False).map(
    lambda p: PHOTO_URL.format(code=p)
)

full_key = (fpl["first_name"].fillna("") + " " + fpl["second_name"].fillna("")).map(norm)
web_key = fpl["web_name"].map(norm)

# Full-name matches take priority over nickname matches
lookup = pd.concat([
    pd.DataFrame({"key": full_key, "photo_url": fpl["photo_url"]}),
    pd.DataFrame({"key": web_key, "photo_url": fpl["photo_url"]}),
]).drop_duplicates("key")

gw = pd.read_csv(ROOT / "data_raw" / "pl_2025-26_gw35.csv")[["player_name"]]
gw["key"] = gw["player_name"].map(norm)

out = gw.merge(lookup, on="key", how="left")[["player_name", "photo_url"]]
out.to_csv(ROOT / "data_processed" / "player_photos.csv", index=False)

print(f"Photos matched: {out['photo_url'].notna().sum()} of {len(out)}")
print("Not matched (will show initials):")
print(out[out["photo_url"].isna()]["player_name"].tolist())