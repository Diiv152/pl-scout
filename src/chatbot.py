import re
import pandas as pd
import os
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash") # if this name is rejected, check the current name at ai.google.dev
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]
POS_NAMES = {"D": "Defenders", "M": "Midfielders", "F": "Forwards"}

SYSTEM_PROMPT = """You are PL Scout, an assistant for a Premier League 2025-26 player analysis dashboard.

Rules:
- Answer only from the PLAYER DATA provided. Do not use your own memory of player stats.
- If the data does not contain the answer, say so plainly. Do not guess.
- The data covers the 2025-26 season up to Gameweek 35, for outfield players with 900+ minutes.
- The PL Scout Ranking is a stats-based guide within each position group, not a verdict on who is best.
- "Similar players" means similar playing style, not similar quality.
- Quote the numbers you use. Keep answers short and in plain English.
- If asked about prices, transfers, injuries or betting, say you can only answer from the dashboard data.
"""


def build_index(groups, similar_fn, abbr_fn):
    """Pre-build text for every player and for the rankings. Runs once, then is cached."""
    players, rankings = {}, []
    for g, (df, X, F) in groups.items():
        top = df.sort_values("ps_rank").head(10)
        lines = [f"{POS_NAMES[g]}: top 10 by PL Scout Ranking (out of {len(df)})"]
        for _, r in top.iterrows():
            lines.append(f"#{int(r['ps_rank'])} {r['player_name']} ({abbr_fn(r['team_name'])}), rating {r['rating']:.2f}")
        rankings.append("\n".join(lines))

        feature_cols = [c for c in F.columns if c not in ID_COLS]
        for idx, r in df.iterrows():
            stats = F.loc[idx, feature_cols].round(2)
            stat_text = ", ".join(f"{k}={v}" for k, v in stats.items())
            goals = int(r["goals"]) if pd.notna(r["goals"]) else 0
            assists = int(r["assists"]) if pd.notna(r["assists"]) else 0
            players[r["player_name"]] = (
                f"{r['player_name']} ({abbr_fn(r['team_name'])}, {POS_NAMES[g][:-1]}). "
                f"Scout ranking #{int(r['ps_rank'])} of {len(df)}. Rating {r['rating']:.2f}. "
                f"Minutes {int(r['minutesPlayed'])}. Goals {goals}, assists {assists}. "
                f"Stats: {stat_text}. "
                f"Similar players: {', '.join(similar_fn(r['player_name']))}."
            )
    return {"rankings": "\n\n".join(rankings), "players": players}


def relevant_context(index, question):
    """Send the rankings plus only the players the question mentions (up to 6)."""
    q = question.lower()
    hits = [n for n in index["players"] if n.lower() in q]
    if not hits:
        words = set(re.findall(r"[\w'-]+", q))
        hits = [n for n in index["players"] if n.split()[-1].lower() in words]
    parts = [index["rankings"]] + [index["players"][n] for n in hits[:6]]
    return "\n\n".join(parts)


def api_key_present():
    return bool(os.getenv("GEMINI_API_KEY"))


def _config(system, thinking_off):
    kwargs = dict(system_instruction=system, temperature=0.2, max_output_tokens=400)
    if thinking_off:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def ask_stream(question, history, context):
    """Yields the answer piece by piece, so it appears while it is being written."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    contents = []
    for m in history[-4:]:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    system = SYSTEM_PROMPT + "\n\nPLAYER DATA\n" + context

    # Try with thinking switched off (fastest). If this model rejects that setting,
    # retry without it.
    last_error = None
    for thinking_off in (True, False):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL, contents=contents, config=_config(system, thinking_off))
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            last_error = e
    