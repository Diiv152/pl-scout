from pathlib import Path
import base64
import io
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import MODEL, build_index, relevant_context, ask_stream, api_key_present
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
ID_COLS = ["player_name", "team_name", "position", "minutesPlayed"]
PRIMARY_K = {"D": 2, "M": 3, "F": 2}
POS_NAMES = {"D": "Defender", "M": "Midfielder", "F": "Forward"}

TEAM_ABBR = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton & Hove Albion": "BHA", "Burnley": "BUR",
    "Chelsea": "CHE", "Crystal Palace": "CRY", "Everton": "EVE",
    "Fulham": "FUL", "Leeds United": "LEE", "Liverpool FC": "LIV",
    "Liverpool": "LIV", "Manchester City": "MCI", "Manchester United": "MUN",
    "Newcastle United": "NEW", "Nottingham Forest": "NFO", "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT", "West Ham United": "WHU",
    "Wolverhampton": "WOL", "Wolverhampton Wanderers": "WOL",
}

# Stats used on the radar chart, per position group
RADAR_STATS = {
    "D": ["tackles_p90", "interceptions_p90", "clearances_p90",
          "ballRecovery_p90", "tacklesWonPercentage", "aerialDuelsWonPercentage"],
    "M": ["keyPasses_p90", "successfulDribbles_p90", "ballRecovery_p90",
          "tackles_p90", "accuratePassesPercentage", "expectedGoals_p90"],
    "F": ["expectedGoals_p90", "totalShots_p90", "keyPasses_p90",
          "successfulDribbles_p90", "assists_p90", "successfulDribblesPercentage"],
}

# PL Scout Ranking weights, per position group. Judgement, not a fitted model.
RANK_WEIGHTS = {
    "F": {"goals_p90": 2.0, "expectedGoals_p90": 1.0, "assists_p90": 1.0,
          "totalShots_p90": 1.0, "keyPasses_p90": 0.5, "successfulDribbles_p90": 0.5},
    "M": {"goals_p90": 1.0, "assists_p90": 1.5, "expectedGoals_p90": 1.0,
          "keyPasses_p90": 1.5, "successfulDribbles_p90": 1.0,
          "ballRecovery_p90": 1.0, "tackles_p90": 0.5, "interceptions_p90": 0.5},
    "D": {"tackles_p90": 1.0, "interceptions_p90": 1.0, "clearances_p90": 1.0,
          "ballRecovery_p90": 1.0, "aerialDuelsWonPercentage": 1.0,
          "tacklesWonPercentage": 0.5, "assists_p90": 0.5, "keyPasses_p90": 0.5},
}

# Stats that can be compared between players in any position group
CANDIDATE_STATS = [
    "assists_p90", "expectedGoals_p90", "totalShots_p90", "keyPasses_p90",
    "successfulDribbles_p90", "tackles_p90", "interceptions_p90",
    "ballRecovery_p90", "clearances_p90", "successfulDribblesPercentage",
    "tacklesWonPercentage", "aerialDuelsWonPercentage", "accuratePassesPercentage",
]

LABELS = {
    "goals_p90": "Goals per 90", "assists_p90": "Assists per 90",
    "expectedGoals_p90": "Expected goals per 90", "totalShots_p90": "Shots per 90",
    "keyPasses_p90": "Key passes per 90", "successfulDribbles_p90": "Dribbles per 90",
    "successfulDribblesPercentage": "Dribble success %", "tackles_p90": "Tackles per 90",
    "interceptions_p90": "Interceptions per 90", "clearances_p90": "Clearances per 90",
    "ballRecovery_p90": "Ball recoveries per 90", "tacklesWonPercentage": "Tackles won %",
    "aerialDuelsWonPercentage": "Aerial duels won %", "accuratePassesPercentage": "Pass accuracy %",
}

COL_BG, COL_PANEL, COL_GRID = "#0A0F0C", "#121C16", "#2C4A3A"
COL_TEXT, COL_ME, COL_RIVAL = "#E8F5EC", "#00FF85", "#FF8A00"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Barlow:wght@400;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.5px; }}
.stApp {{ background: {COL_BG}; color: {COL_TEXT}; }}
div[data-testid="stMetric"] {{
    background: {COL_PANEL}; border-left: 3px solid {COL_ME};
    padding: 10px 14px; border-radius: 6px;
}}
div[data-testid="stMetricValue"] {{ color: {COL_ME}; font-family: 'Barlow Condensed', sans-serif; }}
.season {{ color: {COL_ME}; font-family: 'Barlow Condensed', sans-serif; font-size: 0.55em;
          vertical-align: middle; margin-left: 10px; border: 1px solid {COL_ME};
          padding: 2px 8px; border-radius: 4px; }}
.st-key-pl_chat {{ position: fixed; bottom: 24px; right: 24px; z-index: 999; }}
</style>
"""


def abbr(team):
    return TEAM_ABBR.get(team, str(team)[:3].upper())


def num(v):
    return int(v) if pd.notna(v) else 0


@st.cache_data
def load_data():
    raw = (pd.read_csv(ROOT / "data_raw" / "pl_2025-26_gw35.csv")
           [["player_name", "rating", "goals", "assists"]]
           .drop_duplicates("player_name"))

    groups = {}
    for g, k in PRIMARY_K.items():
        df = pd.read_csv(ROOT / "data_processed" / f"players_{g}_scaled.csv").reset_index(drop=True)
        X = df.drop(columns=ID_COLS).reset_index(drop=True)
        df["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)

        # PL Scout Ranking: weighted average of z-scored unscaled per-90 stats
        src = pd.read_csv(ROOT / "data_processed" / f"players_{g}.csv")
        w = RANK_WEIGHTS[g]
        Z = pd.DataFrame({c: (src[c] - src[c].mean()) / src[c].std() for c in w}).fillna(0)
        score = sum(w[c] * Z[c] for c in w) / sum(w.values())
        rank_df = pd.DataFrame({"player_name": src["player_name"].values,
                                "ps_score": score.values})
        df = df.merge(rank_df, on="player_name", how="left")
        df["ps_rank"] = df["ps_score"].rank(ascending=False, method="min").astype(int)
        df["ps_n"] = len(df)

        df = df.merge(raw, on="player_name", how="left")
        df["rating"] = df["rating"].fillna(df["rating"].mean())
        df["rating_z"] = (df["rating"] - df["rating"].mean()) / df["rating"].std()
        F = pd.read_csv(ROOT / "data_processed" / f"players_{g}_features.csv").reset_index(drop=True)
        groups[g] = (df, X, F)

    photos = {}
    photo_path = ROOT / "data_processed" / "player_photos.csv"
    if photo_path.exists():
        p = pd.read_csv(photo_path).dropna(subset=["photo_url"])
        photos = dict(zip(p["player_name"], p["photo_url"]))
    return groups, photos


def find_group(groups, name):
    return next((g for g, (df, _, _) in groups.items() if name in df["player_name"].values), None)


def locate(groups, name):
    g = find_group(groups, name)
    df = groups[g][0]
    return g, df.index[df["player_name"] == name][0]


def similar(groups, name, top_n, w_role, w_qual, min_min, min_rat, excl_club):
    g, i = locate(groups, name)
    df, X, _ = groups[g]

    style_gap = pairwise_distances(X.iloc[[i]], X)[0]
    role_pen = (df["cluster"].values != df["cluster"].iloc[i]).astype(float)
    rating_gap = np.abs(df["rating_z"].values - df["rating_z"].iloc[i])
    score = style_gap + w_role * role_pen + w_qual * rating_gap

    # np.array makes a writable copy, which avoids the read-only error
    mask = np.array((df["minutesPlayed"] >= min_min) & (df["rating"] >= min_rat), dtype=bool)
    mask &= np.arange(len(df)) != i
    if excl_club:
        mask &= np.array(df["team_name"] != df["team_name"].iloc[i], dtype=bool)

    out = pd.DataFrame({
        "idx": np.arange(len(df)),
        "Player": df["player_name"].values,
        "Team": df["team_name"].values,
        "Rating": df["rating"].values,
        "Minutes": df["minutesPlayed"].values,
        "Style gap": style_gap,
        "Score": score,
    })
    out = out[mask].nsmallest(top_n, "Score").reset_index(drop=True)
    return g, i, out


@st.cache_data
def load_index(_groups):
    # The leading underscore stops Streamlit hashing the groups object
    return build_index(
        _groups,
        lambda n: similar(_groups, n, 5, 1.0, 1.0, 900, 6.0, False)[2]["Player"].tolist(),
        abbr,
    )


def radar_figure(X, stats, i, j, name_a, name_b):
    cols = [c for c in stats if c in X.columns]
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    closed = angles + angles[:1]

    def values(idx):
        v = X.iloc[idx][cols].clip(-2.5, 2.5).values + 2.5  # 2.5 = position average
        return np.concatenate([v, v[:1]])

    fig = plt.figure(figsize=(5, 5), facecolor=COL_BG)
    ax = fig.add_subplot(111, polar=True, facecolor=COL_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([LABELS.get(c, c) for c in cols], color=COL_TEXT, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([2.5])
    ax.set_yticklabels([])
    ax.grid(color=COL_GRID, linewidth=0.8)
    ax.spines["polar"].set_color(COL_GRID)

    ax.plot(closed, values(i), color=COL_ME, linewidth=2.2, label=name_a)
    ax.fill(closed, values(i), color=COL_ME, alpha=0.25)
    ax.plot(closed, values(j), color=COL_RIVAL, linewidth=2.2, label=name_b)
    ax.fill(closed, values(j), color=COL_RIVAL, alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              labelcolor=COL_TEXT, frameon=False, fontsize=9)
    return fig


def show_radar(fig):
    # Vector (SVG) output stays sharp at any size; capped at 420px in both views
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", facecolor=COL_BG, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f"<img src='data:image/svg+xml;base64,{b64}' "
        f"style='width:100%;max-width:420px;display:block;margin:0 auto'>",
        unsafe_allow_html=True,
    )


def stat_table(Fa, ia, Fb, ib, stats, a, b):
    rows = []
    for c in stats:
        rows.append({
            "Stat": LABELS.get(c, c),
            a: round(float(Fa.iloc[ia][c]), 2),
            b: round(float(Fb.iloc[ib][c]), 2),
        })
    return pd.DataFrame(rows)


def highlight_higher(d, a, b):
    styles = pd.DataFrame("", index=d.index, columns=d.columns)
    green = "background-color: #00FF85; color: #0A0F0C; font-weight: 600"
    styles.loc[d[a] > d[b], a] = green
    styles.loc[d[b] > d[a], b] = green
    return styles


def show_stat_table(d, a, b):
    st.dataframe(d.style.apply(highlight_higher, a=a, b=b, axis=None),
                 hide_index=True, width="stretch")
    st.caption("Per 90 minutes, except percentages. The higher value in each row is highlighted green.")


def player_photo(name, photos, width=120):
    url = photos.get(name)
    if url:
        st.image(url, width=width)
    else:
        initials = "".join(w[0] for w in name.split()[:2]).upper()
        st.markdown(
            f"<div style='width:{width}px;height:{int(width * 1.27)}px;background:{COL_PANEL};"
            f"border-left:3px solid {COL_ME};display:flex;align-items:center;justify-content:center;"
            f"font-family:Barlow Condensed,sans-serif;font-size:36px;color:{COL_ME}'>{initials}</div>",
            unsafe_allow_html=True,
        )


def player_card(name, groups, photos):
    g, i = locate(groups, name)
    row = groups[g][0].iloc[i]
    player_photo(name, photos, width=150)
    st.markdown(f"**{name}**")
    st.caption(f"{row['team_name']} ({abbr(row['team_name'])}) · {POS_NAMES[g]}")
    st.metric("PL Scout Ranking", f"#{row['ps_rank']} of {row['ps_n']}")

def floating_chat(groups):
    index = load_index(groups)
    with st.container(key="pl_chat"):
        with st.popover("💬 Ask PL Scout"):
            st.caption("Answers come only from the dashboard data, up to Gameweek 35 of 2025-26.")
            if not api_key_present():
                st.info("No Gemini key found. Add GEMINI_API_KEY to the .env file and restart the app.")
                return

            if "chat" not in st.session_state:
                st.session_state.chat = []

            if st.button("Clear chat"):
                st.session_state.chat = []
                st.rerun()

            # Previous messages first
            for m in st.session_state.chat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            with st.form("chat_form", clear_on_submit=True):
                question = st.text_input("Your question",
                                         placeholder="e.g. Who is the top-ranked forward?")
                sent = st.form_submit_button("Send")

            # New question: show it, then stream the answer as it's written
            if sent and question.strip():
                q = question.strip()
                history = list(st.session_state.chat)
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    try:
                        answer = st.write_stream(
                            ask_stream(q, history, relevant_context(index, q)))
                    except Exception as e:
                        answer = (f"Something went wrong: {e}. If this mentions a quota or "
                                  "rate limit, wait a minute or try again later.")
                        st.markdown(answer)
                st.session_state.chat += [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ]
st.set_page_config(page_title="PL Scout", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("# PL Scout <span class='season'>25/26</span>", unsafe_allow_html=True)

groups, photos = load_data()
all_df = pd.concat([df for df, _, _ in groups.values()]).reset_index(drop=True)
team_of = dict(zip(all_df["player_name"], all_df["team_name"]))
all_players = sorted(team_of.keys())


def label(n):
    return f"{n} ({abbr(team_of[n])})"


unmapped = sorted(set(all_df["team_name"]) - set(TEAM_ABBR))
if unmapped:
    st.warning(f"No abbreviation set for: {', '.join(unmapped)}. Add them to TEAM_ABBR in app.py.")

if not photos:
    st.info("Player photos not found. Run src/build_photos.py to add them. Initials are shown instead.")

view = st.radio("View", ["Player matching", "Player comparison", "Rankings"],
                horizontal=True, label_visibility="collapsed")

if view == "Player matching":
    name = st.selectbox("Choose a player", all_players, format_func=label)

    left, right = st.columns([3, 1.2], gap="large")

    with right:
        st.subheader("Settings")
        top_n = st.slider("Number of matches", 3, 10, 5)
        st.caption("How many similar players to list.")
        w_role = st.slider("Same-role weight", 0.0, 3.0, 1.0, 0.5)
        st.caption("Higher values push players from a different style group further down. 0 ignores role.")
        w_qual = st.slider("Rating-match weight", 0.0, 3.0, 1.0, 0.5)
        st.caption("Higher values favour players whose season rating is close to the selected player's. 0 ignores rating.")
        min_min = st.slider("Minimum minutes", 900, 3000, 900, 100)
        st.caption("Only list players who have played at least this many minutes.")
        min_rat = st.slider("Minimum rating", 6.0, 8.0, 6.0, 0.1)
        st.caption("Only list players with at least this season rating.")
        excl = st.checkbox("Exclude the selected player's club")
        st.caption("Hide players from the same club, to find alternatives elsewhere.")

    g, i, out = similar(groups, name, top_n, w_role, w_qual, min_min, min_rat, excl)
    df, X, F = groups[g]
    row = df.iloc[i]

    with left:
        c_photo, c_info = st.columns([1, 2])
        with c_photo:
            player_photo(name, photos)
        with c_info:
            st.subheader(name)
            st.caption(f"{row['team_name']} ({abbr(row['team_name'])}) · {POS_NAMES[g]}")
            m1, m2 = st.columns(2)
            m1.metric("Rating", f"{row['rating']:.2f}")
            m2.metric("Minutes", f"{int(row['minutesPlayed']):,}")
            m3, m4 = st.columns(2)
            m3.metric("Goals", num(row["goals"]))
            m4.metric("Assists", num(row["assists"]))
            st.metric("PL Scout Ranking", f"#{row['ps_rank']} of {row['ps_n']}")
            st.caption(f"Rank among all {POS_NAMES[g].lower()}s, based on this position's stats.")

        st.subheader(f"Most similar {POS_NAMES[g].lower()}s")
        if out.empty:
            st.info("No players match these filters. Lower the minimum minutes or rating, or untick the club filter.")
        else:
            show = out.drop(columns=["idx"]).copy()
            show.insert(1, "Club", show["Team"].map(abbr))
            show = show.drop(columns=["Team"])
            st.dataframe(
                show,
                column_config={
                    "Rating": st.column_config.NumberColumn(format="%.2f"),
                    "Style gap": st.column_config.NumberColumn(format="%.2f"),
                    "Score": st.column_config.NumberColumn(format="%.2f"),
                },
                hide_index=True,
                width="stretch",
            )

            with st.expander("What each column means"):
                st.markdown(
                    """
**Player** and **Club:** the name and club code (for example MCI is Manchester City).

**Rating:** the average match rating over the season. Higher is better, and most players sit around 6 to 7.5. *Example: Player X rated 7.2 and Player Y rated 6.8, so X had the stronger season by this measure.*

**Minutes:** total minutes played this season. *Example: Player X played 2,800 minutes and Player Y played 1,200, so X has a bigger sample behind their numbers.*

**Style gap:** how different the two players' stats are. Smaller means a closer style. *Example: if Player X is 3.6 away from the selected player and Player Y is 4.2 away, X plays more like them. Compare gaps only within this list.*

**Score:** the ranking number. It adds the style gap, a penalty for a different role, and the rating gap. Smaller means a better match. *Example: a score of 4.7 ranks above a score of 5.9, so the first player is the closer match.*

**PL Scout Ranking:** the player's position among all players in their group, based on their stats for the season. *Example: Player X is #1 of 55 and Player Y is #20 of 55, so X has the stronger profile for that position.*
                    """
                )

            rival = st.selectbox("Compare with a player from the list", out["Player"].tolist(), format_func=label)
            j = df.index[df["player_name"] == rival][0]

            st.markdown(f"#### {name} vs {rival}")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                stats = [s for s in RADAR_STATS[g] if s in F.columns]
                show_radar(radar_figure(X, stats, i, j, name, rival))
                st.caption("Each corner is one stat. The ring is the position average.")
            with c2:
                show_stat_table(stat_table(F, i, F, j, stats, name, rival), name, rival)

elif view == "Player comparison":
    st.subheader("Player comparison")
    sel_a, mid, sel_b = st.columns([1, 1.6, 1], gap="medium")

    with sel_a:
        name_a = st.selectbox("Player A", all_players,
                              index=all_players.index("Bukayo Saka") if "Bukayo Saka" in all_players else 0,
                              format_func=label)
        player_card(name_a, groups, photos)

    with sel_b:
        name_b = st.selectbox("Player B", all_players,
                              index=all_players.index("Cole Palmer") if "Cole Palmer" in all_players else 1,
                              format_func=label)
        player_card(name_b, groups, photos)

    ga, ia = locate(groups, name_a)
    gb, ib = locate(groups, name_b)
    Fa, Fb = groups[ga][2], groups[gb][2]
    Xa = groups[ga][1]
    common = [c for c in CANDIDATE_STATS if c in Fa.columns and c in Fb.columns]

    with mid:
        if name_a == name_b:
            st.info("Choose two different players to compare.")
        elif ga == gb:
            stats = [s for s in RADAR_STATS[ga] if s in Fa.columns]
            show_radar(radar_figure(Xa, stats, ia, ib, name_a, name_b))
            st.caption("Each corner is one stat. The ring is the position average.")
        else:
            st.info("These players are in different position groups, so the radar chart is not shown. "
                    "The table below compares the stats both players have.")

    if name_a != name_b:
        show_stat_table(stat_table(Fa, ia, Fb, ib, common, name_a, name_b), name_a, name_b)

else:
    st.subheader("Rankings")
    st.markdown(
        """
Each player gets a **PL Scout Ranking** within their position group. Forwards are ranked against forwards, midfielders against midfielders, and defenders against defenders.

The ranking is built from their stats for the season. Each stat is compared with the other players in the same group. A player who is well above average in a stat gets a higher score for it. Some stats count for more than others. For example, goals count a lot for forwards, and tackles and recoveries count a lot for defenders.

The players are then ordered from highest score (#1) to lowest.

**Important:** this is a stats-based guide, not a verdict on who is the best player. It doesn't see position, role, team tactics, or minutes played beyond the cut-off. Use it to start a conversation, then watch the players.
        """
    )

    for g in ["F", "M", "D"]:
        df = groups[g][0]
        ranked = (df.sort_values("ps_rank")[["ps_rank", "player_name", "team_name", "rating"]]
                  .rename(columns={"ps_rank": "Rank", "player_name": "Player",
                                   "team_name": "Team", "rating": "Rating"}))
        ranked["Team"] = ranked["Team"].map(abbr)

        st.markdown(f"#### {POS_NAMES[g]}s ({len(ranked)} players)")
        st.dataframe(
            ranked,
            column_config={"Rating": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True,
            width="stretch",
            height=min(35 * len(ranked) + 40, 600),
        )

floating_chat(groups)