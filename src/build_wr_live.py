from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
LIVE.mkdir(parents=True, exist_ok=True)

STAT_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SCHED_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv"

TEAM_ALIASES = {"LA":"LA","LAR":"LA","JAC":"JAX"}


def read_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def normalize_team(s):
    return s.map(lambda x: TEAM_ALIASES.get(x, x) if pd.notna(x) else x)


def load_player_stats(season: int) -> pd.DataFrame:
    df = read_csv_url(STAT_URL.format(season=season))
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")].copy()
    if "recent_team" in df.columns:
        df["team"] = normalize_team(df["recent_team"])
    elif "team" in df.columns:
        df["team"] = normalize_team(df["team"])
    else:
        raise RuntimeError("No team column found")
    if "player_display_name" in df.columns:
        df["player"] = df["player_display_name"]
    elif "player_name" in df.columns:
        df["player"] = df["player_name"]
    else:
        raise RuntimeError("No player name column found")
    return df


def choose_baseline() -> tuple[int, pd.DataFrame, str]:
    # Use current 2026 data as soon as a completed regular-season sample exists.
    try:
        cur = load_player_stats(2026)
        if len(cur) and float(cur.get("targets", pd.Series(dtype=float)).fillna(0).sum()) > 0:
            return 2026, cur, "CURRENT_2026"
    except Exception:
        pass
    return 2025, load_player_stats(2025), "HISTORICAL_2025_BASELINE"


def build_summary(season: int, df: pd.DataFrame, status: str) -> pd.DataFrame:
    needed = ["targets", "receptions", "receiving_yards", "receiving_tds"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Team target denominator includes all receiving positions so WR target share is true team target share.
    team_week = df.groupby(["team", "week"], as_index=False)["targets"].sum().rename(columns={"targets":"team_targets"})
    df = df.merge(team_week, on=["team", "week"], how="left")
    df["weekly_target_share"] = (df["targets"] / df["team_targets"].replace(0, pd.NA)).fillna(0)

    pos_col = "position" if "position" in df.columns else None
    if pos_col:
        wr = df[df[pos_col].eq("WR")].copy()
    else:
        raise RuntimeError("position column required to isolate WRs")

    team_season_targets = df.groupby("team", as_index=False)["targets"].sum().rename(columns={"targets":"team_season_targets"})
    g = wr.groupby(["team", "player"], as_index=False).agg(
        games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("receiving_tds", "sum"),
        target_share_weekly_mean=("weekly_target_share", "mean"),
        target_share_weekly_median=("weekly_target_share", "median"),
        receiving_yards_median=("receiving_yards", "median"),
        receiving_yards_std=("receiving_yards", "std"),
    )
    td_games = wr.assign(td_hit=(wr["receiving_tds"] > 0).astype(int)).groupby(["team", "player"], as_index=False)["td_hit"].sum().rename(columns={"td_hit":"games_with_td"})
    g = g.merge(td_games, on=["team", "player"], how="left").merge(team_season_targets, on="team", how="left")
    g["target_share"] = (g["targets"] / g["team_season_targets"].replace(0, pd.NA)).fillna(0)
    g["yards_per_game"] = g["receiving_yards"] / g["games"].replace(0, pd.NA)
    g["td_game_rate"] = g["games_with_td"] / g["games"].replace(0, pd.NA)
    g["catch_rate"] = g["receptions"] / g["targets"].replace(0, pd.NA)
    g["receiving_yards_std"] = g["receiving_yards_std"].fillna(0)

    # Role = top 3 WRs on team by true season target share. This is historical/current usage role, not a coaching depth-chart claim.
    g["target_rank_team"] = g.groupby("team")["target_share"].rank(method="first", ascending=False).astype(int)
    g["yards_rank_team"] = g.groupby("team")["receiving_yards"].rank(method="first", ascending=False).astype(int)
    g["td_consistency_rank_team"] = g.sort_values(["team","td_game_rate","receiving_tds","targets"], ascending=[True,False,False,False]).groupby("team").cumcount() + 1
    top = g[g["target_rank_team"] <= 3].copy()
    top["usage_role"] = top["target_rank_team"].map({1:"WR1",2:"WR2",3:"WR3"})
    top["stat_season"] = season
    top["data_status"] = status
    cols = [
        "team","usage_role","player","games","targets","target_share","target_share_weekly_mean","target_share_weekly_median",
        "receptions","catch_rate","receiving_yards","yards_per_game","receiving_yards_median","receiving_yards_std","yards_rank_team",
        "receiving_tds","games_with_td","td_game_rate","td_consistency_rank_team","stat_season","data_status"
    ]
    return top[cols].sort_values(["team","target_rank_team" if "target_rank_team" in top.columns else "usage_role"])


def write_csv(df: pd.DataFrame, name: str):
    df.to_csv(LIVE / name, index=False)


def build_matchups():
    try:
        sched = read_csv_url(SCHED_URL)
        sched = sched[(sched["season"] == 2026) & (sched["game_type"] == "REG")].copy()
        sched["home_team"] = normalize_team(sched["home_team"])
        sched["away_team"] = normalize_team(sched["away_team"])
        out=[]
        for _,r in sched.iterrows():
            out.append([r["week"],r["away_team"],r["home_team"],"A",r.get("gameday","")])
            out.append([r["week"],r["home_team"],r["away_team"],"H",r.get("gameday","")])
        pd.DataFrame(out, columns=["week","team","opponent","site","gameday"]).to_csv(LIVE/"schedule_2026.csv",index=False)
    except Exception as e:
        pd.DataFrame(columns=["week","team","opponent","site","gameday"]).to_csv(LIVE/"schedule_2026.csv",index=False)


def main():
    season, df, status = choose_baseline()
    summary = build_summary(season, df, status)
    write_csv(summary, "wr_top3_summary.csv")

    # Ranked global leaderboards for the live dashboard.
    write_csv(summary.sort_values(["target_share","targets"], ascending=False), "target_share_leaderboard.csv")
    write_csv(summary.sort_values(["td_game_rate","receiving_tds","targets"], ascending=False), "td_consistency_leaderboard.csv")
    write_csv(summary.sort_values(["receiving_yards","yards_per_game"], ascending=False), "receiving_yards_leaderboard.csv")

    # Sharp is deliberately a separate verified-input layer. Weekly automation updates these values.
    sharp_path = ROOT / "config" / "sharp_pass_def_2026.csv"
    if sharp_path.exists():
        pd.read_csv(sharp_path).to_csv(LIVE / "sharp_pass_def_2026.csv", index=False)

    build_matchups()
    pd.DataFrame([
        ["stat_season",season],
        ["data_status",status],
        ["teams_with_top3",summary["team"].nunique()],
        ["wr_rows",len(summary)],
        ["sharp_metric","Pass Efficiency DEF"],
        ["sharp_rule","Never guess; preserve last verified value and flag stale"],
    ], columns=["key","value"]).to_csv(LIVE / "model_status.csv", index=False)
    print(f"PASS: {summary['team'].nunique()} teams / {len(summary)} WR top-3 rows / source season {season}")

if __name__ == "__main__":
    main()
