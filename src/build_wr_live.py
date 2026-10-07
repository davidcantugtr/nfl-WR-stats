from __future__ import annotations

from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

SEASON = 2026
MIN_SNAP_SHARE = 0.20
ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
ARCHIVE = ROOT / "data" / "archive"
STAT_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
SCHED_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv",
]
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX"}
GENERATED_FILES = [
    "current_week_model.csv", "model_status.csv", "primary_wr_dashboard.csv",
    "receiving_yards_leaderboard.csv", "schedule_2026.csv", "target_share_leaderboard.csv",
    "td_consistency_leaderboard.csv", "wr_top3_summary.csv", "wr_weekly_actuals_2026.csv",
]


def read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url, low_memory=False)


def normalize_team(series: pd.Series) -> pd.Series:
    return series.map(lambda value: TEAM_ALIASES.get(value, value) if pd.notna(value) else value)


def clean_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", str(value).lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text).strip()


def get_target_week() -> int:
    """Advance only after the latest week has broad team coverage."""
    stats = read_csv(STAT_URL.format(season=SEASON))
    if "season_type" in stats:
        stats = stats.loc[stats["season_type"].eq("REG")].copy()
    team_col = "recent_team" if "recent_team" in stats else "team"
    stats["week_num"] = pd.to_numeric(stats["week"], errors="coerce")
    coverage = stats.dropna(subset=["week_num"]).groupby("week_num")[team_col].nunique()
    completed = coverage.loc[coverage.ge(24)]
    if completed.empty:
        raise RuntimeError("No broadly completed 2026 week is available")
    return int(completed.index.max()) + 1


def archive_live(week: int) -> None:
    destination = ARCHIVE / f"week_{week}" / "pre_refresh"
    destination.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_FILES:
        source = LIVE / name
        target = destination / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def load_stats() -> pd.DataFrame:
    stats = read_csv(STAT_URL.format(season=SEASON))
    if "season_type" in stats:
        stats = stats.loc[stats["season_type"].eq("REG")].copy()
    if stats.empty:
        raise RuntimeError(f"The {SEASON} player feed is empty; refusing a historical fallback")
    team_col = "recent_team" if "recent_team" in stats else "team"
    name_col = "player_display_name" if "player_display_name" in stats else "player_name"
    stats["team"] = normalize_team(stats[team_col])
    stats["player"] = stats[name_col]
    stats["player_key"] = stats["player"].map(clean_name)
    return stats


def load_snaps() -> pd.DataFrame:
    snaps = read_csv(SNAP_URL.format(season=SEASON))
    if "game_type" in snaps:
        snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    snaps["team"] = normalize_team(snaps["team"])
    snaps["player_key"] = snaps["player"].map(clean_name)
    for column in ["offense_snaps", "offense_pct"]:
        snaps[column] = pd.to_numeric(snaps[column], errors="coerce").fillna(0)
    return snaps


def build_summary(stats: pd.DataFrame, snaps: pd.DataFrame, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = stats.loc[pd.to_numeric(stats["week"], errors="coerce").lt(week)].copy()
    if completed.empty:
        raise RuntimeError(f"No completed player statistics exist before target Week {week}")
    for column in ["targets", "receptions", "receiving_yards", "receiving_tds"]:
        completed[column] = pd.to_numeric(completed.get(column, 0), errors="coerce").fillna(0)
    snap_history = snaps.loc[pd.to_numeric(snaps["week"], errors="coerce").lt(week), [
        "week", "team", "player_key", "offense_snaps", "offense_pct"
    ]].copy()
    snap_history["team_off_snaps_est"] = np.where(
        snap_history["offense_pct"].gt(0),
        snap_history["offense_snaps"] / snap_history["offense_pct"],
        np.nan,
    )
    team_week_snaps = snap_history.groupby(["week", "team"], as_index=False)[
        "team_off_snaps_est"
    ].median().rename(columns={"team_off_snaps_est": "team_off_snaps"})
    snap_history = snap_history.merge(team_week_snaps, on=["week", "team"], how="left")
    completed = completed.merge(snap_history, on=["week", "team", "player_key"], how="left")
    completed[["offense_snaps", "offense_pct", "team_off_snaps"]] = completed[
        ["offense_snaps", "offense_pct", "team_off_snaps"]
    ].fillna(0)
    team_week = completed.groupby(["team", "week"], as_index=False)["targets"].sum().rename(
        columns={"targets": "team_targets"}
    )
    completed = completed.merge(team_week, on=["team", "week"], how="left")
    completed["weekly_target_share"] = (
        completed["targets"] / completed["team_targets"].replace(0, pd.NA)
    ).fillna(0)
    actual_columns = [
        "week", "game_id", "team", "opponent_team", "player", "player_key", "position",
        "targets", "receptions", "receiving_yards", "receiving_tds", "weekly_target_share",
        "offense_snaps", "offense_pct",
    ]
    weekly_actuals = completed[actual_columns].sort_values(
        ["week", "team", "targets"], ascending=[True, True, False]
    )
    wr = completed.loc[completed["position"].eq("WR")].copy()
    latest = wr.sort_values("week").groupby(["team", "player_key"], as_index=False).tail(1)[
        ["team", "player_key", "offense_pct"]
    ].rename(columns={"offense_pct": "latest_snap_share"})
    team_targets = completed.groupby("team", as_index=False)["targets"].sum().rename(
        columns={"targets": "team_season_targets"}
    )
    grouped = wr.groupby(["team", "player_key", "player"], as_index=False).agg(
        games=("week", "nunique"), targets=("targets", "sum"), receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"), receiving_tds=("receiving_tds", "sum"),
        target_share_weekly_mean=("weekly_target_share", "mean"),
        target_share_weekly_median=("weekly_target_share", "median"),
        receiving_yards_median=("receiving_yards", "median"),
        receiving_yards_std=("receiving_yards", "std"), offense_snaps=("offense_snaps", "sum"),
        team_off_snaps=("team_off_snaps", "sum"), snap_share_mean=("offense_pct", "mean"),
        max_snap_share=("offense_pct", "max"),
    )
    td_games = wr.assign(td_hit=wr["receiving_tds"].gt(0).astype(int)).groupby(
        ["team", "player_key"], as_index=False
    )["td_hit"].sum().rename(columns={"td_hit": "games_with_td"})
    grouped = grouped.merge(td_games, on=["team", "player_key"], how="left")
    grouped = grouped.merge(team_targets, on="team", how="left").merge(
        latest, on=["team", "player_key"], how="left"
    )
    grouped["target_share"] = (
        grouped["targets"] / grouped["team_season_targets"].replace(0, pd.NA)
    ).fillna(0)
    grouped["yards_per_game"] = grouped["receiving_yards"] / grouped["games"].replace(0, pd.NA)
    grouped["td_game_rate"] = grouped["games_with_td"] / grouped["games"].replace(0, pd.NA)
    grouped["catch_rate"] = grouped["receptions"] / grouped["targets"].replace(0, pd.NA)
    grouped["receiving_yards_std"] = grouped["receiving_yards_std"].fillna(0)
    grouped["s2d_snap_share"] = (
        grouped["offense_snaps"] / grouped["team_off_snaps"].replace(0, pd.NA)
    ).fillna(0)
    # Strict rule: exactly 20.0% remains suppressed; target share never overrides.
    grouped["snap_eligible"] = grouped["s2d_snap_share"].gt(MIN_SNAP_SHARE)
    eligible = grouped.loc[grouped["snap_eligible"]].copy()
    eligible["target_rank_team"] = eligible.groupby("team")["target_share"].rank(
        method="first", ascending=False
    ).astype(int)
    eligible["yards_rank_team"] = eligible.groupby("team")["receiving_yards"].rank(
        method="first", ascending=False
    ).astype(int)
    eligible["td_consistency_rank_team"] = eligible.sort_values(
        ["team", "td_game_rate", "receiving_tds", "targets"], ascending=[True, False, False, False]
    ).groupby("team").cumcount() + 1
    top = eligible.loc[eligible["target_rank_team"].le(3)].copy().sort_values(
        ["team", "target_rank_team"]
    )
    top["usage_role"] = top["target_rank_team"].map({1: "WR1", 2: "WR2", 3: "WR3"})
    top["stat_season"] = SEASON
    top["data_status"] = f"CURRENT_{SEASON}_THROUGH_WEEK_{week - 1}"
    columns = [
        "team", "usage_role", "player", "games", "targets", "target_share",
        "target_share_weekly_mean", "target_share_weekly_median", "receptions", "catch_rate",
        "receiving_yards", "yards_per_game", "receiving_yards_median", "receiving_yards_std",
        "yards_rank_team", "receiving_tds", "games_with_td", "td_game_rate",
        "td_consistency_rank_team", "offense_snaps", "team_off_snaps", "snap_share_mean",
        "latest_snap_share", "s2d_snap_share", "snap_eligible", "stat_season", "data_status",
    ]
    return top[columns], weekly_actuals


def build_schedule() -> tuple[int, str]:
    last_error: Exception | None = None
    for url in SCHED_URLS:
        try:
            schedule = read_csv(url)
            schedule = schedule.loc[pd.to_numeric(schedule["season"], errors="coerce").eq(SEASON)].copy()
            if "game_type" in schedule:
                schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
            schedule["home_team"] = normalize_team(schedule["home_team"])
            schedule["away_team"] = normalize_team(schedule["away_team"])
            date_col = "gameday" if "gameday" in schedule else "game_date"
            rows = []
            for _, game in schedule.iterrows():
                rows.append([game["week"], game["away_team"], game["home_team"], "A", game.get(date_col, "")])
                rows.append([game["week"], game["home_team"], game["away_team"], "H", game.get(date_col, "")])
            result = pd.DataFrame(rows, columns=["week", "team", "opponent", "site", "gameday"])
            if result["team"].nunique() != 32:
                raise RuntimeError(f"schedule coverage has {result['team'].nunique()} teams")
            result.to_csv(LIVE / "schedule_2026.csv", index=False)
            return len(result), url
        except Exception as exc:
            last_error = exc
    existing_path = LIVE / "schedule_2026.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        required = {"week", "team", "opponent", "site"}
        if required.issubset(existing.columns) and existing["team"].nunique() == 32:
            return len(existing), "validated repository schedule fallback; remote release unavailable"
    raise RuntimeError(f"schedule ingestion failed: {last_error}")


def enrich_current(summary: pd.DataFrame, week: int) -> pd.DataFrame:
    current = pd.read_csv(LIVE / "current_week_model.csv")
    generated = {
        "player_key", "official_targets_through_prior_week",
        "official_receptions_through_prior_week", "official_rec_yds_through_prior_week",
        "yards_per_game", "target_share", "offense_snaps", "team_off_snaps",
        "latest_snap_share", "s2d_snap_share", "snap_eligible",
        "model_rec_yds_projection", "model_floor", "model_ceiling",
        "model_eligible", "history_source",
    }
    current = current.drop(columns=[c for c in generated if c in current.columns])
    current["week"] = week
    schedule = pd.read_csv(LIVE / "schedule_2026.csv")
    selected = schedule.loc[pd.to_numeric(schedule["week"], errors="coerce").eq(week), [
        "team", "opponent", "site"
    ]]
    current = current.drop(columns=["opponent", "site"], errors="ignore").merge(
        selected, on="team", how="left"
    )
    current["player_key"] = current["player"].map(clean_name)
    usage = summary[[
        "team", "player", "targets", "receptions", "receiving_yards", "yards_per_game",
        "target_share", "offense_snaps", "team_off_snaps", "latest_snap_share",
        "s2d_snap_share", "snap_eligible",
    ]].copy()
    usage["player_key"] = usage["player"].map(clean_name)
    usage = usage.drop(columns="player").rename(columns={
        "targets": "official_targets_through_prior_week",
        "receptions": "official_receptions_through_prior_week",
        "receiving_yards": "official_rec_yds_through_prior_week",
    })
    current = current.merge(usage, on=["team", "player_key"], how="left")
    history = pd.to_numeric(current["yards_per_game"], errors="coerce")
    current["model_rec_yds_projection"] = history
    current["model_floor"] = (current["model_rec_yds_projection"] * 0.62).clip(lower=0)
    current["model_ceiling"] = current["model_rec_yds_projection"] * 1.38
    current["status"] = "PENDING"
    current["availability"] = f"PENDING WEEK {week} OFFICIAL REPORT"
    current["projection_source"] = (
        f"Official 2026 full-game usage through Week {week - 1}; Week {week} injury report pending"
    )
    current["model_eligible"] = current["snap_eligible"].fillna(False)
    current["history_source"] = f"nflverse weekly stats and snap counts through Week {week - 1}"
    current["refresh_version"] = "v0.7-current-season-integrity"
    current.to_csv(LIVE / "current_week_model.csv", index=False)
    current.loc[current["model_eligible"]].sort_values(
        ["model_rec_yds_projection", "rank"], ascending=[False, True]
    ).to_csv(LIVE / "primary_wr_dashboard.csv", index=False)
    return current


def main() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    week = get_target_week()
    archive_live(week)
    summary, weekly_actuals = build_summary(load_stats(), load_snaps(), week)
    if summary["team"].nunique() != 32:
        raise AssertionError(f"Expected 32 teams after snap gate, got {summary['team'].nunique()}")
    summary.to_csv(LIVE / "wr_top3_summary.csv", index=False)
    weekly_actuals.to_csv(LIVE / "wr_weekly_actuals_2026.csv", index=False)
    summary.sort_values(["target_share", "targets"], ascending=False).to_csv(
        LIVE / "target_share_leaderboard.csv", index=False
    )
    summary.sort_values(["td_game_rate", "receiving_tds", "targets"], ascending=False).to_csv(
        LIVE / "td_consistency_leaderboard.csv", index=False
    )
    summary.sort_values(["receiving_yards", "yards_per_game"], ascending=False).to_csv(
        LIVE / "receiving_yards_leaderboard.csv", index=False
    )
    sharp_path = ROOT / "config" / "sharp_pass_def_2026.csv"
    if sharp_path.exists():
        pd.read_csv(sharp_path).to_csv(LIVE / "sharp_pass_def_2026.csv", index=False)
    schedule_rows, schedule_source = build_schedule()
    current = enrich_current(summary, week)
    pd.DataFrame([
        ["stat_season", SEASON], ["current_target_week", week],
        ["data_status", f"CURRENT_{SEASON}_THROUGH_WEEK_{week - 1}"],
        ["teams_with_top3", summary["team"].nunique()], ["wr_rows", len(summary)],
        ["schedule_rows", schedule_rows], ["schedule_source", schedule_source],
        ["snap_source", SNAP_URL.format(season=SEASON)],
        ["snap_gate", f"strict S2D offensive snap share > {MIN_SNAP_SHARE:.0%}; exactly 20% suppressed"],
        ["partial_week_rule", f"Week {week} actuals excluded from Week {week} projections"],
        ["validation_status", f"32 teams; current-season data through Week {week - 1}; history archived"],
    ], columns=["key", "value"]).to_csv(LIVE / "model_status.csv", index=False)
    if not current["week"].eq(week).all():
        raise AssertionError("Current week model contains a stale week")
    print(f"PASS: 32 teams / {len(summary)} WR1-3 rows")
    print(f"PASS: {SEASON} stats and snaps through Week {week - 1}; partial Week {week} excluded")
    print(f"PASS: {schedule_rows} schedule rows / target Week {week}")


if __name__ == "__main__":
    main()
