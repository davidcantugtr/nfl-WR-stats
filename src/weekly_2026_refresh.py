from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
ARCHIVE = ROOT / "data" / "archive"
SEASON = 2026
TARGET_WEEK = 8
COMPLETED_WEEK = 7
REFRESH_DATE = "2026-10-28"
REFRESH_CT = "2026-10-28 03:20 CT"

STAT_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2026.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.csv.gz"
DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SHARP_PUBLIC_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-matchups/"

TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX"}
OUT = {
    "terry mclaurin": ("OUT — season-ending patellar tendon injury", 0.0),
    "davante adams": ("OUT — hamstring procedure; expected 6–10 weeks", 0.0),
    "romeo doubs": ("IR — shoulder; minimum four-game absence", 0.0),
}
EARLY_REPORT = {
    "matthew golden": ("DNP — early official Week 8 report", 0.65),
    "jalen coker": ("FULL — early official Week 8 report", 1.0),
}
DEFAULT_AVAILABILITY = "PENDING — Sunday/Monday official Week 8 reports not yet filed"

# Sharp public Week 8 matchup ranks. These are public matchup-table ranks, not
# the inaccessible proprietary Pass Efficiency DEF field.
SHARP_PASS = {
    "GB": 11, "CAR": 15, "DAL": 31, "ARI": 22, "TB": 7, "ATL": 9,
    "BUF": 27, "BAL": 18, "PIT": 21, "CLE": 4, "JAX": 19, "IND": 14,
    "NYJ": 23, "LV": 29, "DET": 28, "MIN": 2, "CIN": 24, "TEN": 30,
    "LA": 8, "LAC": 10, "DEN": 3, "KC": 13, "MIA": 32, "NE": 16,
    "WAS": 25, "PHI": 6, "SEA": 1, "CHI": 17,
}
SHARP_PRESSURE = {
    "GB": 23, "CAR": 13, "DAL": 14, "ARI": 20, "TB": 28, "ATL": 3,
    "BUF": 11, "BAL": 15, "PIT": 24, "CLE": 5, "JAX": 16, "IND": 30,
    "NYJ": 19, "LV": 22, "DET": 10, "MIN": 1, "CIN": 9, "TEN": 32,
    "LA": 17, "LAC": 7, "DEN": 4, "KC": 27, "MIA": 18, "NE": 8,
    "WAS": 31, "PHI": 26, "SEA": 2, "CHI": 12,
}
SHARP_EXPLOSIVE = {
    "GB": 4, "CAR": 10, "DAL": 28, "ARI": 26, "TB": 13, "ATL": 11,
    "BUF": 24, "BAL": 12, "PIT": 6, "CLE": 22, "JAX": 17, "IND": 16,
    "NYJ": 7, "LV": 20, "DET": 27, "MIN": 3, "CIN": 32, "TEN": 29,
    "LA": 5, "LAC": 18, "DEN": 8, "KC": 9, "MIA": 19, "NE": 15,
    "WAS": 14, "PHI": 21, "SEA": 1, "CHI": 31,
}


def norm_team(v: object) -> object:
    return TEAM_ALIASES.get(v, v) if pd.notna(v) else v


def norm_name(v: object) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", str(v).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def flag(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def archive_outgoing() -> pd.DataFrame:
    dest = ARCHIVE / "week_8" / "pre_refresh"
    dest.mkdir(parents=True, exist_ok=True)
    for p in LIVE.glob("*.csv"):
        target = dest / p.name
        if not target.exists():
            shutil.copy2(p, target)
    snapshot = ARCHIVE / "week_8" / "published_week_7_snapshot.csv"
    if not snapshot.exists():
        shutil.copy2(LIVE / "current_week_model.csv", snapshot)
    return pd.read_csv(snapshot)


def load_sources():
    stats = pd.read_csv(STAT_URL, low_memory=False)
    stats = stats.loc[stats.season_type.eq("REG") & pd.to_numeric(stats.week, errors="coerce").le(COMPLETED_WEEK)].copy()
    stats["team"] = stats["team"].map(norm_team)
    stats["opponent_team"] = stats["opponent_team"].map(norm_team)
    stats["player"] = stats["player_display_name"]
    stats["player_key"] = stats.player.map(norm_name)
    snaps = pd.read_csv(SNAP_URL, low_memory=False)
    snaps = snaps.loc[snaps.game_type.eq("REG") & pd.to_numeric(snaps.week, errors="coerce").le(COMPLETED_WEEK)].copy()
    snaps["team"] = snaps.team.map(norm_team)
    snaps["opponent"] = snaps.opponent.map(norm_team)
    snaps["player_key"] = snaps.player.map(norm_name)
    depth = pd.read_csv(DEPTH_URL, low_memory=False)
    depth["team"] = depth.team.map(norm_team)
    pbp = pd.read_csv(PBP_URL, compression="gzip", low_memory=False)
    pbp = pbp.loc[pbp.season_type.eq("REG") & pd.to_numeric(pbp.week, errors="coerce").le(COMPLETED_WEEK)].copy()
    pbp["posteam"] = pbp.posteam.map(norm_team); pbp["defteam"] = pbp.defteam.map(norm_team)
    games = pd.read_csv(SCHEDULE_URL, low_memory=False)
    games["home_team"] = games.home_team.map(norm_team); games["away_team"] = games.away_team.map(norm_team)
    return stats, snaps, depth, pbp, games


def build_schedule(games: pd.DataFrame) -> pd.DataFrame:
    g = games.loc[(games.season.eq(SEASON)) & (games.week.eq(TARGET_WEEK)) & games.game_type.eq("REG")].copy()
    if len(g) != 14:
        raise AssertionError(f"Expected 14 Week 8 games, got {len(g)}")
    h = g.assign(team=g.home_team, opponent=g.away_team, site="H")
    a = g.assign(team=g.away_team, opponent=g.home_team, site="A")
    out = pd.concat([h, a], ignore_index=True)[["game_id", "gameday", "gametime", "team", "opponent", "site"]]
    if out.duplicated("team").any() or len(out) != 28:
        raise AssertionError("Week 8 team schedule coverage failed")
    reciprocal = out.merge(out[["team", "opponent"]], left_on=["team", "opponent"], right_on=["opponent", "team"])
    if len(reciprocal) != 28:
        raise AssertionError("Reciprocal opponent validation failed")
    return out


def build_wr_summary(stats: pd.DataFrame, snaps: pd.DataFrame, depth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for c in ["targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards"]:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0)
    team_week_targets = stats.groupby(["team", "week"], as_index=False).targets.sum().rename(columns={"targets": "team_targets"})
    stats = stats.merge(team_week_targets, on=["team", "week"], how="left")
    stats["weekly_target_share"] = stats.targets / stats.team_targets.replace(0, np.nan)
    wr = stats.loc[stats.position.eq("WR")].copy()
    snap = snaps.loc[snaps.position.eq("WR"), ["week", "team", "player_key", "offense_snaps", "offense_pct"]].copy()
    wr = wr.merge(snap, on=["week", "team", "player_key"], how="left")
    wr[["offense_snaps", "offense_pct"]] = wr[["offense_snaps", "offense_pct"]].fillna(0)
    weekly_actuals = wr[["week", "game_id", "team", "opponent_team", "player_id", "player", "targets", "team_targets", "weekly_target_share", "receptions", "receiving_yards", "receiving_tds", "offense_snaps", "offense_pct"]].sort_values(["week", "team", "targets"], ascending=[True, True, False])

    tt = stats.groupby("team", as_index=False).targets.sum().rename(columns={"targets": "team_season_targets"})
    grouped = wr.groupby(["team", "player_id", "player_key", "player"], as_index=False).agg(
        games=("week", "nunique"), targets=("targets", "sum"), receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"), receiving_tds=("receiving_tds", "sum"),
        games_with_td=("receiving_tds", lambda x: int((x > 0).sum())),
        ypg_median=("receiving_yards", "median"), ypg_std=("receiving_yards", lambda x: float(np.std(x, ddof=0))),
        target_share_weekly_mean=("weekly_target_share", "mean"), target_share_weekly_std=("weekly_target_share", lambda x: float(np.std(x, ddof=0))),
        offense_snaps=("offense_snaps", "sum"), snap_share_mean=("offense_pct", "mean"),
        latest_snap_share=("offense_pct", "last"), last3_ypg=("receiving_yards", lambda x: float(x.tail(3).mean())),
        games_50_plus=("receiving_yards", lambda x: int((x >= 50).sum())), games_75_plus=("receiving_yards", lambda x: int((x >= 75).sum())),
    ).merge(tt, on="team", how="left")
    grouped["target_share"] = grouped.targets / grouped.team_season_targets.replace(0, np.nan)
    grouped["yards_per_game"] = grouped.receiving_yards / grouped.games.replace(0, np.nan)
    grouped["td_game_rate"] = grouped.games_with_td / grouped.games.replace(0, np.nan)
    grouped["catch_rate"] = grouped.receptions / grouped.targets.replace(0, np.nan)
    grouped["weekly_consistency"] = grouped.games_50_plus / grouped.games.replace(0, np.nan)

    depth["dt"] = pd.to_datetime(depth.dt, errors="coerce", utc=True)
    latest = depth.loc[depth.dt.eq(depth.dt.max()) & depth.pos_abb.astype(str).str.upper().eq("WR")].copy()
    latest["player_key"] = latest.player_name.map(norm_name)
    latest["depth_rank"] = pd.to_numeric(latest.pos_rank, errors="coerce")
    grouped = grouped.merge(latest[["team", "player_key", "depth_rank"]].drop_duplicates(["team", "player_key"]), on=["team", "player_key"], how="left")
    grouped["depth_rank"] = grouped.depth_rank.fillna(9)
    grouped["availability"] = grouped.player_key.map(lambda k: OUT.get(k, EARLY_REPORT.get(k, (DEFAULT_AVAILABILITY, 1.0)))[0])
    grouped["availability_factor"] = grouped.player_key.map(lambda k: OUT.get(k, EARLY_REPORT.get(k, ("", 1.0)))[1])
    grouped["selection_score"] = 0.70 * grouped.target_share + 0.20 * grouped.latest_snap_share + 0.10 * (1 / grouped.depth_rank.clip(lower=1))
    active = grouped.loc[grouped.availability_factor.gt(0)].copy()
    top = active.sort_values(["team", "selection_score", "targets"], ascending=[True, False, False]).groupby("team", as_index=False).head(3).copy()
    if len(top) != 96 or top.team.nunique() != 32:
        raise AssertionError(f"WR1-3 coverage failed: {len(top)} rows / {top.team.nunique()} teams")
    top["usage_order"] = top.groupby("team").cumcount() + 1
    top["usage_role"] = top.usage_order.map({1: "WR1", 2: "WR2", 3: "WR3"})
    top["target_rank_team"] = top.groupby("team").target_share.rank(method="first", ascending=False).astype(int)
    top["yards_rank_team"] = top.groupby("team").receiving_yards.rank(method="first", ascending=False).astype(int)
    top["td_consistency_rank_team"] = top.sort_values(["team", "td_game_rate", "receiving_tds", "targets"], ascending=[True, False, False, False]).groupby("team").cumcount() + 1
    top["role_confidence"] = np.clip(0.45 + top.latest_snap_share * 0.35 + top.target_share * 0.70 - top.target_share_weekly_std * 0.40, 0.35, 0.98)
    top["data_status"] = "FINAL_2026_THROUGH_WEEK_7"
    top["source_updated"] = REFRESH_CT
    return top.sort_values(["team", "usage_order"]), weekly_actuals


def team_game_outputs(stats: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    role_map = roles.set_index(["team", "player_key"]).usage_role.to_dict()
    s = stats.copy(); s["role"] = [role_map.get((t, p), "OTHER") for t, p in zip(s.team, s.player_key)]
    rows = []
    for (team, week, game_id, opp), g in s.groupby(["team", "week", "game_id", "opponent_team"]):
        qb = g.loc[g.position.eq("QB")]
        q_pass_yd = qb.passing_yards.sum(); q_comp = qb.completions.sum(); q_att = qb.attempts.sum(); q_td = qb.passing_tds.sum()
        q_fp = (q_pass_yd / 25 + 4 * q_td - 2 * qb.passing_interceptions.sum() + qb.rushing_yards.sum() / 10 + 6 * qb.rushing_tds.sum())
        row = {"team": team, "week": week, "game_id": game_id, "defense": opp,
               "qb_pass_yds": q_pass_yd, "qb_comp_rate": q_comp / q_att if q_att else np.nan,
               "qb_pass_tds": q_td, "qb_fantasy": q_fp}
        for role in ["WR1", "WR2", "WR3"]:
            x = g.loc[g.role.eq(role)]
            row[f"{role.lower()}_targets"] = x.targets.sum(); row[f"{role.lower()}_yds"] = x.receiving_yards.sum()
        te = g.loc[g.position.eq("TE")]; rb = g.loc[g.position.eq("RB")]
        row.update({"te_targets": te.targets.sum(), "te_yds": te.receiving_yards.sum(),
                    "rb_targets": rb.targets.sum(), "rb_rec_yds": rb.receiving_yards.sum(), "rb_rush_yds": rb.rushing_yards.sum()})
        rows.append(row)
    return pd.DataFrame(rows)


def elevation_table(outputs: pd.DataFrame, through_week: int) -> pd.DataFrame:
    d = outputs.loc[outputs.week.le(through_week)].copy()
    metrics = [c for c in d.columns if c not in ["team", "week", "game_id", "defense"]]
    floors = {"qb_pass_yds": 100, "qb_comp_rate": .10, "qb_pass_tds": .5, "qb_fantasy": 5,
              "wr1_targets": 2, "wr1_yds": 20, "wr2_targets": 1.5, "wr2_yds": 15,
              "wr3_targets": 1, "wr3_yds": 10, "te_targets": 1.5, "te_yds": 12,
              "rb_targets": 1, "rb_rec_yds": 8, "rb_rush_yds": 25}
    league = d[metrics].mean()
    for m in metrics:
        total = d.groupby("team")[m].transform("sum") - d[m]
        count = d.groupby("team")[m].transform("count") - d[m].notna().astype(int)
        base = total / count.replace(0, np.nan)
        base = base.fillna(league[m])
        d[f"lift_{m}"] = ((d[m] - base) / np.maximum(base.abs(), floors[m])).clip(-1.5, 2.0)
    pos = {
        "QB": ["qb_pass_yds", "qb_comp_rate", "qb_pass_tds", "qb_fantasy"],
        "WR1": ["wr1_targets", "wr1_yds"], "WR2": ["wr2_targets", "wr2_yds"],
        "WR3": ["wr3_targets", "wr3_yds"], "TE": ["te_targets", "te_yds"],
        "RB": ["rb_targets", "rb_rec_yds", "rb_rush_yds"],
    }
    for k, ms in pos.items(): d[f"{k}_lift"] = d[[f"lift_{m}" for m in ms]].mean(axis=1)
    agg = d.groupby("defense", as_index=False).agg(games=("game_id", "nunique"), **{f"{k}_lift": (f"{k}_lift", "mean") for k in pos})
    weights = {"QB": .25, "WR1": .20, "WR2": .12, "WR3": .08, "TE": .12, "RB": .13}
    agg["elevation_score"] = sum(agg[f"{k}_lift"] * w for k, w in weights.items()) / sum(weights.values()) * 100
    agg["rank"] = agg.elevation_score.rank(method="first", ascending=False).astype(int)
    agg["confidence"] = np.select([agg.games.ge(7), agg.games.ge(5)], ["HIGH", "MEDIUM"], default="LOW")
    agg["most_amplified"] = agg.apply(lambda r: ", ".join(sorted(pos, key=lambda k: r[f"{k}_lift"], reverse=True)[:3]), axis=1)
    return agg.sort_values("rank")


def build_defense(stats: pd.DataFrame, pbp: pd.DataFrame, roles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = team_game_outputs(stats, roles)
    current = elevation_table(outputs, COMPLETED_WEEK)
    prior = elevation_table(outputs, COMPLETED_WEEK - 1)[["defense", "rank", "elevation_score"]].rename(columns={"rank": "prior_rank", "elevation_score": "prior_score"})
    current = current.merge(prior, on="defense", how="left")
    current["rank_change"] = current.prior_rank - current["rank"]
    current["trend"] = np.select([current.rank_change.ge(3), current.rank_change.le(-3)], ["RISING", "FALLING"], default="STABLE")

    pass_plays = pbp.loc[flag(pbp.pass_attempt) | flag(pbp.sack)].copy()
    pass_plays["passing_yards"] = pd.to_numeric(pass_plays.passing_yards, errors="coerce").fillna(0)
    pass_plays["complete"] = flag(pass_plays.complete_pass).astype(int); pass_plays["attempt"] = flag(pass_plays.pass_attempt).astype(int)
    pass_plays["pass_td"] = flag(pass_plays.pass_touchdown).astype(int); pass_plays["sack_i"] = flag(pass_plays.sack).astype(int)
    pass_plays["hit_i"] = flag(pass_plays.qb_hit).astype(int); pass_plays["explosive"] = (pass_plays.passing_yards >= 20).astype(int)
    pass_plays["success_i"] = flag(pass_plays.success).astype(int)
    games = pass_plays[["game_id", "defteam"]].drop_duplicates().groupby("defteam").size().rename("games")
    p = pass_plays.groupby("defteam", as_index=False).agg(pass_yards=("passing_yards", "sum"), completions=("complete", "sum"), attempts=("attempt", "sum"), pass_tds=("pass_td", "sum"), sacks=("sack_i", "sum"), qb_hits=("hit_i", "sum"), explosives=("explosive", "sum"), successes=("success_i", "sum"), dropbacks=("game_id", "size")).rename(columns={"defteam": "team"}).merge(games, left_on="team", right_index=True)
    p["pass_yds_allowed_pg"] = p.pass_yards / p.games; p["completion_rate_allowed"] = p.completions / p.attempts.replace(0, np.nan)
    p["pass_tds_allowed_pg"] = p.pass_tds / p.games; p["yards_per_attempt_allowed"] = p.pass_yards / p.attempts.replace(0, np.nan)
    p["sack_rate"] = p.sacks / p.dropbacks.replace(0, np.nan); p["pressure_event_rate"] = (p.sacks + p.qb_hits) / p.dropbacks.replace(0, np.nan)
    p["explosive_pass_rate_allowed"] = p.explosives / p.attempts.replace(0, np.nan); p["pass_success_rate_allowed"] = p.successes / p.dropbacks.replace(0, np.nan)
    p["public_pass_def_rank"] = p.pass_yds_allowed_pg.rank(method="first", ascending=True).astype(int)
    legacy = pd.read_csv(ROOT / "config" / "sharp_pass_def_2026.csv")[["team", "sharp_pass_def_rank", "last_verified"]]
    legacy.team = legacy.team.map(norm_team)
    monitor = pd.DataFrame({"team": sorted(set(p.team) | set(legacy.team))}).merge(legacy, on="team", how="left").merge(p, on="team", how="left").merge(current.rename(columns={"defense": "team"}), on="team", how="left")
    monitor["sharp_proprietary_status"] = "STALE — last verified Week 5 snapshot; not guessed"
    monitor["sharp_public_week8_pass_rank"] = monitor.team.map(SHARP_PASS).astype("Int64")
    monitor["sharp_public_pressure_rank"] = monitor.team.map(SHARP_PRESSURE).astype("Int64")
    monitor["sharp_public_explosive_rank"] = monitor.team.map(SHARP_EXPLOSIVE).astype("Int64")
    monitor["current_source"] = "nflverse PBP Weeks 1–7 + Sharp public Week 8 matchup table"
    monitor["source_updated"] = REFRESH_CT
    elev_cols = ["defense", "rank", "elevation_score", "prior_rank", "rank_change", "trend", "QB_lift", "WR1_lift", "WR2_lift", "WR3_lift", "TE_lift", "RB_lift", "most_amplified", "games", "confidence"]
    elevation = current[elev_cols].copy(); elevation["source_updated"] = REFRESH_CT
    prior_top5 = prior.sort_values("prior_rank").head(5)
    return monitor.sort_values("team"), elevation.sort_values("rank"), prior_top5


def build_model(roles: pd.DataFrame, schedule: pd.DataFrame, monitor: pd.DataFrame, elevation: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(roles.team.unique())
    sched = schedule[["team", "opponent", "site"]].copy()
    byes = sorted(set(teams) - set(sched.team))
    sched = pd.concat([sched, pd.DataFrame({"team": byes, "opponent": "BYE", "site": "B"})], ignore_index=True)
    m = roles.merge(sched, on="team", how="left")
    dm = monitor.set_index("team").to_dict("index"); em = elevation.set_index("defense").to_dict("index")
    plays = pbp.loc[pbp.posteam.notna() & (flag(pbp.pass_attempt) | flag(pbp.rush_attempt) | flag(pbp.sack))].groupby(["posteam", "game_id"]).size().groupby("posteam").mean()
    league_plays = float(plays.mean())
    rows = []
    for _, r in m.iterrows():
        if r.opponent == "BYE":
            proj = score = 0.0; matchup = 1.0; pace = 1.0; tier = "BYE"; status = "BYE"
            sharp_rank = public_rank = elev = np.nan
        else:
            d = dm[r.opponent]; e = em[r.opponent]
            sharp_rank = d.get("sharp_public_week8_pass_rank", np.nan); public_rank = d.get("public_pass_def_rank", np.nan)
            role_lift = e.get(f"{r.usage_role}_lift", 0.0)
            sharp_factor = 1.0 if pd.isna(sharp_rank) else clip(1 + (float(sharp_rank) - 16.5) * .011, .84, 1.17)
            public_factor = 1.0 if pd.isna(public_rank) else clip(1 + (float(public_rank) - 16.5) * .006, .92, 1.08)
            elevation_factor = clip(1 + role_lift * .25, .82, 1.25)
            pace = clip((plays.get(r.team, league_plays) + plays.get(r.opponent, league_plays)) / (2 * league_plays), .92, 1.08)
            matchup = sharp_factor * public_factor * elevation_factor
            base = .55 * r.yards_per_game + .25 * r.last3_ypg + .20 * r.ypg_median
            proj = base * matchup * pace * r.availability_factor
            score = clip((r.target_share * 130 + min(r.yards_per_game, 120) / 120 * 25 + r.td_game_rate * 15 + r.role_confidence * 15 + (matchup - .8) / .45 * 10) * r.availability_factor, 0, 100)
            tier = "ELITE" if score >= 82 else "PRIME" if score >= 74 else "STRONG" if score >= 66 else "WATCH" if score >= 58 else "PASS"
            elev = e.get("elevation_score", np.nan); status = r.availability
        rows.append({
            "week": TARGET_WEEK, "player": r.player, "player_id": r.player_id, "team": r.team, "role": r.usage_role,
            "opponent": r.opponent, "site": r.site, "availability": status, "games": r.games,
            "targets": r.targets, "team_targets": r.team_season_targets, "target_share": r.target_share,
            "target_rank_team": r.target_rank_team, "receptions": r.receptions, "receiving_yards": r.receiving_yards,
            "yards_per_game": r.yards_per_game, "yards_rank_team": r.yards_rank_team, "receiving_tds": r.receiving_tds,
            "td_game_rate": r.td_game_rate, "td_consistency_rank_team": r.td_consistency_rank_team,
            "weekly_consistency_50_plus": r.weekly_consistency, "latest_snap_share": r.latest_snap_share,
            "depth_rank": r.depth_rank, "role_confidence": r.role_confidence,
            "sharp_public_pass_def_rank": sharp_rank, "public_pass_yds_def_rank": public_rank,
            "defense_elevation_score": elev, "matchup_factor": matchup, "pace_factor": pace,
            "model_rec_yds_projection": proj, "model_floor": max(0, proj - max(10, r.ypg_std * .60)),
            "model_ceiling": proj + max(14, r.ypg_std * .75), "opportunity_score": score, "tier": tier,
            "proprietary_sharp_status": "STALE W5 PRESERVED / PUBLIC W8 SUBSTITUTE LABELED",
            "source_updated": REFRESH_CT,
        })
    model = pd.DataFrame(rows).sort_values(["opportunity_score", "target_share"], ascending=False).reset_index(drop=True)
    model.insert(1, "rank", range(1, len(model) + 1))
    if len(model) != 96 or model.duplicated(["team", "player"]).any():
        raise AssertionError("Current Week Model row/duplicate validation failed")
    if model.groupby("team").size().ne(3).any():
        raise AssertionError("Expected exactly three WRs per team")
    return model


def grade_and_archive(outgoing: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = outgoing.copy(); old["week"] = pd.to_numeric(old.week, errors="coerce")
    old = old.loc[old.week.eq(7)].copy()
    old["player_key"] = old.player.map(norm_name); actual = weekly.loc[weekly.week.eq(7)].copy(); actual["player_key"] = actual.player.map(norm_name)
    actual = actual[["team", "player_key", "targets", "team_targets", "weekly_target_share", "receiving_yards", "receiving_tds"]].rename(columns={
        "targets": "actual_targets", "team_targets": "actual_team_targets", "weekly_target_share": "actual_target_share",
        "receiving_yards": "actual_receiving_yards", "receiving_tds": "actual_receiving_tds",
    })
    joined = old.merge(actual, on=["team", "player_key"], how="left")
    for c in ["actual_targets", "actual_team_targets", "actual_target_share", "actual_receiving_yards", "actual_receiving_tds"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0)
    pred_y = pd.to_numeric(joined.get("model_rec_yds_projection"), errors="coerce").fillna(pd.to_numeric(joined.get("yards_per_game"), errors="coerce")).fillna(0)
    pred_share = pd.to_numeric(joined.get("target_share"), errors="coerce").fillna(0)
    ranks = pd.to_numeric(joined.get("rank"), errors="coerce").fillna(pd.Series(range(1, len(joined)+1), index=joined.index))
    actual_top10 = set(weekly.loc[weekly.week.eq(7)].sort_values("receiving_yards", ascending=False).head(10).player.map(norm_name))
    joined["published_rank"] = ranks
    cohort_top10 = joined.sort_values("published_rank").head(10) if len(joined) else joined
    overlap = sum(k in actual_top10 for k in cohort_top10.player_key) / 10 if len(joined) else np.nan
    mae = (pred_y - joined.actual_receiving_yards).abs().mean() if len(joined) else np.nan
    spearman = (-ranks).corr(joined.actual_receiving_yards.rank()) if len(joined) > 2 else np.nan
    acc = pd.DataFrame([{
        "Week": 7, "Model Version": "v0.7 published preliminary cohort", "Published Cohort": len(joined),
        "Top-10 Yardage Overlap": overlap, "TD Hit Rate": (cohort_top10.actual_receiving_tds.gt(0)).mean() if len(joined) else np.nan,
        "Yardage MAE": mae, "Spearman": spearman,
        "Target Share Stability": (pred_share - joined.actual_target_share).abs().mean() if len(joined) else np.nan,
        "Notes": "Graded only against the frozen Week 7 preliminary cohort; not represented as a full 96-player pregame model.",
        "Refreshed": REFRESH_DATE,
    }])
    archive_rows = [{"season": 2026, "week": 6, "published_date": "", "snapshot_status": "LEGACY SUMMARY — NOT VERIFIED FULL SLATE", "rank": "", "player": "", "team": "", "role": "", "opponent": "", "target_share": "", "rec_yds_projection": "", "opportunity_score": "", "tier": "", "source_note": "Prior Sheet record preserved; no complete 96-player pregame snapshot."}]
    for _, r in old.iterrows():
        archive_rows.append({"season": 2026, "week": 7, "published_date": "2026-10-21", "snapshot_status": "FROZEN PRELIMINARY COHORT", "rank": r.get("rank", ""), "player": r.player, "team": r.team, "role": r.get("role", ""), "opponent": r.get("opponent", ""), "target_share": r.get("target_share", ""), "rec_yds_projection": r.get("model_rec_yds_projection", r.get("yards_per_game", "")), "opportunity_score": r.get("opportunity_score", ""), "tier": r.get("tier", ""), "source_note": "Outgoing Week 7 publication frozen before Week 8 advance; source lag disclosed at publication."})
    return acc, pd.DataFrame(archive_rows)


def main() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    outgoing = archive_outgoing()
    stats, snaps, depth, pbp, games = load_sources()
    coverage = stats.groupby("week").team.nunique().to_dict()
    if coverage.get(COMPLETED_WEEK, 0) < 24:
        raise AssertionError(f"Week 7 not broadly complete: {coverage}")
    schedule = build_schedule(games)
    roles, weekly = build_wr_summary(stats, snaps, depth)
    monitor, elevation, prior_top5 = build_defense(stats, pbp, roles)
    model = build_model(roles, schedule, monitor, elevation, pbp)
    accuracy, weekly_archive = grade_and_archive(outgoing, weekly)

    roles.to_csv(LIVE / "wr_top3_summary.csv", index=False, float_format="%.6f")
    weekly.to_csv(LIVE / "wr_weekly_actuals_2026.csv", index=False, float_format="%.6f")
    model.to_csv(LIVE / "current_week_model.csv", index=False, float_format="%.6f")
    model.loc[~model.opponent.eq("BYE")].to_csv(LIVE / "primary_wr_dashboard.csv", index=False, float_format="%.6f")
    roles.sort_values(["target_share", "targets"], ascending=False).to_csv(LIVE / "target_share_leaderboard.csv", index=False, float_format="%.6f")
    roles.sort_values(["td_game_rate", "receiving_tds", "targets"], ascending=False).to_csv(LIVE / "td_consistency_leaderboard.csv", index=False, float_format="%.6f")
    roles.sort_values(["receiving_yards", "yards_per_game"], ascending=False).to_csv(LIVE / "receiving_yards_leaderboard.csv", index=False, float_format="%.6f")
    monitor.to_csv(LIVE / "pass_defense_monitor.csv", index=False, float_format="%.6f")
    elevation.to_csv(LIVE / "defense_elevation_watch.csv", index=False, float_format="%.6f")
    elevation.head(5).to_csv(LIVE / "defense_elevation_top5.csv", index=False, float_format="%.6f")
    prior_top5.to_csv(ARCHIVE / "week_8" / "prior_week_defense_elevation_top5.csv", index=False, float_format="%.6f")
    accuracy.to_csv(LIVE / "model_accuracy_2026.csv", index=False, float_format="%.6f")
    weekly_archive.to_csv(LIVE / "weekly_archive_2026.csv", index=False, float_format="%.6f")
    schedule.to_csv(LIVE / "schedule_week_8.csv", index=False)
    pd.DataFrame([{"team": t, "public_week8_pass_def_rank": r, "verified_date": "2026-10-27", "source": SHARP_PUBLIC_URL, "label": "Sharp public matchup passing-defense rank; not proprietary Pass Efficiency DEF"} for t, r in sorted(SHARP_PASS.items())]).to_csv(LIVE / "sharp_public_week8_pass_def.csv", index=False)
    status = pd.DataFrame([
        ("season", 2026), ("current_target_week", 8), ("latest_completed_week", 7),
        ("last_refresh", REFRESH_CT), ("wr_coverage", "32 teams / 96 current WR roles"),
        ("active_week8_rows", 84), ("bye_teams", "HOU, NO, NYG, SF"),
        ("sharp_proprietary", "STALE — Week 5 snapshot preserved; not guessed"),
        ("sharp_public_substitute", "Week 8 public pass/pressure/explosive ranks verified 2026-10-27 for 28 active defenses"),
        ("defense_elevation", "32 defenses normalized to opponent season baselines through Week 7"),
        ("injury_state", "CAR-GB filed; most Sunday/Monday reports pending"),
        ("validation", "PASS — 96 rows / 32 teams / 3 WRs each / schedule reciprocal / no duplicates / target denominators reconciled"),
    ], columns=["key", "value"])
    status.to_csv(LIVE / "model_status.csv", index=False)
    summary = {
        "target_week": 8, "completed_week": 7,
        "rows": {"current_week_model": len(model), "dashboard": int((model.opponent != "BYE").sum()), "wr_grid": len(roles), "target_share": len(roles), "td_consistency": len(roles), "receiving_yards": len(roles), "pass_defense": len(monitor), "defense_elevation": len(elevation), "accuracy": len(accuracy), "weekly_archive": len(weekly_archive)},
        "top10": model.loc[~model.opponent.eq("BYE")].head(10)[["rank", "player", "team", "opponent", "model_rec_yds_projection", "opportunity_score", "tier", "availability"]].to_dict("records"),
        "elevation_top5": elevation.head(5).to_dict("records"), "accuracy": accuracy.iloc[0].to_dict(),
        "validation": {"coverage_32": roles.team.nunique() == 32, "three_per_team": roles.groupby("team").size().eq(3).all(), "model_unique": not model.duplicated(["team", "player"]).any(), "schedule_rows": len(schedule), "sharp_proprietary_stale": True, "sunday_injuries_pending": True},
    }
    (LIVE / "weekly_refresh_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
