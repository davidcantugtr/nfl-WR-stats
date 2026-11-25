from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

SHEETS = {
    "WR Dashboard": 902163795,
    "WR1-3 Grid": 1486725867,
    "Target Share": 1830887260,
    "TD Consistency": 74181414,
    "Receiving Yards": 1884020362,
    "Pass Defense Monitor": 1330909916,
    "Model Accuracy": 1106487887,
    "Weekly Archive": 1454328922,
    "System Status": 1346349814,
    "Defense Elevation": 996343615,
}


def value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {}
    if isinstance(v, (np.integer, int)):
        return {"userEnteredValue": {"numberValue": int(v)}}
    if isinstance(v, (np.floating, float)):
        return {"userEnteredValue": {"numberValue": float(v)}}
    return {"userEnteredValue": {"stringValue": str(v)}}


def write(sheet, row, col, rows):
    return {"updateCells": {"start": {"sheetId": SHEETS[sheet], "rowIndex": row, "columnIndex": col},
                            "rows": [{"values": [value(v) for v in r]} for r in rows],
                            "fields": "userEnteredValue"}}


def clear(sheet, r0, r1, c0, c1):
    return {"updateCells": {"range": {"sheetId": SHEETS[sheet], "startRowIndex": r0, "endRowIndex": r1,
                                      "startColumnIndex": c0, "endColumnIndex": c1},
                            "fields": "userEnteredValue"}}


def numfmt(sheet, r0, r1, c0, c1, pattern):
    return {"repeatCell": {"range": {"sheetId": SHEETS[sheet], "startRowIndex": r0, "endRowIndex": r1,
                                      "startColumnIndex": c0, "endColumnIndex": c1},
                           "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": pattern}}},
                           "fields": "userEnteredFormat.numberFormat"}}


def main():
    roles = pd.read_csv(LIVE / "wr_top3_summary.csv")
    model = pd.read_csv(LIVE / "current_week_model.csv")
    target = pd.read_csv(LIVE / "target_share_leaderboard.csv")
    td = pd.read_csv(LIVE / "td_consistency_leaderboard.csv")
    yards = pd.read_csv(LIVE / "receiving_yards_leaderboard.csv")
    defense = pd.read_csv(LIVE / "pass_defense_monitor.csv")
    elev = pd.read_csv(LIVE / "defense_elevation_watch.csv")
    acc = pd.read_csv(LIVE / "model_accuracy_2026.csv")
    archive = pd.read_csv(LIVE / "weekly_archive_2026.csv")

    req = []
    # Dashboard: retain the designed layout and replace only live values.
    req += [write("WR Dashboard", 0, 0, [["NFL 2026 WR OPPORTUNITY MODEL — WEEK 12 LIVE"]]),
            write("WR Dashboard", 2, 0, [["2026 usage through completed Week 11 • Official Week 12 schedule • Refreshed November 25, 2026"]]),
            write("WR Dashboard", 5, 0, [[12, "", "", "W1–W11 FINAL", "", "", 96, "", "", "PBP W1–W11 / SHARP PUBLIC W11 STALE / PROPRIETARY W5 STALE", "", "", 96]]),
            write("WR Dashboard", 7, 0, [["DATA WARNING: Sharp proprietary Pass Efficiency DEF remains inaccessible; its last verified Week 5 value is preserved as STALE and never guessed. Week 11 public Sharp pass/pressure/explosive ranks are preserved separately as stale because a complete Week 12 table was not reproducibly accessible. Current nflverse PBP through Week 11 drives the matchup layer. Bo Melton and Skyy Moore are questionable; most Sunday/Monday final reports are pending."]]),
            write("WR Dashboard", 9, 0, [["TOP 10 — WEEK 12 RECEIVING OPPORTUNITIES"]]),
            clear("WR Dashboard", 11, 21, 0, 16), clear("WR Dashboard", 25, 30, 0, 16)]
    top = model.loc[model.opponent.ne("BYE")].head(10)
    top_rows = []
    for _, r in top.iterrows():
        top_rows.append([r["rank"], r.player, r.team, r.opponent, r.site, r.role, r.target_share, r.yards_per_game,
                         r.td_game_rate, r.sharp_public_pass_def_rank, r.defense_elevation_score,
                         r.model_rec_yds_projection, r.model_floor, r.model_ceiling, r.opportunity_score, r.tier])
    req.append(write("WR Dashboard", 11, 0, top_rows))
    req.append(write("WR Dashboard", 25, 0, elev.head(5)[["rank", "defense", "elevation_score", "prior_rank", "rank_change", "trend", "QB_lift", "WR1_lift", "WR2_lift", "WR3_lift", "TE_lift", "RB_lift", "most_amplified", "games", "confidence"]].values.tolist()))
    latest_acc = acc.iloc[-1]
    req += [write("WR Dashboard", 23, 0, [["DEFENSE ELEVATION TOP 5 — THROUGH WEEK 11"]]),
            write("WR Dashboard", 32, 0, [["WEEK 11 ACCURACY — PUBLISHED TOP 10 + FULL ROLE COHORT"]]),
            write("WR Dashboard", 34, 0, [[latest_acc["Published Cohort"], latest_acc["Top-10 Yardage Overlap"], latest_acc["TD Hit Rate"], latest_acc["Yardage MAE"], latest_acc.Spearman, latest_acc["Target Share Stability"]]])]

    # Main role grid.
    req += [write("WR1-3 Grid", 0, 0, [["WR1–3 GRID — WEEK 12 CURRENT ROLES"]]),
            write("WR1-3 Grid", 1, 0, [["Actual 2026 usage through Week 11; role confidence combines current depth, targets and snaps. Verified injuries reduce availability; unfiled final reports remain pending."]]),
            clear("WR1-3 Grid", 4, 120, 0, 21)]
    cols = ["team", "usage_role", "player", "games", "targets", "team_season_targets", "target_share", "target_rank_team", "receptions", "receiving_yards", "yards_per_game", "yards_rank_team", "receiving_tds", "td_game_rate", "td_consistency_rank_team", "weekly_consistency", "latest_snap_share", "depth_rank", "role_confidence", "availability", "data_status"]
    req.append(write("WR1-3 Grid", 4, 0, roles[cols].values.tolist()))

    req += [write("Target Share", 0, 0, [["TARGET SHARE — 2026 THROUGH WEEK 11"]]), clear("Target Share", 4, 120, 0, 12)]
    rows = [[i + 1, r.player, r.team, r.usage_role, r.targets, r.team_season_targets, r.target_share, r.target_rank_team, r.receiving_yards, r.yards_per_game, r.td_game_rate, r.availability] for i, (_, r) in enumerate(target.iterrows())]
    req.append(write("Target Share", 4, 0, rows))

    req += [write("TD Consistency", 0, 0, [["TD CONSISTENCY — 2026 THROUGH WEEK 11"]]), clear("TD Consistency", 4, 120, 0, 12)]
    rows = [[i + 1, r.player, r.team, r.usage_role, r.receiving_tds, r.games_with_td, r.td_game_rate, r.td_consistency_rank_team, r.targets, r.target_share, r.yards_per_game, r.availability] for i, (_, r) in enumerate(td.iterrows())]
    req.append(write("TD Consistency", 4, 0, rows))

    req += [write("Receiving Yards", 0, 0, [["RECEIVING YARDS — 2026 THROUGH WEEK 11"]]), clear("Receiving Yards", 4, 120, 0, 12)]
    rows = [[i + 1, r.player, r.team, r.usage_role, r.receiving_yards, r.yards_per_game, r.ypg_median, r.ypg_std, r.weekly_consistency, r.targets, r.target_share, r.availability] for i, (_, r) in enumerate(yards.iterrows())]
    req.append(write("Receiving Yards", 4, 0, rows))

    req += [write("Pass Defense Monitor", 0, 0, [["PASS DEFENSE MONITOR — WEEK 12"]]),
            write("Pass Defense Monitor", 2, 0, [["Sharp proprietary Pass Efficiency DEF: last verified Week 5 snapshot preserved as STALE. Sharp public Week 11 pass / pressure / explosive ranks are retained separately as stale after the Week 12 recheck; current nflverse Weeks 1–11 PBP is the labeled substitute."]]),
            clear("Pass Defense Monitor", 6, 50, 0, 16)]
    dcols = ["team", "sharp_pass_def_rank", "last_verified", "sharp_proprietary_status", "sharp_public_week11_pass_rank_last_verified", "sharp_public_week11_pressure_rank_last_verified", "sharp_public_week11_explosive_rank_last_verified", "public_pass_def_rank", "pass_yds_allowed_pg", "completion_rate_allowed", "pass_tds_allowed_pg", "yards_per_attempt_allowed", "sack_rate", "pressure_event_rate", "explosive_pass_rate_allowed", "pass_success_rate_allowed"]
    req.append(write("Pass Defense Monitor", 6, 0, defense[dcols].values.tolist()))

    req += [write("Defense Elevation", 0, 0, [["DEFENSE ELEVATION WATCH — THROUGH WEEK 11"]]),
            write("Defense Elevation", 1, 0, [["Opponent-normalized lift versus each offense's own season baseline; rank movement compares Week 11 with Week 10."]]),
            clear("Defense Elevation", 5, 50, 0, 16)]
    ecols = ["rank", "defense", "elevation_score", "prior_rank", "rank_change", "trend", "QB_lift", "WR1_lift", "WR2_lift", "WR3_lift", "TE_lift", "RB_lift", "most_amplified", "games", "confidence", "source_updated"]
    req.append(write("Defense Elevation", 5, 0, elev[ecols].values.tolist()))

    a = acc.iloc[-1]
    req += [write("Model Accuracy", 2, 0, [["Week 11 is graded from finalized games. Ranking and yardage metrics use the frozen published Top 10; target-share stability uses all 96 archived WR role rows."]]),
            write("Model Accuracy", 5, 0, [[a.Week, a["Published Cohort"], a["Top-10 Yardage Overlap"], a["TD Hit Rate"], a["Yardage MAE"], a.Spearman, a["Target Share Stability"]]]),
            clear("Model Accuracy", 8, 30, 0, 10), write("Model Accuracy", 8, 0, acc.values.tolist())]

    req += [write("Weekly Archive", 1, 0, [["Outgoing Week 11 full 96-player role cohort and published Top-10 rankings are frozen before Week 12 advance; all earlier historical rows remain preserved."]]),
            clear("Weekly Archive", 4, 500, 0, 14)]
    req.append(write("Weekly Archive", 4, 0, archive.values.tolist()))

    status = [
        ["Current target week", "Week 12", "LIVE", "Official NFL schedule"],
        ["Completed statistical baseline", "Weeks 1–11", "FINAL", "nflverse weekly + PBP"],
        ["WR role coverage", "96 rows / 32 teams", "PASS", "Three current WRs per team"],
        ["Active matchup coverage", "96 rows / 32 teams", "PASS", "No byes"],
        ["Schedule alignment", "16 games / 32 reciprocal rows", "PASS", "NFL Week 12 schedule and flex changes"],
        ["Duplicate players", "0", "PASS", "Team-player key"],
        ["Target denominator", "All team pass targets", "PASS", "nflverse weekly stats"],
        ["Sharp proprietary", "Week 5 last verified", "STALE", "Never guessed"],
        ["Sharp public table", "Week 11 last verified", "STALE", "Week 12 table not reproducibly accessible on 2026-11-25"],
        ["Current public substitute", "PBP Weeks 1–11 pass defense", "CURRENT", "nflverse"],
        ["Defense elevation", "32 defenses", "PASS", "Opponent-baseline normalized"],
        ["Injury reports", "GB-LA final filed; most Sunday/Monday finals pending", "PROVISIONAL", "Official NFL Week 12 report"],
    ]
    req += [write("System Status", 0, 0, [["SYSTEM STATUS — WEEK 12"]]), clear("System Status", 5, 30, 0, 4), write("System Status", 5, 0, status)]

    # Correct previously inherited percentage formats while preserving visual styling.
    for sh, r0, r1, cols_pct in [
        ("WR Dashboard", 11, 21, [(6, 7), (8, 9)]),
        ("WR1-3 Grid", 4, 100, [(6, 7), (13, 14), (15, 17), (18, 19)]),
        ("Target Share", 4, 100, [(6, 7), (10, 11)]),
        ("TD Consistency", 4, 100, [(6, 7), (9, 10)]),
        ("Receiving Yards", 4, 100, [(8, 9), (10, 11)]),
        ("Pass Defense Monitor", 6, 38, [(9, 10), (12, 16)]),
        ("Defense Elevation", 5, 37, [(6, 12)]),
    ]:
        for c0, c1 in cols_pct:
            req.append(numfmt(sh, r0, r1, c0, c1, "0.0%"))
    for sh, r0, r1, c0, c1 in [
        ("WR Dashboard", 11, 21, 7, 8), ("WR Dashboard", 11, 21, 11, 15),
        ("Target Share", 4, 100, 4, 6), ("Target Share", 4, 100, 8, 10),
        ("TD Consistency", 4, 100, 4, 6), ("TD Consistency", 4, 100, 8, 9),
        ("TD Consistency", 4, 100, 10, 11),
        ("Receiving Yards", 4, 100, 4, 8),
        ("WR1-3 Grid", 4, 100, 5, 6), ("WR1-3 Grid", 4, 100, 10, 11),
        ("Weekly Archive", 4, 401, 10, 12),
    ]:
        req.append(numfmt(sh, r0, r1, c0, c1, "0.0"))
    req.append(numfmt("Weekly Archive", 4, 401, 9, 10, "0.0%"))
    expanded = []
    for item in req:
        uc = item.get("updateCells")
        if uc and len(uc.get("rows", [])) > 20 and "start" in uc:
            rows = uc["rows"]
            for offset in range(0, len(rows), 20):
                part = {"updateCells": dict(uc)}
                part["updateCells"]["rows"] = rows[offset:offset + 20]
                part["updateCells"]["start"] = dict(uc["start"])
                part["updateCells"]["start"]["rowIndex"] += offset
                expanded.append(part)
        else:
            expanded.append(item)
    req = expanded
    chunk = os.environ.get("SHEET_REQUEST_CHUNK")
    if chunk is not None:
        size = 1
        start = int(chunk) * size
        req = req[start:start + size]
    print(json.dumps(req, separators=(",", ":")))


if __name__ == "__main__":
    main()
