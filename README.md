# nfl-WR-stats

Data and live-feed repository for the 2026 NFL WR Opportunity Model.

## Purpose
Track each team's WR1/WR2/WR3 with three separate performance views:
- **Target Share** — highest share of team targets, top down.
- **Touchdown Consistency** — receiving-TD hit rate and total TDs, top down.
- **Receiving Yards** — receiving yards per game and total yards, top down.

The weekly matchup layer uses **Sharp Football Analysis Pass Efficiency DEF** as the proprietary schedule/matchup input when it can be verified. Sharp values are never guessed. A last-verified value is preserved separately and marked stale if a fresh value cannot be retrieved.

## Data integrity
- Player statistics: nflverse weekly player stats, derived from official NFL play-by-play/stat feeds.
- Current schedule: nflverse schedule data cross-checked against NFL schedule during weekly publication.
- WR depth/role: refreshed weekly from reputable depth-chart and team reporting sources by the scheduled scan.
- Sharp: Sharp Football Analysis NFL Strength of Schedule Tool, `Pass Efficiency DEF`.
- Injuries: official NFL/team injury reports are preferred.

## Live outputs
`data/live/` is the stable interface consumed by the LIVE Google Sheet. Historical weekly snapshots belong in `data/archive/` so published predictions are never rewritten after the fact.

## Weekly cadence
The LIVE dashboard is refreshed every Thursday at **3:00 AM America/Chicago**, aligned with the existing RB model workflow.
