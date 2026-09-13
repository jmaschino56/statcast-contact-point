# Contact point and swing timing reconstruction: design

Date: 2026-09-13. Branch: `feat/contact-point-notebook`. Status: approved by Jeremy
(sections 1 to 8 on 2026-09-12 evening CT), ready for an implementation plan.

## Goal

Back-calculate, for every tracked MLB swing in `savant_data`, the three numbers
Baseball Savant publishes only in aggregate on its Swing Timing leaderboard
(`/leaderboard/bat-tracking/swing-timing-miss-distance`):

| Axis | Savant field (per swing) | Units | Sign | Category thresholds |
|---|---|---|---|---|
| x | `delta_batball_tiedup_neg_x2` | inches | tied up negative, flail positive | centered within +/- 4 in |
| y | `delta_batball_late_neg_y2_msec` | ms | late negative, early positive | on time within +/- 7 ms |
| z | `ball_pos_above_plane` | inches | over negative, under positive | lined up within +/- 2 in |

Perfect = contact AND centered AND on time AND lined up. Flawed = whiff AND none of
the three. The convention is Savant's; plots label the categories in Jeremy's
words (tied up / centered / flail, early / on time / late, over / lined up / under).

Success is measured against Savant's aggregates, pitcher side first, batter side
second, on seasons the fit never saw. The deliverable is a Jupyter notebook plus a
small library, nothing written to the database.

## Decisions already made

- Axis convention: Savant's (above).
- Seasons: develop on 2025; 2024 and 2026 are holdouts and are never used to fit
  anything. 2023 is excluded (bat tracking starts at the 2023 All-Star break).
- Ground truth: scrape Savant's undocumented per-player endpoints once, cache them.
- Perspective: pitcher aggregates first, batter aggregates second. Nothing persisted
  as tables.
- Heatmap population: per-swing run value over all swings (whiffs and fouls carry
  their `delta_run_exp`, balls in play too), with in-play xwOBA as a secondary view.
- Location: `notebooks/contact_point/` in this repo.
- Approach: B (recover Savant's definitions empirically, then reconstruct), with
  per-dimension fallback to C (weakly supervised histogram matching) when a
  dimension's tail fit has R-squared under 0.8.
- The data pulls run now, one season at a time, cached to parquet.

## What is known going in

### Savant exposes more than the leaderboard shows

1. Leaderboard CSV, `?season[]=YYYY&type={pitcher|batter|league}&min=1&csv=true`:
   one row per player-season with `n_swings`, `whiff_rate`, `competitive_percent`,
   `miss_distance` (mean over whiffs), the nine category rates, `perfect_percent`,
   `flawed_percent`, and the six bucket means (`avg_x_tied_up`, `avg_x_flail`,
   `avg_y_early`, `avg_y_late`, `avg_z_over`, `avg_z_under`). `season=` alone is
   ignored and returns the current season; the array form is required.
   Adding `&split[]=bat_contact_code` gives the same row per contact type:
   code 2 = hit in play, 4 = foul, 9 = whiff. 2025 has 826 pitchers and 671
   batters with at least one swing.
2. Page HTML embeds `TIMING_BINNED_DATA_LEAGUE` (x bins of 2 in, y bins of 2 ms,
   z bins of 1 in, each with `n` and `avg_run_exp`, split by pitch hand and bat
   side) and `TIMING_leagueAverageDataFilteredGrouped` (league counts per
   category). Only the requested season is embedded, so each season's page is
   fetched separately.
3. Per-player histogram endpoint,
   `/leaderboard/bat-tracking/swing-timing-distribution/player/{id}?season[]=YYYY&type={t}`:
   JSON `{x:[{bin,n,avg_run_exp,year}], y:[...], z:[...]}` over ALL that player's
   swings. Works for `type=pitcher` and `type=batter`.
4. Per-swing tail endpoint,
   `/leaderboard/services/swing-timing?id={id}&season[]=YYYY&type={t}`:
   JSON with four lists (`miss_distance`, `x`, `y`, `z`), each the 10 WORST whiffs
   for that player-season by that quantity, each row carrying `play_id`,
   `game_pk`, `game_date`, `pitcher`, `savant_batter_id`, `pitch_type`,
   `miss_distance_inches`, the three signed values and the three category labels.
   The leaked SQL shows the source view `baseball.analytics_batpath_extended_onerow`
   filtered to `miss_distance_inches is not null`, `is_pitch`, `is_swing_typical`,
   game type and year, `limit` fixed at 10. No parameter raises the limit.

### Savant's definitions are NOT the obvious geometry

Tail rows already show `x = 43.6 in` with `miss = 10.2 in` and `x = 30.3 in` with
`miss = 18.8 in`. A signed along-bat offset from the sweet spot cannot reach 43 in
on a 34 in bat, and a ball 30 in past the sweet spot cannot be 18.8 in from the top
half of the bat at closest approach. So x is not "along the bat at closest approach"
referenced to the sweet spot, or it is measured at a different instant than miss
distance. Likewise the contact-type split shows in-play swings are only about 81
percent on time, while contact-instant geometry (ball radius plus bat radius over
bat speed) caps an along-velocity offset near 2 ms; y for contacts is a temporal
quantity defined some other way. Only z behaves like contact-instant geometry
(in play 99.5 percent lined up, whiffs 13 percent).

Therefore the first modeling phase is DISCOVERY of Savant's definitions from the
labeled tails, not derivation from first principles.

### Our data

`savant_data` has bat tracking on 316,987 (2024), 330,420 (2025) and 302,269
(2026 to date) regular-season pitches. `miss_distance` exists only on whiffs
(about 23 percent of swings). `play_id` is NULL in our table, so the tail join uses
`(game_pk, batter, pitcher, round(miss_distance, 6))`.

Pulled 2026-09-12 (7:37 PM CT) into `data/swings_<season>.parquet`: 340,278 rows
(2024), 339,082 (2025), 310,405 (2026 through 2026-09-12). Bat speed is present on
316,987 / 330,420 / 302,269 of those. Savant's 2025 `n_swings` sums to 326,997
over the batter CSV, so the swing filter (section 3) has about 3,400 rows to
account for beyond "bat speed present", of which bunt descriptions are 1,380.
Every row carries `delta_run_exp`; launch data exists on about 70 percent of rows
(in play plus tracked fouls).

`plate_x` / `plate_z` changed reference from front-of-plate (through 2025) to
middle-of-plate (2026). Any ball position used in a candidate is evaluated from
the 9-parameter trajectory (`vx0, vy0, vz0, ax, ay, az` at y = 50 ft) at an
explicit y, never read from the plate columns, so 2026 behaves like 2024 and 2025.

### The Driveline deck (Saberseminar 2026, Pelletier / Ehrlich / Stokey)

Read in full (61 image slides). The deck is not redistributed here.
Relevant content, all from Stage 4:
- They invert a bat-ball collision model built on Nava Wolfish's public model plus
  Alan Nathan's work so that exit velocity and launch angle map back to X = inches
  along the bat from the tip (how much of the collision the ball keeps) and Y =
  angle around the barrel (direction and spin off the bat).
- Perfect-contact ceiling: `EV = 1.23 * bat_speed + 0.23 * pitch_speed`, validated
  against 600k+ MLB swings within -1 percent.
- 12 degrees under the ball is about +19 degrees of launch and +1600 rpm; EV falls
  off the further contact gets from the matching vertical approach angle.
- Bearing model: EV, LA, bat direction at contact and handedness give landing
  bearing (trained on MLB ball flights).
- Timing is not a dimension in the deck. Every timing definition here comes from
  Savant.

## Architecture

```
notebooks/contact_point/
  contact_point.ipynb        the deliverable: scorecard, discovery, validation, heatmaps
  README.md                  how to regenerate from cache; what each cell proves
  cp_lib/
    __init__.py
    pull.py                  one-shot season pull to parquet (named cursor, detached)
    savant_truth.py          scraper + cache + loaders for the four truth sources
    join.py                  tail rows -> our rows; swing filter; denominator checks
    geometry.py              trajectory eval, bat frame, candidate features
    discover.py              ordered candidate fits on tails; recovered/needs-fallback
    reconstruct.py           apply recovered formulas; Driveline inversion for contact
    calibrate.py             histogram calibration; fallback C
    validate.py              the validation tables in the fixed order
    plots.py                 heatmaps and validation figures
  data/                      gitignored: swings_<season>.parquet, savant/ cache, people.json
tests/
  test_contact_point_geometry.py   trajectory eval, frame construction, sign conventions
  test_contact_point_join.py       join key, dedupe, swing filter
  test_contact_point_calibrate.py  quantile map round-trips, fallback weighting
```

Every stage reads only from `data/`. The notebook never opens a database
connection. `*.ipynb` is ignored repo-wide, so `.gitignore` gets
`!notebooks/contact_point/*.ipynb`.

## 1. Data layer

`cp_lib/pull.py --season YYYY` writes `data/swings_YYYY.parquet`.

- Source: `savant_data WHERE game_year = %s AND game_type = 'R' AND (description IN
  (<swing set>) OR bat_speed IS NOT NULL)`. Swing set: `foul, hit_into_play,
  swinging_strike, swinging_strike_blocked, foul_tip, foul_bunt, missed_bunt,
  bunt_foul_tip, foul_pitchout, swinging_pitchout`.
- Columns (about 50): `game_pk, game_date, game_year, at_bat_number, pitch_number,
  batter, pitcher, player_name, stand, p_throws, pitch_type, release_speed,
  effective_speed, release_extension, release_pos_x, release_pos_y, release_pos_z,
  vx0, vy0, vz0, ax, ay, az, plate_x, plate_z, sz_top, sz_bot, vaa, haa, pfx_x,
  pfx_z, bat_speed, swing_length, attack_angle, attack_direction, swing_path_tilt,
  intercept_ball_minus_batter_pos_x_inches, intercept_ball_minus_batter_pos_y_inches,
  miss_distance, description, events, bb_type, launch_speed, launch_angle,
  hc_x, hc_y, hit_distance_sc, estimated_woba_using_speedangle, woba_value,
  woba_denom, delta_run_exp, delta_pitcher_run_exp, batted_ball_xrv, balls, strikes,
  outs_when_up, inning, home_team, away_team, age_bat`.
- Mechanism: direct psycopg2 connection (same parameters as
  `db_connection.get_direct_connection`), named cursor, `itersize` 50,000, one
  parquet row group per fetch, zstd, no ORDER BY (partition pruning gives a plain
  sequential scan; a sort would spill to the production volume).
- Runs detached (`setsid nohup`, verify ppid 1), seasons sequential: 2025, 2024,
  2026. Log to `data/pull.log`. A manifest line per season records row count,
  columns and the UTC pull timestamp.
- Batter attributes: MLB Stats API `people?personIds=...` for every batter id in
  the pulls, cached to `data/people.json` (height, bat side, birth date).

## 2. Ground truth layer

`cp_lib/savant_truth.py fetch --seasons 2024 2025 2026` populates `data/savant/`
with one raw file per URL (sha1 of the URL as the file name plus a `manifest.jsonl`
mapping URL to file, status, bytes, fetched-at). Re-running skips cached URLs.

Order per season:
1. Leaderboard CSVs for `type` in `pitcher, batter, league`, `min=1`, plain and
   `split[]=bat_contact_code`.
2. The leaderboard page HTML for `type=league`, plain and with the contact-code
   split, parsed for `TIMING_BINNED_DATA_LEAGUE` and
   `TIMING_leagueAverageDataFilteredGrouped`.
3. For every id in the pitcher CSV and every id in the batter CSV: the histogram
   endpoint and the tail endpoint, with the matching `type`.

Throttle: 0.5 s between requests, a browser User-Agent, 3 retries with backoff on
non-200 or on a JSON body carrying `error`. Roughly 9,000 requests, about 75
minutes, detached with a log.

Loaders return tidy frames:
- `load_leaderboard(season, type, split=None)`
- `load_league_bins(season)` and `load_league_grouped(season)`
- `load_histograms(season, type)` -> rows `(id, axis, bin, n, avg_run_exp)`
- `load_tails(season)` -> one row per unique `play_id` across every list, every
  type and every player it appeared under, with a `lists` column recording which
  lists selected it.

## 3. Join and denominator reconciliation

- Tail rows join on `(game_pk, batter, pitcher, round(miss_distance, 6))`. Report
  the join rate; expected above 99 percent. Rows that fail on rounding fall back to
  `(game_pk, batter, pitcher, pitch_type)` plus nearest miss distance within 1e-3,
  and any remaining misses are listed, not dropped silently.
- Swing filter: the set of our rows counted as a swing must match Savant's
  per-player `n_swings` within 1 percent median absolute relative error, on both
  the pitcher and batter CSVs. Candidates tried in order: bat speed present; then
  excluding bunt descriptions; then excluding rows Savant would not call a typical
  swing (very low bat speed, checked swings). The chosen filter is recorded with
  its error, and the notebook prints the residual mismatch distribution.
- The histogram endpoint's total `n` versus the CSV `n_swings` (1,345 vs 1,265 for
  one 2025 batter) is investigated (game types, ties across bins, missing axis
  values) and the explanation written into the notebook before histograms are used.

## 4. Discovery protocol

Input: the deduplicated 2025 tail rows joined to our columns. Output: for each
axis, the recovered formula or a "needs fallback" verdict.

Candidates, cheapest first. Each is fit as a linear model (and, where the sign
convention could flip with handedness, with `stand` interaction) and scored by
R-squared on the tails, by the fraction of Savant's own category labels reproduced
when the fitted value is thresholded, and by whether the slope is near 1 (a scale
match) or merely monotone (a reparameterization).

x (tied up / flail):
1. `intercept_ball_minus_batter_pos_x_inches`, raw and sign-flipped by `stand`.
2. Ball lateral position at the intercept depth from the trajectory, relative to
   an estimated sweet-spot lateral position from a per-batter reach constant.
3. Full bat-frame projection: sweet-spot-to-ball vector dotted with the bat axis.

y (early / late):
1. `intercept_ball_minus_batter_pos_y_inches`, raw.
2. Intercept y minus an ideal depth as a function of lateral pitch location and
   `stand`.
3. `attack_direction` (MLB documents it as reflecting timing), alone and with
   intercept y.
4. Time-based: ball time-of-flight between the intercept depth and a reference
   depth, divided by ball speed at the plate, combined with 1 to 3.
5. Full bat-frame projection onto the sweet-spot velocity, divided by bat speed.

z (over / under):
1. Ball height at the intercept depth from the trajectory minus a plane through
   the intercept with `swing_path_tilt`, using `attack_angle` for the plane's
   pitch.
2. `attack_angle` minus the pitch descent angle at the plate (vaa), scaled.
3. Full bat-frame projection onto the swing-plane normal.

Acceptance: R-squared at or above 0.8 on the deduped tails AND label agreement at
or above 90 percent for that axis. If a cheap candidate passes, the expensive ones
are still run and reported, so the notebook shows the ladder.

Each recovered formula is then classified as ALL-SWING (uses only fields present on
every swing) or WHIFF-ONLY (uses `miss_distance`). That classification drives
section 5.

The tails are the 10 worst per list per player-season, selected on the label. They
are used only to identify functional form, sign and rough scale. No centering or
scaling parameter that ships is fit on them.

## 5. Reconstruction and calibration

- ALL-SWING formulas apply to every swing directly.
- WHIFF-ONLY formulas apply to whiffs; contact swings (in play and foul) use the
  Driveline inversion:
  - x: EV deficit relative to `1.23 * bat_speed + 0.23 * pitch_speed_at_plate`
    (pitch speed from the trajectory at the intercept depth) mapped through an
    asymmetric EV-loss curve (steeper toward the handle, per Nathan's node
    geometry) to an unsigned along-bat distance; sign from the intercept geometry
    (ball nearer the body than the sweet-spot estimate is negative).
  - z: `launch_angle - attack_angle` mapped through a monotone offset curve
    (Nathan-style oblique collision; the deck's 12 degrees under to +19 degrees
    launch is one calibration point), bounded to +/- 2.75 in for balls in play and
    unbounded for fouls.
  - y: `attack_direction` and intercept depth, using whatever functional form
    discovery found on whiffs, re-centered for contact.
  - Fouls without launch data get the x and z estimate from geometry alone with a
    wider prior; the notebook reports how many such rows exist.
- Calibration (2025 only): for each axis and each contact type, a monotone quantile
  map so that the per-player predicted histograms match Savant's per-player
  histograms (objective: mean per-player Wasserstein-1 distance on the bin grid,
  weighted by swings). The six published bucket means are checked after
  calibration, not used as targets.
- Fallback C, per axis, only when discovery fails that axis: a LightGBM regressor
  on all public features fit to the tails with inverse-selection weights (the
  probability a swing with label value v was selected into a list is estimated
  from Savant's per-player histogram: 10 over the count of swings in the tail
  region for that player), then the same per-player quantile map as above. The
  notebook states plainly which axes are recovered and which are modeled.

## 6. Validation

Fixed order, every table shown for 2024 and 2026 (holdout) and 2025 (flagged
in-sample):
1. Pitcher rates: our nine category rates, perfect and flawed rates versus Savant,
   per pitcher with at least 100 swings. MAE, Pearson r, and a scatter per rate.
   League totals side by side.
2. Batter rates: same.
3. Contact-type split, pitcher and batter: the same rates per contact type. This
   is the sharpest test because in-play z must sit near 99 percent lined up and
   whiff z near 13 percent.
4. Bucket means: the six `avg_*` values, league and per player.
5. Histograms: per-player Wasserstein-1 distance per axis versus Savant's
   histograms; league bin counts and `avg_run_exp` per bin versus
   `TIMING_BINNED_DATA_LEAGUE`.
6. Tails: per-swing scatter of our value against Savant's on the labeled whiffs,
   with the selection caveat printed next to it.

A scorecard cell at the top of the notebook summarizes 1 to 6 with one line per
season per table, and states the swing filter, the join rate and the snapshot date
of the 2026 data.

## 7. Heatmaps

Per-swing run value (`delta_run_exp`, batter perspective) over (x, y), (x, z) and
(y, z), all swings, league-wide, on Savant's bin grid (2 in, 2 ms, 1 in), minimum 50
swings per cell, diverging palette centered at zero. Overlays: whiff share per
cell. Helper functions filter to a pitcher id or batter id. Secondary view: in-play
xwOBA (`estimated_woba_using_speedangle`) on the same grids. Figures are saved as
PNG under `data/figures/` and delivered through `bash ~/projects/media/drop.sh`
(one album, one URL).

## 8. Runtime and tests

- Branch `feat/contact-point-notebook` off `main`.

## Out of scope

- Any production pipeline integration or new tables.
- College or minors data.
- Reproducing Savant's competitive-swing definition beyond what denominator
  reconciliation needs.
- The Driveline swing-space, feasibility and sweep stages (5 to 11 of the deck).
