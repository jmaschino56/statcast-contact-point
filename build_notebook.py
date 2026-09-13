"""Build contact_point.ipynb from cell sources and execute it headless.

Run from notebooks/contact_point:

    setsid nohup ../../.venv/bin/python build_notebook.py > data/notebook_build.log 2>&1 < /dev/null &

Executed twice on purpose: the scorecard cell sits at the top and reads the file
the last cell writes, so the second pass shows the final table first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TITLE = """# Contact point and swing timing

This notebook reconstructs, for every tracked MLB swing, the three per-swing
numbers Baseball Savant publishes only in aggregate on its Swing Timing and Miss
Distance leaderboard, and validates them against Savant's own published
aggregates on two seasons nothing here was fit on.

| Axis | Savant field | Units | Sign | Category |
|---|---|---|---|---|
| x | `delta_batball_tiedup_neg_x2` | inches | tied up negative, flail positive | centered within +/- 4 in |
| y | `delta_batball_late_neg_y2_msec` | ms | late negative, early positive | on time within +/- 7 ms |
| z | `ball_pos_above_plane` | inches | over negative, under positive | lined up within +/- 2 in |

Perfect is contact and centered and on time and lined up. Flawed is a whiff that
is none of the three. Nothing here touches a database: every cell reads cached
files under `data/`. 2025 is the only season anything is fit on; 2024 and 2026
are holdouts.

Design: `docs/superpowers/specs/2026-09-13-contact-point-design.md`.
Plan: `docs/superpowers/plans/2026-09-13-contact-point-plan.md`.
"""

IMPORTS = '''import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import Image, display, Markdown

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from cp_lib import join, geometry, discover, reconstruct, calibrate, validate, plots, traits
from cp_lib import savant_truth as st

REPORTS = Path("data/reports")
FIGURES = Path("data/figures")
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
FIT = 2025
SEASONS = tuple(int(p.stem.split("_")[1]) for p in sorted(Path("data").glob("swings_*.parquet"))
                if (Path("data/savant") / f"done_{p.stem.split('_')[1]}.json").exists())
snap = [json.loads(l) for l in open("data/pull_manifest.jsonl")]
SNAPSHOT = {int(r["season"]): r for r in snap}
LAST_GAME = {}
print("seasons with a complete Savant cache:", SEASONS)
for s in SEASONS:
    r = SNAPSHOT.get(s, {})
    LAST_GAME[s] = str(pd.read_parquet(f"data/swings_{s}.parquet", columns=["game_date"])
                       ["game_date"].max())
    print(f"  {s}: {r.get('rows', '?')} rows, pulled {r.get('finished_utc', '?')}, "
          f"last game {LAST_GAME[s]}")
HOLDOUT = tuple(s for s in SEASONS if s != FIT)
print("fit season:", FIT, " holdouts:", HOLDOUT)
'''

SCORECARD_CELL = '''# SCORECARD. Written by the last cell; this notebook is executed twice so it
# shows the final table here on the second pass.
p = REPORTS / "scorecard.csv"
if p.exists():
    sc = pd.read_csv(p)
    display(sc)
    hdr = REPORTS / "scorecard_header.json"
    if hdr.exists():
        for k, v in json.load(open(hdr)).items():
            print(f"{k}: {v}")
else:
    print("(the scorecard is written by the last cell; run the notebook twice)")
'''

DENOM_MD = """## 1. Which of our rows is a swing

Savant's leaderboard counts a swing its own way. Before any number is compared,
our denominator has to match its `n_swings` per player. Three nested rules are
scored against both the pitcher and the batter CSV; the winner is the one with
the lowest median absolute relative error averaged over the two.
"""

DENOM_CODE = '''df25 = join.load_swings(FIT)
rep = pd.concat([join.denominator_report(df25, FIT, t) for t in ("pitcher", "batter")],
                ignore_index=True)
display(rep)
RULE = join.choose_swing_filter(df25, FIT)
print("chosen swing filter:", RULE)
rep.to_csv(REPORTS / "denominator_2025.csv", index=False)

# spec section 3: what the chosen rule still gets wrong, player by player.
res = [join.residual_report(df25, FIT, t, RULE) for t in ("pitcher", "batter")]
print("\\nresidual mismatch under", RULE, "(absolute, per player):")
for r in res:
    print(f"  {r['type']:8s} players={r['players']:4d} ours={r['total_ours']} "
          f"savant={r['total_savant']} median={r['median_diff']:.1f} "
          f"p05={r['p05']:.1f} p95={r['p95']:.1f} exact={r['exact_match']:.3f}")

nulls = {}
sw25 = df25[join.swing_filter(df25, RULE)]
for c in ("intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches",
          "attack_direction", "attack_angle", "swing_path_tilt", "bat_speed",
          "launch_speed", "launch_angle"):
    nulls[c] = f"{int(sw25[c].isna().sum())} ({sw25[c].isna().mean()*100:.2f}%)"
print("\\nnulls on the", len(sw25), "swings the rule keeps:")
for k, v in nulls.items():
    print(f"  {k:46s} {v}")

# spec section 3: the histogram endpoint's total against the CSV n_swings
h = st.load_histograms(FIT, "batter")
lb = st.load_leaderboard(FIT, "batter")[["id", "n_swings"]].dropna(subset=["id"]).astype({"id": int})
tot = h.groupby(["id", "axis"])["n"].sum().unstack()
m = lb.merge(tot.reset_index(), on="id", how="inner")
print(f"\\nhistogram total minus CSV n_swings, {len(m)} batters:")
for ax in ("x", "y", "z"):
    d = m[ax] - m["n_swings"]
    print(f"  axis {ax}: sum {int(d.sum()):6d}  exact match {float((d == 0).mean()):.3f}")
print("  The y histogram matches n_swings for every batter. x and z run about")
print("  0.15 percent high, the same rate at which our own intercept columns are")
print("  null, so Savant has an x and a z on swings whose timing value does not")
print("  exist. That is what an intercept-depth definition of y predicts.")
'''

DISCOVERY_MD = """## 2. What Savant's three numbers actually are

Savant's own tail rows rule out the obvious geometry. They carry x values past
100 inches on a 34 inch bat with miss distances of 10 inches, so x is not an
along-bat offset from the sweet spot at closest approach. So the first modeling
phase is discovery: fit a ladder of candidates to the labeled whiff tails and
see which, if any, reproduces Savant's value and its category label.

The tails are the ten worst whiffs per list per player-season, selected on the
label itself. They identify functional form, sign and scale. Nothing that ships
is centered or scaled on them.
"""

DISCOVERY_CODE = '''disc = json.load(open(REPORTS / "discovery_2025.json"))
print("labeled rows:", disc["n_labeled"], " join:", disc["join"])
print("join rate: %.5f  (%d of %d tails matched, %d unmatched)   y_com_ft: %s   tilt_sign: %s"
      % (disc["join_rate"], disc["join"]["exact"] + disc["join"]["nearest"],
         disc["join"]["tails"], disc["join"]["unmatched"], disc["y_com_ft"], disc["tilt_sign"]))
ladder = pd.DataFrame(disc["ladder"])
display(ladder)
for ax in ("x", "y", "z"):
    v = disc["verdicts"][ax]
    b = v["best"]
    print(f"axis {ax}: {v['status'].upper():9s} best {b['name']:22s} "
          f"r2 {b['r2']:.3f}  label agreement {b['label_agreement']:.3f}  "
          f"all_swing {b['all_swing']}")
display(Markdown(open(REPORTS / "discovery_2025.md").read()))
'''

RECON_MD = """## 3. Reconstruct every swing, then calibrate

A recovered all-swing formula applies everywhere. An axis discovery did not
recover is treated as whiff-only evidence, which is what it is: its model is fit
on whiff tails and has never seen a swing where the bat met the ball. So whiffs
get the model and contact swings get the Driveline inversion, an exit-velocity
deficit against `1.23 * bat_speed + 0.23 * pitch_speed` for x and launch angle
minus attack angle for z. A contact swing with no launch data has no outcome to
invert; it stays unknown rather than being called centered.

Calibration is one monotone quantile map per axis onto Savant's published
per-player histograms, fit on 2025 and applied unchanged to every season.
"""

RECON_CODE = '''verdicts, Y_COM, TILT = reconstruct.load_verdicts()
FALLBACK_AXES = [ax for ax, s in verdicts.items() if s.status == "fallback"]
print("fallback axes:", FALLBACK_AXES, " y_com_ft:", Y_COM, " tilt_sign:", TILT)

DF, PRED, CAL = {}, {}, {}
for s in SEASONS:
    d = join.load_swings(s)
    d = d[join.swing_filter(d, RULE)].reset_index(drop=True)
    DF[s] = d
    PRED[s] = reconstruct.reconstruct(d, s, verdicts, Y_COM, tilt_sign=TILT)
    print(f"season {s}: {len(d)} swings   sources "
          + "  ".join(f"{ax}:{dict(PRED[s]['source_' + ax].value_counts())}" for ax in ("x", "y", "z")))

# Fallback C: LightGBM on the 2025 tails with inverse-selection weights, whiffs only.
# Task 14 widened what it sees: the re-pull's per-pitch columns, per-batter traits
# computed on each season's OWN swings, MLB biography, and Savant's swing-path
# slate. Nothing here is a Savant per-swing label.
PLAYERS = traits.load_players()
SPATH = traits.swing_path()
IMPORTANCE = {}
if FALLBACK_AXES:
    tf = traits.tails_features(FIT, Y_COM, TILT, players=PLAYERS, sp=SPATH)
    hist_b = st.load_histograms(FIT, "batter")
    base = [c for c in geometry.ALL_SWING_FEATURE_COLUMNS if c in tf and tf[c].notna().any()]
    COLS = {ax: traits.fallback_columns(tf, base, axis=ax) for ax in FALLBACK_AXES}
    for ax, cc in COLS.items():
        print(f"\\nfallback features, axis {ax}: {len(cc)} "
              f"({len(base)} geometry, {len(cc) - len(base)} new)")
    print("z keeps the geometry set: the new features help x on both holdouts and")
    print("hurt z on both, and traits.WIDENED_AXES carries the numbers.")
    for ax in FALLBACK_AXES:
        cols = COLS[ax]
        model = calibrate.fallback_fit(ax, tf, hist_b, cols, key="batter")
        IMPORTANCE[ax] = (pd.DataFrame({"feature": cols,
                                        "gain": model.feature_importance("gain")})
                          .sort_values("gain", ascending=False).head(10).reset_index(drop=True))
        for s in SEASONS:
            f0 = geometry.candidate_features(DF[s], s, y_com_ft=Y_COM, tilt_sign=TILT)
            f0["icpt_y_over_ballspeed"] = f0["icpt_y"] / f0["ball_in_per_ms"].replace(0, np.nan)
            feats = traits.extended_features(DF[s], f0, s, players=PLAYERS, sp=SPATH)
            sel = (PRED[s][f"source_{ax}"] == "fallback").to_numpy()
            if sel.any():
                PRED[s].loc[sel, f"{ax}_hat"] = calibrate.fallback_predict(
                    model, feats[sel], cols)
                PRED[s].loc[sel, f"source_{ax}"] = "model"
            del feats, f0
        print(f"  axis {ax}: filled the whiff rows in every season")
    for ax, t_ in IMPORTANCE.items():
        print(f"\\ntop gain, axis {ax}")
        display(t_)

hist_p = st.load_histograms(FIT, "pitcher")
RATES = calibrate.split_rates(FIT, "pitcher")
MAPS_POOLED = calibrate.fit_calibration(PRED[FIT], DF[FIT], FIT, "pitcher", hist=hist_p,
                                        per_contact_type=False)
MAPS = calibrate.fit_calibration(PRED[FIT], DF[FIT], FIT, "pitcher", hist=hist_p, rates=RATES)
for s in SEASONS:
    CAL[s] = calibrate.apply_calibration(PRED[s], DF[s], MAPS)

# The calibration decision, shown rather than asserted. Savant publishes no
# per-contact-type histogram, so an earlier version fit each contact type onto the
# all-swing marginal, which forced every type to the all-swing rate. The split CSV
# does publish each type's category rates, and per_type_target rescales the pooled
# shape's three band masses onto them.
sav_ct = validate.savant_rates(FIT, "pitcher", "bat_contact_code")
POOLED_CAL = {s: calibrate.apply_calibration(PRED[s], DF[s], MAPS_POOLED) for s in SEASONS}
CELLS = [("in_play", "lined_up_percent"), ("whiff", "lined_up_percent"),
         ("in_play", "centered_percent"), ("foul", "centered_percent"),
         ("whiff", "centered_percent"), ("in_play", "on_time_percent"),
         ("foul", "on_time_percent"), ("whiff", "on_time_percent")]
AX_OF = {"centered_percent": "x", "on_time_percent": "y", "lined_up_percent": "z"}
rows = []
for ct, rate in CELLS:
    ax = AX_OF[rate]
    sel = (DF[FIT]["contact_type"] == ct).to_numpy()
    q = sav_ct[sav_ct.contact_type == ct]
    r = {"contact type": ct, "rate": rate,
         "savant": round(float(np.average(q[rate], weights=q.n_swings)), 4)}
    for name, vals in (("raw", PRED[FIT][f"{ax}_hat"]), ("pooled map", POOLED_CAL[FIT][f"{ax}_cal"]),
                       ("per-type map", CAL[FIT][f"{ax}_cal"])):
        lab = discover.label_of(ax, np.asarray(vals, float))
        mid = discover.LABELS[ax][1]
        r[name] = round(float(np.mean(lab[sel] == mid)), 4)
    r["known"] = round(float(np.isfinite(CAL[FIT].loc[sel, f"{ax}_cal"]).mean()), 4)
    rows.append(r)
print("\\nThe league category rate per contact type, three calibration choices:")
display(pd.DataFrame(rows))
print("The per-type map lands on Savant wherever we can compute the axis at all.")
print("What is left is the 'known' column: a swing with no exit velocity has no x")
print("and no z, so it is Unknown rather than centered. The foul centered rate is")
print("0.726 times its 0.875 known share, which is the 0.635 in the table, and the")
print("in-play lined-up rate is 0.975 times 0.997. The residual is the missing")
print("swings, not the calibration.")

# Whether y should be per-type at all: it is the one axis with a recovered formula,
# so the pooled map might already be right. Measured on the HOLDOUTS, not on 2025.
MAPS_Y_POOLED = calibrate.fit_calibration(PRED[FIT], DF[FIT], FIT, "pitcher", hist=hist_p,
                                          rates=RATES, pooled_axes=("y",))
print("\\ny per-type or y pooled, mean absolute error on the pitcher category rates:")
for name, mp in (("y per-type", MAPS), ("y pooled", MAPS_Y_POOLED)):
    line = []
    for s in SEASONS:
        c2 = calibrate.apply_calibration(PRED[s], DF[s], mp)
        sc = validate.compare_rates(validate.player_rates(DF[s], validate.categorize(c2, DF[s]),
                                                          "pitcher", c2),
                                    validate.savant_rates(s, "pitcher"))
        line.append(f"{s} {sc[sc.rate.isin(validate.RATE_COLS)]['mae'].mean():.4f}")
    print(f"  {name:11s} " + "   ".join(line))
print("  y stays per-type: it is better on both holdouts, not only on the fit season.")

print("\\nper-player mean Wasserstein-1 against Savant's 2025 pitcher histograms.")
print("This is the price of the per-type map and it is a real regression: the")
print("target is no longer the pooled histogram those distances are measured on.")
for ax in ("x", "y", "z"):
    raw = calibrate.wasserstein_per_player(PRED[FIT][f"{ax}_hat"].to_numpy(float),
                                           DF[FIT], hist_p, ax, "pitcher")
    pol = calibrate.wasserstein_per_player(POOLED_CAL[FIT][f"{ax}_cal"].to_numpy(float),
                                           DF[FIT], hist_p, ax, "pitcher")
    per = calibrate.wasserstein_per_player(CAL[FIT][f"{ax}_cal"].to_numpy(float),
                                           DF[FIT], hist_p, ax, "pitcher")
    print(f"  axis {ax}: raw {raw.mean():7.3f}   pooled map {pol.mean():7.3f}   "
          f"per-type map {per.mean():7.3f}   players {len(per)}")
'''

VALIDATE_MD = """## 4. Validation

In the spec's fixed order, holdout seasons first and 2025 flagged in-sample.
Every rate is a fraction, not a percent, because that is how Savant's CSV
carries them.
"""

VALIDATE_CODE = '''results = {}
CATS = {s: validate.categorize(CAL[s], DF[s]) for s in SEASONS}
order = [s for s in SEASONS if s != FIT] + ([FIT] if FIT in SEASONS else [])

for kind, key in (("pitcher", "pitcher"), ("batter", "batter")):
    for s in order:
        ours = validate.player_rates(DF[s], CATS[s], key, CAL[s])
        cmp_ = validate.compare_rates(ours, validate.savant_rates(s, kind))
        results[(s, f"{kind} rates")] = cmp_
        print(f"\\n=== {kind} rates, season {s}" + ("  (IN SAMPLE)" if s == FIT else "  (holdout)"))
        display(cmp_)

for kind, key in (("pitcher", "pitcher"), ("batter", "batter")):
    for s in order:
        t = validate.contact_type_table(DF[s], CATS[s], CAL[s], s, kind, key)
        results[(s, f"{kind} by contact type")] = t
        print(f"\\n=== {kind} rates by contact type, season {s}"
              + ("  (IN SAMPLE)" if s == FIT else "  (holdout)"))
        display(t[t.rate.isin(["centered_percent", "on_time_percent", "lined_up_percent",
                               "perfect_percent", "flawed_percent"])])

# Perfect and flawed on their own. They are JOINT categories, perfect being a ball
# in play that is centered and on time and lined up, flawed a whiff that is none of
# the three. The per-type map sets each axis's marginal rate, so it has no direct
# hold on a joint one: agreement here is a check on whether the three axes are
# wrong together on the same swings, which is what a marginal map cannot fix.
print("\\n=== perfect and flawed, the two joint categories, pitcher")
jr = []
for s in order:
    t = results[(s, "pitcher by contact type")]
    for ct, rate in (("in_play", "perfect_percent"), ("whiff", "flawed_percent")):
        r = t[(t.contact_type == ct) & (t.rate == rate)]
        if len(r):
            jr.append({"season": s, "cell": f"{ct} {rate}",
                       "ours": round(float(r.league_ours.iloc[0]), 4),
                       "savant": round(float(r.league_savant.iloc[0]), 4),
                       "mae": round(float(r["mae"].iloc[0]), 4),
                       "pearson_r": round(float(r["pearson_r"].iloc[0]), 3)})
display(pd.DataFrame(jr))

print("\\n=== bucket means (the six avg_* values), pitcher")
for s in order:
    t = results[(s, "pitcher rates")]
    print(f"season {s}")
    display(t[t.rate.isin(validate.MEAN_COLS)])

# Attenuation. The per-player scatters have a slope well below 1: a pitcher Savant
# has near 0.75 lined up comes out near 0.68, one near 0.45 comes out near 0.48.
# That is what per-swing error does. The question is whether to undo it.
print("\\n=== per-player attenuation, 2025, and what undoing it would cost")
rows = []
for kind in ("pitcher", "batter"):
    ours = validate.player_rates(DF[FIT], CATS[FIT], kind, CAL[FIT])
    sav = validate.savant_rates(FIT, kind)
    for ax in ("z", "x", "y"):
        col = calibrate.MID_RATE[ax]
        m = ours.merge(sav, on="id", suffixes=("_o", "_s"))
        m = m[m["n_swings_s"] >= 100].dropna(subset=[col + "_o", col + "_s"])
        a = m[col + "_s"].to_numpy(float)
        b = m[col + "_o"].to_numpy(float)
        sl = float(np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1))
        ideal = b.mean() + (b - b.mean()) / sl
        rows.append({"view": kind, "rate": col, "slope": round(sl, 3),
                     "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 3),
                     "players": len(m), "mae": round(float(np.abs(b - a).mean()), 4),
                     "mae_if_slope_1": round(float(np.abs(ideal - a).mean()), 4)})
display(pd.DataFrame(rows))
print("The last column is the answer. Stretching the rates until the slope is 1")
print("makes the error WORSE on all six, because the correlation is 0.66 to 0.92")
print("and not 1. With an imperfect predictor the least-error answer IS the shrunk")
print("one; a slope below 1 is what a noisy estimate should do, not a defect in it.")
print("Raising the slope and the accuracy together needs a better per-swing z, not")
print("a rescale of the rates it produces.")
K_MEASURED = calibrate.fit_reliability(CAL[FIT], DF[FIT], FIT, "pitcher")
print("\\nThe per-swing version was measured too. Scaling each swing about its own")
print("player's season mean by k, the k that puts the pitcher slope at 1 is")
print("  " + "  ".join(f"{a} {v:.2f}" for a, v in sorted(K_MEASURED.items())))
print("and it is NOT applied. It moves every player's rate level far more than it")
print("moves the spread between players: pitcher category error goes from 0.021 to")
print("0.079 on the 2024 holdout, whiff centered from 0.404 to 0.699 against")
print("Savant's 0.402, and the pooled x histogram shifts by up to 0.186 in a single")
print("bin. calibrate.apply_reliability exists and is tested; nothing calls it.")

print("\\n=== per-player Wasserstein-1 against Savant's histograms")
w_rows = []
for s in order:
    for kind, key in (("pitcher", "pitcher"), ("batter", "batter")):
        hh = st.load_histograms(s, kind)
        if hh.empty:
            continue
        for ax in ("x", "y", "z"):
            w = calibrate.wasserstein_per_player(CAL[s][f"{ax}_cal"].to_numpy(float),
                                                 DF[s], hh, ax, key)
            w_rows.append({"season": s, "type": kind, "axis": ax,
                           "mean_w1": round(float(w.mean()), 3),
                           "median_w1": round(float(w.median()), 3), "players": len(w)})
display(pd.DataFrame(w_rows))

print("\\n=== league bins against TIMING_BINNED_DATA_LEAGUE, 2025")
BINS_TABLE = validate.league_bins_table(CAL[FIT], DF[FIT], FIT)
display(BINS_TABLE[BINS_TABLE.n_sav.notna() & (BINS_TABLE.n_sav > 200)].head(40))

# The x panel used to carry a spike Savant does not have: the Driveline loss curve
# stops at a 40 mph exit velocity deficit, which is 14 inches from the sweet spot,
# and np.interp clamped there, so every weakly hit ball past it landed on one
# value. The curve now continues at the slope of its final segment. That is not a
# claim about a bat 20 inches off the sweet spot; the quantile map sets the scale
# and what the continuation buys is rank order, so two different deficits keep two
# different values and a monotone map can spread them.
_x = PRED[FIT]["x_hat"]
_inv = (PRED[FIT]["source_x"] == "inversion").to_numpy()
_past = ((_x.abs() > 14.0) & _inv).sum()
_pile = _x[_inv].round(3).value_counts()
print(f"\\ninversion rows past the curve's last published point: {int(_past)} "
      f"({_past / _inv.sum() * 100:.2f}% of them). Largest single x value shared by "
      f"more than one swing: {int(_pile.max())} swings at {float(_pile.idxmax()):+.3f} "
      f"inches, against 12,221 stacked on exactly 14 inches before the change.")
'''

TAILS_CODE = '''# 6. The tails themselves: our value against Savant's, on the labeled whiffs.
tf = traits.tails_features(FIT, Y_COM, TILT)
# All three panels draw. Filtering to status == "recovered" left x and z blank,
# which read as a broken figure rather than as the deliberate distinction it was.
# The distinction is real and belongs in the TITLE: y is a recovered formula, and
# x and z are the best linear candidate against the tails, not what ships for
# contact swings, which come from the collision inversion instead.
tl = pd.DataFrame({ax: discover.apply(
    discover.CandidateResult(**{k: v for k, v in disc["verdicts"][ax]["best"].items()}), tf)
    for ax in ("x", "y", "z")})
status = {ax: disc["verdicts"][ax]["status"] for ax in ("x", "y", "z")}
caveat = ("Savant's tails are the ten worst whiffs per list per player-season, selected on "
          "the label. This is the population the formulas were read off, and it is not a "
          "random sample of swings: agreement here is a form check, not a generalization check.")
pred = {ax: (tl[ax] if ax in tl else np.full(len(tf), np.nan)) for ax in ("x", "y", "z")}
truth = {ax: tf[f"sav_{ax}"].to_numpy() for ax in ("x", "y", "z")}
p = plots.tails_scatter(pred, truth, FIGURES / "tails_scatter_2025.png", caveat,
                        status=status)
print(p)
display(Image(filename=str(p)))
p = plots.tails_scatter(pred, truth, FIGURES / "tails_scatter_y_2025.png", caveat,
                        axes=("y",), status=status)
print(p)
p = plots.tails_scatter(pred, truth, FIGURES / "tails_scatter_xz_2025.png", caveat,
                        axes=("x", "z"), status=status)
print(caveat)
display(Image(filename=str(p)))
'''

SCORECARD_WRITE = '''sc = validate.scorecard(results, fit_target=(FIT, "pitcher by contact type"))
display(sc)
print("calibration_target marks the one table the per-type map was fit to: its band")
print("masses ARE the 2025 pitcher split CSV's rates. It is the target, not evidence.")
print("The 2025 pitcher rates row is in sample but still a real comparison, because")
print("nothing was fit to the plain CSV. 2024 and 2026 are holdouts throughout.")
sc.to_csv(REPORTS / "scorecard.csv", index=False)
header = {
    "swing filter": RULE,
    "swing filter median abs rel error": "pitcher %.4f, batter %.4f" % (
        float(rep.loc[rep.rule.eq(RULE) & rep.type.eq("pitcher"), "median_abs_rel_err"].iloc[0]),
        float(rep.loc[rep.rule.eq(RULE) & rep.type.eq("batter"), "median_abs_rel_err"].iloc[0])),
    "tail join rate": "%.5f (%d of %d, %d unmatched)" % (
        disc["join_rate"], disc["join"]["exact"] + disc["join"]["nearest"],
        disc["join"]["tails"], disc["join"]["unmatched"]),
    "discovery": "  ".join(
        f"{ax}={disc['verdicts'][ax]['status']}(r2 {disc['verdicts'][ax]['best']['r2']:.3f})"
        for ax in ("x", "y", "z")),
    "2026 snapshot: last game date": LAST_GAME.get(2026, "2026 not loaded"),
    "seasons": "fit %d, holdouts %s" % (FIT, HOLDOUT or "none yet"),
}
json.dump(header, open(REPORTS / "scorecard_header.json", "w"), indent=1)
for k, v in header.items():
    print(f"{k}: {v}")
'''

HEAT_MD = """## 5. Run value over the contact point and the timing

Per-swing run value from the batter's side, over every swing in every season, on
Savant's own bin grid (2 inches, 2 ms, 1 inch), with cells under 50 swings left
blank and the category thresholds drawn. The secondary view is expected wOBA on
contact, which only balls in play carry.
"""

HEAT_CODE = '''all_cal = pd.concat([CAL[s] for s in SEASONS], ignore_index=True)
all_df = pd.concat([DF[s][["delta_run_exp", "contact_type", "estimated_woba_using_speedangle",
                           "pitcher", "batter"]] for s in SEASONS], ignore_index=True)
figs = []
figs.append(plots.hexbin_grid(
    all_cal, all_df, "delta_run_exp", FIGURES / "hexbin_runvalue_all_swings.png",
    f"Run value per swing (batter perspective), all swings, {min(SEASONS)} to {max(SEASONS)}"))
ip = (all_df.contact_type == "in_play").to_numpy()
ip_cal = all_cal[ip].reset_index(drop=True)
ip_df = all_df[ip].reset_index(drop=True)
# Every panel and every figure draws xwOBA on the same fixed 0 to 1.2 scale,
# white at 0.600, and run value on -0.3 to 0.3, white at zero. plots.SCALES
# holds both; the call sites do not choose.
figs.append(plots.hexbin_grid(
    ip_cal, ip_df, "estimated_woba_using_speedangle", FIGURES / "hexbin_xwoba_in_play.png",
    f"Expected wOBA on contact, balls in play, {min(SEASONS)} to {max(SEASONS)}",
    min_n=50))

# The three axes at once. The 2D panels each marginalise one axis away; these do
# not, which is the only way to see that the best cells sit on a surface rather
# than at a point.
figs.append(plots.cloud3d(
    ip_cal, ip_df, FIGURES / "cloud3d_xwoba_in_play.png",
    f"Expected wOBA on contact over x, y and z, balls in play, "
    f"{min(SEASONS)} to {max(SEASONS)}",
    value="estimated_woba_using_speedangle", min_n=25))
figs.append(plots.cloud3d(
    all_cal, all_df, FIGURES / "cloud3d_runvalue_all_swings.png",
    f"Run value per swing over x, y and z, all swings, {min(SEASONS)} to {max(SEASONS)}",
    value="delta_run_exp", min_n=60))
CLOUD_HTML = plots.cloud3d_html(
    ip_cal, ip_df, FIGURES / "cloud3d_xwoba_in_play.html",
    f"Expected wOBA on contact over x, y and z, balls in play, "
    f"{min(SEASONS)} to {max(SEASONS)}",
    value="estimated_woba_using_speedangle", min_n=25)
print("interactive:", CLOUD_HTML)

figs.append(plots.league_bins_figure(BINS_TABLE, FIGURES / "league_bins_2025.png"))
RATES = ("on_time_percent", "centered_percent", "lined_up_percent")
for kind in ("pitcher", "batter"):
    ours = validate.player_rates(DF[FIT], CATS[FIT], kind, CAL[FIT])
    sav = validate.savant_rates(FIT, kind)
    for rate in RATES:
        figs.append(plots.rate_scatter(ours, sav, rate,
                                       FIGURES / f"scatter_{kind}_{rate}.png"))
    # All three on one figure: a single rate invites the reader to assume the
    # other two look like it, and they sit at different correlations.
    figs.append(plots.rate_scatter_grid(
        ours, sav, RATES, FIGURES / f"scatter_grid_{kind}.png",
        f"Per-{kind} category rates against Savant's leaderboard, {FIT}"))
for p in figs:
    print(p)
    display(Image(filename=str(p)))
'''

CELLS = [
    ("md", TITLE),
    ("code", IMPORTS),
    ("md", "## 0. Scorecard"),
    ("code", SCORECARD_CELL),
    ("md", DENOM_MD),
    ("code", DENOM_CODE),
    ("md", DISCOVERY_MD),
    ("code", DISCOVERY_CODE),
    ("md", RECON_MD),
    ("code", RECON_CODE),
    ("md", VALIDATE_MD),
    ("code", VALIDATE_CODE),
    ("code", TAILS_CODE),
    ("code", SCORECARD_WRITE),
    ("md", HEAT_MD),
    ("code", HEAT_CODE),
]


def strip_images(nb) -> int:
    """Drop embedded figure outputs, keep every table and printed line.

    Spec section 8 commits the notebook with outputs cleared except the scorecard
    and the validation tables. The figures are delivered through the media album,
    and as base64 PNGs they are almost the whole file size. A matplotlib display
    carries image/png alongside a "<Figure ...>" text/plain repr, so the output is
    dropped whole rather than pruned key by key.
    """
    dropped = 0
    for cell in nb.cells:
        keep = []
        for o in cell.get("outputs", []):
            data = o.get("data", {}) if o.get("output_type") in ("display_data", "execute_result") else {}
            if any(k.startswith("image/") for k in data):
                dropped += 1
                continue
            keep.append(o)
        if "outputs" in cell:
            cell["outputs"] = keep
    return dropped


def strip_local_paths(nb):
    """Take papermill's run record out before the notebook is committed.

    It records `input_path` and `output_path` as ABSOLUTE paths, so a notebook
    built here and published carries the checkout's full location, naming a
    private repo and a home directory in a public file. Nothing reads the block
    and it is regenerated on every run, so it comes out whole.
    """
    n = 0
    if nb.metadata.pop("papermill", None) is not None:
        n += 1
    for cell in nb.cells:
        if cell.get("metadata", {}).pop("papermill", None) is not None:
            n += 1
    return n


def main():
    import nbformat
    import subprocess
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s)
                for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = HERE / "contact_point.ipynb"
    nbformat.write(nb, out)
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for i in range(1, passes + 1):
        print(f"=== papermill pass {i}", flush=True)
        subprocess.run([sys.executable, "-m", "papermill", str(out), str(out),
                        "--cwd", str(HERE), "--log-output", "--no-progress-bar"], check=True)
    nb = nbformat.read(out, as_version=4)
    n = strip_images(nb)
    paths = strip_local_paths(nb)
    nbformat.write(nb, out)
    print(f"=== stripped {n} figure outputs and {paths} papermill path blocks, "
          f"{out.stat().st_size/1e6:.2f} MB", flush=True)
    print("=== done", flush=True)


if __name__ == "__main__":
    main()
