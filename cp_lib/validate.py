"""Validation tables in the spec's fixed order, and the scorecard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrate import to_bin
from .discover import UNKNOWN, label_of
from .savant_truth import load_leaderboard, load_league_bins

RATE_COLS = ["tied_up_percent", "centered_percent", "flailed_percent",
             "early_percent", "on_time_percent", "late_percent",
             "over_percent", "lined_up_percent", "under_percent",
             "perfect_percent", "flawed_percent"]
MEAN_COLS = ["avg_x_tied_up", "avg_x_flail", "avg_y_early", "avg_y_late",
             "avg_z_over", "avg_z_under"]
UNKNOWN_COLS = ["unknown_x_percent", "unknown_y_percent", "unknown_z_percent"]


def categorize(cal: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cal.index)
    for ax in ("x", "y", "z"):
        out[f"cat_{ax}"] = label_of(ax, cal[f"{ax}_cal"].to_numpy(float))
    # Savant's perfect is a ball IN PLAY that is centered, on time and lined up,
    # not any contact: its contact-split CSV reports perfect_percent as exactly
    # 0.0000 for fouls, and 0.6294 of balls in play against a league 0.2274.
    in_play = (df["contact_type"] == "in_play").to_numpy()
    whiff = (df["contact_type"] == "whiff").to_numpy()
    known = np.ones(len(out), bool)
    for ax in ("x", "y", "z"):
        known &= (out[f"cat_{ax}"].to_numpy() != UNKNOWN)
    good = ((out.cat_x == "Centered") & (out.cat_y == "OnTime") & (out.cat_z == "Linedup")).to_numpy()
    none = ((out.cat_x != "Centered") & (out.cat_y != "OnTime") & (out.cat_z != "Linedup")).to_numpy()
    out["perfect"] = in_play & good & known
    out["flawed"] = whiff & none & known
    return out


def player_rates(df: pd.DataFrame, cats: pd.DataFrame, key: str,
                 cal: pd.DataFrame | None = None) -> pd.DataFrame:
    g = pd.DataFrame({"id": df[key].to_numpy(),
                      "whiff": (df["contact_type"] == "whiff").to_numpy()})
    spec = (("x", ("Tiedup", "Centered", "Flail"),
             ("tied_up_percent", "centered_percent", "flailed_percent")),
            ("y", ("Late", "OnTime", "Early"),
             ("late_percent", "on_time_percent", "early_percent")),
            ("z", ("Over", "Linedup", "Under"),
             ("over_percent", "lined_up_percent", "under_percent")))
    for ax, labels, names in spec:
        c = cats[f"cat_{ax}"].to_numpy()
        for lab, name in zip(labels, names):
            g[name] = (c == lab)
        g[f"unknown_{ax}_percent"] = (c == UNKNOWN)
    g["perfect_percent"] = cats["perfect"].to_numpy()
    g["flawed_percent"] = cats["flawed"].to_numpy()
    by = g.groupby("id")
    rates = by.mean(numeric_only=True).rename(columns={"whiff": "whiff_rate"})
    rates["n_swings"] = by.size()
    if cal is not None:
        for ax, (lo, hi), (nlo, nhi) in (("x", ("Tiedup", "Flail"), ("avg_x_tied_up", "avg_x_flail")),
                                         ("y", ("Late", "Early"), ("avg_y_late", "avg_y_early")),
                                         ("z", ("Over", "Under"), ("avg_z_over", "avg_z_under"))):
            vals = pd.Series(np.asarray(cal[f"{ax}_cal"], float), index=g.index)
            for lab, name in ((lo, nlo), (hi, nhi)):
                m = cats[f"cat_{ax}"].to_numpy() == lab
                rates[name] = vals[m].groupby(g.loc[m, "id"]).mean()
    else:
        for name in MEAN_COLS:
            rates[name] = np.nan
    return rates.reset_index()


def compare_rates(ours: pd.DataFrame, savant: pd.DataFrame, min_swings: int = 100) -> pd.DataFrame:
    m = ours.merge(savant, on="id", suffixes=("_ours", "_sav"))
    swings = "n_swings_sav" if "n_swings_sav" in m else "n_swings"
    m = m[m[swings] >= min_swings]
    rows = []
    for c in RATE_COLS + MEAN_COLS + ["whiff_rate"]:
        a, b = f"{c}_ours", f"{c}_sav"
        if a not in m or b not in m:
            continue
        s = m[[a, b]].dropna()
        if s.empty:
            rows.append({"rate": c, "players": 0, "mae": np.nan, "pearson_r": np.nan,
                         "league_ours": np.nan, "league_savant": np.nan})
            continue
        w_ours = m.loc[s.index, "n_swings_ours"] if "n_swings_ours" in m else None
        w_sav = m.loc[s.index, "n_swings_sav"] if "n_swings_sav" in m else None
        rows.append({"rate": c, "players": len(s),
                     "mae": float((s[a] - s[b]).abs().mean()),
                     "pearson_r": float(s[a].corr(s[b])) if len(s) > 2 else np.nan,
                     "league_ours": float(np.average(s[a], weights=w_ours)),
                     "league_savant": float(np.average(s[b], weights=w_sav))})
    return pd.DataFrame(rows)


def savant_rates(season: int, type_: str, split: str | None = None) -> pd.DataFrame:
    df = load_leaderboard(season, type_, split)
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    return df


def contact_type_table(df: pd.DataFrame, cats: pd.DataFrame, cal: pd.DataFrame,
                       season: int, type_: str, key: str, min_swings: int = 50) -> pd.DataFrame:
    sav = savant_rates(season, type_, "bat_contact_code")
    rows = []
    for ct in ("in_play", "foul", "whiff"):
        sel = (df["contact_type"] == ct).to_numpy()
        if not sel.any():
            continue
        ours = player_rates(df[sel], cats[sel], key, cal[sel])
        cmp_ = compare_rates(ours, sav[sav.contact_type == ct], min_swings=min_swings)
        cmp_["contact_type"] = ct
        rows.append(cmp_)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def league_bins_table(cal: pd.DataFrame, df: pd.DataFrame, season: int) -> pd.DataFrame:
    sav = load_league_bins(season)
    sav = (sav.groupby(["axis", "bin"])
              .apply(lambda g: pd.Series({"n_sav": g["n"].sum(),
                                          "rv_sav": np.average(g["avg_run_exp"], weights=g["n"])}),
                     include_groups=False)
              .reset_index())
    sav["bin"] = sav["bin"].astype(float)
    rows = []
    for ax in ("x", "y", "z"):
        t = (pd.DataFrame({"bin": to_bin(ax, cal[f"{ax}_cal"].to_numpy(float)),
                           "rv": df["delta_run_exp"].to_numpy(float)})
             .dropna().groupby("bin").agg(n_ours=("rv", "size"), rv_ours=("rv", "mean")).reset_index())
        t["axis"] = ax
        rows.append(t)
    ours = pd.concat(rows, ignore_index=True)
    ours["bin"] = ours["bin"].astype(float)
    return ours.merge(sav, on=["axis", "bin"], how="outer").sort_values(["axis", "bin"])


def scorecard(results: dict, fit_target: tuple | None = None) -> pd.DataFrame:
    """results[(season, table)] -> a compare_rates or contact_type_table frame.

    fit_target names the one table the calibration was fit to. The per-contact-type
    map takes its band masses from the 2025 pitcher split CSV, so that table is the
    target rather than evidence. in_sample does not say this on its own: the 2025
    pitcher rates table is in sample too and is still a real comparison, because
    nothing was fit to the plain CSV's rates.
    """
    rows = []
    for (season, table), t in sorted(results.items()):
        if "rate" not in t:
            continue
        core = t[t.rate.isin(RATE_COLS)]
        players = core["players"].max()
        rows.append({"season": season, "table": table, "in_sample": season == 2025,
                     "calibration_target": fit_target is not None and (season, table) == tuple(fit_target),
                     "mean_mae_rates": round(float(core["mae"].mean()), 4),
                     "mean_r_rates": round(float(core["pearson_r"].mean()), 3),
                     "players": int(players) if pd.notna(players) else 0})
    return pd.DataFrame(rows)
