"""Recover Savant's per-swing definitions from the labeled whiff tails.

The tails are the 10 worst whiffs per list per player-season, selected on the
label, so they identify functional form, sign and rough scale only. Nothing
that ships is centered or scaled here (see calibrate.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .geometry import FEATURE_COLUMNS

DERIVED_ALL_SWING = {"icpt_y_over_ballspeed"}
ALL_SWING_FEATURES = set(FEATURE_COLUMNS) | DERIVED_ALL_SWING
THRESH = {"x": 4.0, "y": 7.0, "z": 2.0}
LABELS = {"x": ("Tiedup", "Centered", "Flail"),
          "y": ("Late", "OnTime", "Early"),
          "z": ("Over", "Linedup", "Under")}
SAV = {"x": "sav_x", "y": "sav_y", "z": "sav_z"}
MIN_PER_STAND = 30


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple
    by_stand: bool = False
    cost: int = 1


@dataclass
class CandidateResult:
    name: str
    axis: str
    r2: float
    label_agreement: float
    slope: float
    intercept: float
    coef: dict
    n: int
    cost: int
    all_swing: bool
    by_stand: bool = False
    stand_models: dict = field(default_factory=dict)
    n_features: int = 0


def _rank(r: "CandidateResult") -> tuple:
    """Cheapest first, then fewest features, then best fit.

    The feature count is a real tiebreaker, not decoration: at equal cost the
    plan's rule took the highest in-sample r2, which always prefers the wider
    model. On the tails that means preferring the more overfit explanation of
    a definition we are trying to read off, so parsimony wins first.
    """
    return (r.cost, r.n_features, -r.r2)


@dataclass
class Verdict:
    axis: str
    status: str
    best: CandidateResult | None
    ladder: list


CANDIDATES = {
    "x": [
        Candidate("icpt_x_field", ("icpt_x_field",), cost=1),
        Candidate("icpt_x_body", ("icpt_x_body",), cost=1),
        Candidate("icpt_x_body_by_stand", ("icpt_x_body",), by_stand=True, cost=1),
        Candidate("icpt_xy_body", ("icpt_x_body", "icpt_y"), by_stand=True, cost=2),
        Candidate("ball_lateral_at_icpt", ("bx_icpt", "icpt_x_body", "lateral_from_com_ft"),
                  by_stand=True, cost=2),
        Candidate("along_bat_from_com", ("along_bat_from_com",), by_stand=True, cost=3),
        Candidate("along_bat_plus_icpt",
                  ("along_bat_from_com", "icpt_x_body", "icpt_y", "attack_direction"),
                  by_stand=True, cost=3),
        Candidate("x_full_geometry",
                  ("icpt_x_body", "icpt_y", "attack_direction", "bat_speed",
                   "along_bat_from_com", "icpt_y_over_ballspeed", "bz_icpt", "bx_icpt",
                   "bz_minus_zone_mid"),
                  by_stand=True, cost=4),
    ],
    "y": [
        Candidate("icpt_y_raw", ("icpt_y",), cost=1),
        Candidate("icpt_y_by_stand", ("icpt_y",), by_stand=True, cost=1),
        Candidate("icpt_y_over_ballspeed", ("icpt_y_over_ballspeed",), by_stand=True, cost=1),
        Candidate("icpt_y_and_x", ("icpt_y", "icpt_x_body"), by_stand=True, cost=2),
        Candidate("attack_direction", ("attack_direction",), by_stand=True, cost=2),
        Candidate("icpt_y_attack_dir", ("icpt_y", "attack_direction"), by_stand=True, cost=2),
        Candidate("icpt_y_attack_dir_x",
                  ("icpt_y", "attack_direction", "icpt_x_body", "icpt_y_over_ballspeed"),
                  by_stand=True, cost=2),
        Candidate("time_based",
                  ("t_icpt_minus_plate_ms", "icpt_y_over_ballspeed", "attack_direction", "icpt_x_body"),
                  by_stand=True, cost=3),
    ],
    "z": [
        Candidate("z_above_plane", ("z_above_plane",), cost=1),
        Candidate("attack_minus_descent", ("attack_minus_descent",), cost=1),
        Candidate("bz_icpt_and_plane", ("bz_icpt", "plane_z_at_icpt", "swing_path_tilt", "attack_angle"),
                  cost=2),
        Candidate("z_plane_and_tilt", ("z_above_plane", "attack_minus_descent", "swing_path_tilt", "icpt_y"),
                  by_stand=True, cost=2),
        Candidate("z_with_miss", ("z_above_plane", "miss_distance", "attack_minus_descent"),
                  by_stand=True, cost=3),
        Candidate("z_zone_relative",
                  ("bz_minus_zone_mid", "zone_height", "descent_deg", "attack_angle",
                   "swing_path_tilt", "icpt_y", "bz_icpt"),
                  by_stand=True, cost=4),
        Candidate("z_miss_decomp",
                  ("z_from_miss", "z_above_plane", "attack_minus_descent", "bz_minus_zone_mid",
                   "descent_deg", "miss_distance"),
                  by_stand=True, cost=4),
    ],
}


UNKNOWN = "Unknown"


def label_of(axis: str, values) -> np.ndarray:
    """Savant's three category labels, plus Unknown for a value we could not compute.

    A NaN must never fall through to the middle label: Centered, OnTime and
    Linedup are the categories the validation is measuring, so calling an
    unknown swing one of them would inflate the very rate being checked.
    """
    v = np.asarray(values, float)
    lo, mid, hi = LABELS[axis]
    t = THRESH[axis]
    lab = np.where(v < -t, lo, np.where(v > t, hi, mid))
    return np.where(np.isfinite(v), lab, UNKNOWN)


def _prep(feats: pd.DataFrame) -> pd.DataFrame:
    f = feats.copy()
    if "icpt_y_over_ballspeed" not in f and {"icpt_y", "ball_in_per_ms"} <= set(f.columns):
        f["icpt_y_over_ballspeed"] = f["icpt_y"] / f["ball_in_per_ms"].replace(0, np.nan)
    return f


def _fit_one(X: np.ndarray, y: np.ndarray):
    m = LinearRegression().fit(X, y)
    return m, float(m.score(X, y))


def fit_candidate(feats: pd.DataFrame, axis: str, cand: Candidate) -> CandidateResult:
    f = _prep(feats)
    cols = list(cand.features)
    need = cols + [SAV[axis]] + (["stand"] if cand.by_stand else [])
    f = f.dropna(subset=[c for c in need if c in f.columns])
    y = f[SAV[axis]].to_numpy(float)
    all_swing = all(c in ALL_SWING_FEATURES for c in cols)
    if len(f) < MIN_PER_STAND:
        return CandidateResult(cand.name, axis, float("nan"), float("nan"), float("nan"),
                               float("nan"), {}, len(f), cand.cost, all_swing, cand.by_stand, {}, len(cols))
    if cand.by_stand:
        pred = np.full(len(f), np.nan)
        models = {}
        for s in ("R", "L"):
            m = (f["stand"] == s).to_numpy()
            if m.sum() < MIN_PER_STAND:
                continue
            model, _ = _fit_one(f.loc[m, cols].to_numpy(float), y[m])
            models[s] = {"coef": dict(zip(cols, model.coef_.tolist())),
                         "intercept": float(model.intercept_)}
            pred[m] = model.predict(f.loc[m, cols].to_numpy(float))
        ok = ~np.isnan(pred)
        if ok.sum() < MIN_PER_STAND:
            return CandidateResult(cand.name, axis, float("nan"), float("nan"), float("nan"),
                                   float("nan"), {}, int(ok.sum()), cand.cost, all_swing, True, models, len(cols))
        r2 = 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        slope = float(np.polyfit(pred[ok], y[ok], 1)[0])
        coef, intercept = {}, float("nan")
    else:
        model, r2 = _fit_one(f[cols].to_numpy(float), y)
        pred = model.predict(f[cols].to_numpy(float))
        ok = np.ones(len(f), bool)
        coef = dict(zip(cols, model.coef_.tolist()))
        intercept = float(model.intercept_)
        slope = float(model.coef_[0]) if len(cols) == 1 else float(np.polyfit(pred, y, 1)[0])
        models = {}
    agree = float(np.mean(label_of(axis, pred[ok]) == label_of(axis, y[ok])))
    return CandidateResult(cand.name, axis, float(r2), agree, slope, intercept, coef,
                           int(ok.sum()), cand.cost, all_swing, cand.by_stand, models, len(cols))


def run_ladder(feats: pd.DataFrame, axis: str, candidates: list | None = None) -> list:
    f = _prep(feats)
    have = set(f.columns)
    out = []
    for cand in (candidates if candidates is not None else CANDIDATES[axis]):
        if not set(cand.features) <= have:
            continue
        res = fit_candidate(f, axis, cand)
        if np.isfinite(res.r2):
            out.append(res)
    return sorted(out, key=_rank)


def verdict(results: list, r2_min: float = 0.8, agree_min: float = 0.9) -> Verdict:
    if not results:
        return Verdict("", "fallback", None, [])
    passing = [r for r in results if r.r2 >= r2_min and r.label_agreement >= agree_min]
    if passing:
        best = sorted(passing, key=_rank)[0]
        return Verdict(best.axis, "recovered", best, results)
    best = max(results, key=lambda r: r.r2)
    return Verdict(best.axis, "fallback", best, results)


def apply(res: CandidateResult, feats: pd.DataFrame) -> np.ndarray:
    f = _prep(feats)
    if res.by_stand:
        if not res.stand_models:
            return np.full(len(f), np.nan)
        out = np.full(len(f), np.nan)
        for s, m in res.stand_models.items():
            sel = (f["stand"] == s).to_numpy()
            cols = list(m["coef"])
            out[sel] = (f.loc[sel, cols].to_numpy(float)
                        @ np.array([m["coef"][c] for c in cols]) + m["intercept"])
        return out
    cols = list(res.coef)
    return f[cols].to_numpy(float) @ np.array([res.coef[c] for c in cols]) + res.intercept


def ladder_table(verdicts: dict) -> pd.DataFrame:
    rows = []
    for axis, v in verdicts.items():
        for r in v.ladder:
            rows.append({"axis": axis, "candidate": r.name, "cost": r.cost, "n_features": r.n_features, "n": r.n,
                         "r2": round(r.r2, 3), "label_agreement": round(r.label_agreement, 3),
                         "slope": round(r.slope, 3), "all_swing": r.all_swing,
                         "chosen": r is v.best, "status": v.status})
    return pd.DataFrame(rows)
