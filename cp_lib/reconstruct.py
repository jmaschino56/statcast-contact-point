"""Apply the recovered formulas to every swing; invert outcomes where a formula needs whiff-only fields."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import DATA
from .geometry import candidate_features

# inches from the sweet spot -> mph of exit velocity lost, toward the tip and
# toward the handle. The handle side is steeper (Nathan's node geometry), so the
# same deficit implies a contact point closer to the sweet spot on that side.
LOSS_CURVE_TIP = ([0, 2, 4, 6, 8, 14], [0, 2, 6, 12, 20, 40])
LOSS_CURVE_HANDLE = ([0, 2, 4, 6, 8, 14], [0, 3, 9, 18, 30, 60])
OFFSET_PER_DEG = 0.030
Z_CLIP = {"in_play": 2.75, "foul": 6.0, "whiff": 6.0}
AXES = ("x", "y", "z")


@dataclass
class FormulaSpec:
    status: str
    all_swing: bool
    coef: dict
    intercept: float
    by_stand: bool
    stand_models: dict = field(default_factory=dict)


def load_verdicts(path=DATA / "reports" / "discovery_2025.json") -> tuple[dict, float, float]:
    j = json.load(open(path))
    out = {}
    for ax, v in j["verdicts"].items():
        b = v["best"]
        out[ax] = FormulaSpec(v["status"], b["all_swing"], b["coef"], b["intercept"],
                              b["by_stand"], b.get("stand_models", {}))
    return out, float(j.get("y_com_ft", 0.0)), float(j.get("tilt_sign", 1.0))


def ev_ceiling(bat_speed_mph, pitch_speed_mph):
    """The deck's perfect-contact ceiling, validated there against 600k+ MLB swings."""
    return 1.23 * np.asarray(bat_speed_mph, float) + 0.23 * np.asarray(pitch_speed_mph, float)


def _loss_curve(d: np.ndarray, curve) -> np.ndarray:
    """Inches from the sweet spot for an exit velocity deficit, rising without bound.

    np.interp clamps past the last published point, which put every swing with a
    deficit over 40 mph on exactly 14 inches: 12,221 of 2025's swings, a spike
    Savant's own histogram does not have, and one a monotone quantile map cannot
    spread. The continuation is the slope of the final published segment. It is
    not a claim about a bat 20 inches off the sweet spot; the quantile map sets
    the scale afterwards, and what this has to preserve is rank order, so that two
    swings with different deficits keep different values.
    """
    inches, mph = np.asarray(curve[0], float), np.asarray(curve[1], float)
    out = np.interp(d, mph, inches)
    slope = (inches[-1] - inches[-2]) / (mph[-1] - mph[-2])
    past = d > mph[-1]
    out[past] = inches[-1] + (d[past] - mph[-1]) * slope
    return out


def x_from_ev_deficit(deficit_mph, toward_handle):
    d = np.clip(np.asarray(deficit_mph, float), 0, None)
    tip = _loss_curve(d, LOSS_CURVE_TIP)
    hand = _loss_curve(d, LOSS_CURVE_HANDLE)
    return np.where(np.asarray(toward_handle, bool), -hand, tip)


def z_from_launch(launch_angle, attack_angle, in_play):
    z = (np.asarray(launch_angle, float) - np.asarray(attack_angle, float)) * OFFSET_PER_DEG
    clip = np.where(np.asarray(in_play, bool), Z_CLIP["in_play"], Z_CLIP["foul"])
    return np.clip(z, -clip, clip)


def _apply_formula(spec: FormulaSpec, feats: pd.DataFrame) -> np.ndarray:
    if spec.by_stand:
        out = np.full(len(feats), np.nan)
        for s, m in spec.stand_models.items():
            sel = (feats["stand"] == s).to_numpy()
            cols = list(m["coef"])
            out[sel] = (feats.loc[sel, cols].to_numpy(float)
                        @ np.array([m["coef"][c] for c in cols]) + m["intercept"])
        return out
    cols = list(spec.coef)
    if not cols:
        return np.full(len(feats), np.nan)
    return feats[cols].to_numpy(float) @ np.array([spec.coef[c] for c in cols]) + spec.intercept


def reconstruct(df: pd.DataFrame, season: int, verdicts: dict, y_com_ft: float,
                tilt_sign: float | None = None) -> pd.DataFrame:
    """x_hat, y_hat, z_hat plus the provenance of each.

    source_* is one of:
      formula    the recovered all-swing formula, or the whiff-only formula on a whiff
      inversion  a contact swing whose outcome was inverted for a whiff-only formula
      no_outcome a contact swing whose outcome fields are missing, so nothing was
                 inverted. These stay NaN on purpose: filling them with zero would
                 read as a perfectly centered, lined-up swing and inflate exactly
                 the rates this notebook is validating.
      fallback   a whiff on an axis discovery did not recover; calibrate.py's
                 model fills it later

    A fallback axis is routed exactly like a whiff-only recovered one. The
    fallback model is fit on Savant's whiff tails and has never seen a swing
    where the bat met the ball, so letting it score contact swings would put
    their values in the whiff range and the quantile map would keep them there.
    Spec section 5 already says whiff-only evidence applies to whiffs and
    contact swings get the outcome inversion; a LightGBM fit on whiff tails is
    whiff-only evidence.
    """
    feats = candidate_features(df, season, y_com_ft=y_com_ft, tilt_sign=tilt_sign)
    feats["stand"] = df["stand"].to_numpy()
    feats["miss_distance"] = df["miss_distance"].to_numpy(float)
    feats["icpt_y_over_ballspeed"] = feats["icpt_y"] / feats["ball_in_per_ms"].replace(0, np.nan)

    whiff = (df["contact_type"] == "whiff").to_numpy()
    in_play = (df["contact_type"] == "in_play").to_numpy()
    launch_speed = df["launch_speed"].to_numpy(float)
    launch_angle = df["launch_angle"].to_numpy(float)

    deficit = ev_ceiling(df["bat_speed"].to_numpy(float),
                         feats["ball_speed_mph"].to_numpy(float)) - launch_speed
    toward_handle = feats["icpt_x_body"].to_numpy() < np.nanmedian(feats["icpt_x_body"].to_numpy())
    x_inv = x_from_ev_deficit(np.nan_to_num(deficit, nan=0.0), toward_handle)
    x_inv[np.isnan(launch_speed)] = np.nan
    z_inv = z_from_launch(launch_angle, df["attack_angle"].to_numpy(float), in_play)
    inversion = {
        "x": x_inv,
        # timing has no outcome inversion; the contact re-centering lives in calibrate
        "y": np.zeros(len(df)),
        "z": z_inv,
    }

    out = pd.DataFrame(index=df.index)
    stats = {}
    for ax in AXES:
        spec = verdicts[ax]
        vals = np.full(len(df), np.nan)
        src = np.full(len(df), "fallback", dtype=object)
        if spec.status == "recovered" and spec.all_swing:
            vals, src = _apply_formula(spec, feats), np.full(len(df), "formula", dtype=object)
        else:
            if spec.status == "recovered":
                vals[whiff] = _apply_formula(spec, feats)[whiff]
                src[whiff] = "formula"
            vals[~whiff] = inversion[ax][~whiff]
            src[~whiff] = "inversion"
            missing = (~whiff) & np.isnan(vals)
            # A contact swing with no exit velocity has nothing to invert. If this
            # axis's best candidate is an all-swing formula it can still be scored,
            # and that is strictly better than reporting the swing Unknown: it is
            # one foul ball in eight, and the whole of the last unclosed cell. The
            # formula is weaker evidence than the inversion, so it is used only
            # where the inversion has nothing, and it says so in its provenance.
            if missing.any() and spec.all_swing and (spec.coef or spec.stand_models):
                filled = _apply_formula(spec, feats)
                take = missing & np.isfinite(filled)
                vals[take] = filled[take]
                src[take] = "geometry"
                missing = missing & ~take
            src[missing] = "no_outcome"
        stats[f"no_outcome_{ax}"] = int((src == "no_outcome").sum())
        out[f"{ax}_hat"] = vals
        out[f"source_{ax}"] = src
    out.attrs["reconstruct_stats"] = stats
    return out
