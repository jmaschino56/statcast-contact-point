"""Center and scale every axis so per-player histograms match Savant's; fallback C for unrecovered axes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FIT_SEASON
from .discover import THRESH
from .savant_truth import BIN_WIDTH as BINS  # one definition, shared with the loaders
from .savant_truth import bin_of
from .savant_truth import load_histograms, load_leaderboard  # noqa: F401  (re-exported)

CONTACT_TYPES = ("in_play", "foul", "whiff")
MIN_PLAYER_SWINGS = 25


def to_bin(axis: str, values) -> np.ndarray:
    """Savant's bin, named by its lower edge. Float, so NaN survives.

    The grid lives in savant_truth next to the loader that puts Savant's own bins
    on it, because the two have to agree exactly.
    """
    return bin_of(axis, values)


TRIM = 5e-4   # mass trimmed from each tail of a target histogram before mapping


def trim_tails(target_hist: pd.Series, trim: float = TRIM) -> pd.Series:
    """Drop the target histogram's extreme singleton bins.

    Savant's own published x histogram carries bins at -16,156 and +4,092 inches
    holding one swing each. A quantile map built straight onto that grid sends our
    lowest-ranked swings to the lowest edge, which put our avg_x_tied_up at -109
    inches against Savant's -6.9. The map is defined on the bulk and clamps
    outside it instead.
    """
    h = target_hist.sort_index()
    total = h.sum()
    if total <= 0:
        return h
    cum = h.cumsum() / total
    before = cum - h / total
    return h[(cum > trim) & (before < 1.0 - trim)]


class QuantileMap:
    """Monotone piecewise-linear map from a predicted distribution onto a target histogram."""

    def fit(self, pred_values, target_hist: pd.Series, axis: str | None = None):
        p = np.asarray(pred_values, float)
        p = np.sort(p[np.isfinite(p)])
        if p.size == 0:
            raise ValueError("no finite predictions to calibrate")
        h = trim_tails(target_hist)
        h = h[h > 0]
        edges = h.index.to_numpy(float)
        if axis is not None:
            width = float(grid_widths(axis, edges)[-1])
        else:
            width = float(np.min(np.diff(edges))) if len(edges) > 1 else 1.0
        cdf = np.concatenate([[0.0], np.cumsum(h.to_numpy(float)) / h.sum()])
        grid = np.concatenate([edges, [edges[-1] + width]])
        q = np.linspace(0, 1, 201)
        self.src = np.quantile(p, q)
        self.dst = np.interp(q, cdf, grid)
        return self

    def transform(self, values):
        v = np.asarray(values, float)
        out = np.interp(v, self.src, self.dst, left=self.dst[0], right=self.dst[-1])
        return np.where(np.isfinite(v), out, np.nan)


def _hist_for(hist: pd.DataFrame, axis: str, ids=None) -> pd.Series:
    h = hist[hist.axis == axis]
    if ids is not None:
        h = h[h.id.isin(ids)]
    return h.groupby("bin")["n"].sum()


def wasserstein_per_player(values: np.ndarray, df: pd.DataFrame, hist: pd.DataFrame,
                           axis: str, key: str) -> pd.Series:
    """Per player, the 1-Wasserstein distance in axis units between our binned
    distribution and Savant's published one."""
    ours = pd.DataFrame({"id": df[key].to_numpy(), "bin": to_bin(axis, values)}).dropna()
    mine_by_id = {pid: g["bin"].value_counts() for pid, g in ours.groupby("id")}
    out = {}
    for pid, h in hist[hist.axis == axis].groupby("id"):
        mine = mine_by_id.get(pid)
        if mine is None or mine.sum() < MIN_PLAYER_SWINGS:
            continue
        bins = np.union1d(h["bin"].to_numpy(float), mine.index.to_numpy(float))
        a = mine.reindex(bins).fillna(0).to_numpy() / mine.sum()
        b = h.set_index("bin")["n"].reindex(bins).fillna(0).to_numpy() / h["n"].sum()
        out[pid] = float(np.sum(np.abs(np.cumsum(a - b))) * BINS[axis])
    return pd.Series(out, name=f"w1_{axis}", dtype=float)


BAND_RATES = {"x": ("tied_up_percent", "centered_percent", "flailed_percent"),
              "y": ("late_percent", "on_time_percent", "early_percent"),
              "z": ("over_percent", "lined_up_percent", "under_percent")}


def grid_widths(axis: str, bins) -> np.ndarray:
    """Each bin's width on a grid that may have been refined at the thresholds.

    Savant's own grid is uniform with gaps where a bin held nothing, so the nominal
    width is the answer there. per_type_target cuts a bin in two at a threshold, and
    those two are narrower than nominal; the distance to the next edge finds them.
    """
    w = BINS[axis]
    e = np.asarray(bins, float)
    if e.size < 2:
        return np.full(e.size, w)
    return np.minimum(np.append(np.diff(e), w), w)


def band_fractions(axis: str, bins, widths=None) -> np.ndarray:
    """Each bin's share of the low, middle and high category, as (n_bins, 3).

    The split is proportional at the thresholds discover.label_of uses, because
    label_of is what the validation labels with and the quantile map spreads a
    bin's mass uniformly across it. Where a threshold lands on a bin edge this is
    a whole-bin assignment and reproduces Savant's asymmetric edges exactly: on z
    bin -3 is over, bin -2 is lined up, bin 2 is under. Only y straddles, because
    its threshold of 7 ms is not a multiple of the 2 ms bin.
    """
    t = THRESH[axis]
    lower = np.asarray(bins, float)
    w = grid_widths(axis, lower) if widths is None else np.asarray(widths, float)
    lo = np.clip((-t - lower) / w, 0.0, 1.0)
    hi = np.clip((lower + w - t) / w, 0.0, 1.0)
    return np.column_stack([lo, np.clip(1.0 - lo - hi, 0.0, 1.0), hi])


def band_masses(hist: pd.Series, axis: str) -> np.ndarray:
    """The histogram's mass in each of the three categories."""
    h = hist.sort_index()
    return band_fractions(axis, h.index.to_numpy(float)).T @ h.to_numpy(float)


def per_type_target(pooled_hist: pd.Series, axis: str, rates) -> pd.Series:
    """The pooled histogram with its three band masses rescaled to one contact
    type's league rates, keeping the within-band shape and the total.

    Savant publishes no per-contact-type histogram, so the shape has to come from
    the pooled one. What it does publish per contact type is the split CSV's
    category rates, and those are the only thing that distinguishes a ball in play
    from a whiff. Rescaling the bands to them is the spec's map-per-contact-type
    with the target in rate form.

    The grid is cut at the two thresholds first. Without that, a bin straddling a
    threshold carries mass into two bands in a fixed ratio that no scaling can
    change, which puts a ceiling on what can be asked for: on the real z histogram
    the most lined-up mass reachable is 0.8979, and balls in play need 0.9751. Cut
    at the threshold, every bin sits in one band, the three scales are independent,
    and the result is exact and never negative.

    `rates` is a mapping with keys "lo", "mid" and "hi", normalised to sum to one.
    """
    h = pooled_hist.sort_index().astype(float)
    total = float(h.sum())
    want = np.array([float(rates["lo"]), float(rates["mid"]), float(rates["hi"])])
    if total <= 0 or want.sum() <= 0:
        raise ValueError("per_type_target needs a non-empty histogram and non-zero rates")
    want = want / want.sum()

    w, t = BINS[axis], THRESH[axis]
    edges, counts, widths = [], [], []
    for L, n in h.items():
        pts = [L] + sorted(c for c in (-t, t) if L < c < L + w) + [L + w]
        for a, b in zip(pts[:-1], pts[1:]):
            edges.append(a)
            widths.append(b - a)
            counts.append(n * (b - a) / w)
    edges = np.asarray(edges, float)
    counts = np.asarray(counts, float)
    f = band_fractions(axis, edges, widths)
    have = f.T @ counts
    for i, name in enumerate(("lo", "mid", "hi")):
        if want[i] > 0 and have[i] <= 0:
            raise ValueError(
                f"the pooled {axis} histogram has no mass in the {name} band, so a "
                f"target rate of {want[i]:.4f} cannot be built from its shape")
    scale = np.zeros(3)
    live = have > 0
    scale[live] = (want * total)[live] / have[live]
    out = (f * scale).sum(axis=1) * counts
    return pd.Series(out, index=edges, name=h.name)


def split_rates(season: int, type_: str = "pitcher") -> dict:
    """League category rates per contact type, from the split CSV, weighted by swings.

    Keyed (axis, contact_type) -> {"lo", "mid", "hi"}.
    """
    sp = load_leaderboard(season, type_, "bat_contact_code").dropna(subset=["id"])
    out = {}
    for ct, g in sp.groupby("contact_type"):
        w = g["n_swings"].to_numpy(float)
        if w.sum() <= 0:
            continue
        for ax, cols in BAND_RATES.items():
            out[(ax, ct)] = {k: float(np.average(g[c].to_numpy(float), weights=w))
                             for k, c in zip(("lo", "mid", "hi"), cols)}
    return out


def fit_calibration(pred: pd.DataFrame, df: pd.DataFrame, season: int,
                    type_: str = "pitcher", hist: pd.DataFrame | None = None,
                    per_contact_type: bool = True, rates: dict | None = None,
                    pooled_axes: tuple = ()) -> dict:
    """One quantile map per axis per contact type, onto that type's own target.

    Savant publishes no per-contact-type histogram, so an earlier version fit each
    contact type onto the all-swing marginal, which forces every type to the
    all-swing rate and erases the contrast spec section 6 item 3 calls the sharpest
    test. That is why the pooled map was the default. What Savant does publish per
    contact type is the split CSV's category rates, and per_type_target turns those
    plus the pooled shape into a real per-type target.

    pooled_axes names axes that keep the single pooled map even here, for an axis
    whose holdout rates the per-type map makes worse. per_contact_type=False keeps
    the whole pooled path available so the notebook can show the comparison.
    """
    assert season == FIT_SEASON, "calibration fits on the development season only"
    if hist is None:
        hist = load_histograms(season, type_)
    if per_contact_type and rates is None:
        rates = split_rates(season, type_)
    maps = {}
    for ax in ("x", "y", "z"):
        pooled = _hist_for(hist, ax)
        if pooled.empty:
            continue
        vals = pred[f"{ax}_hat"].to_numpy(float)
        if not per_contact_type or ax in pooled_axes:
            if np.isfinite(vals).any():
                maps[(ax, "all")] = QuantileMap().fit(vals, pooled, axis=ax)
            continue
        for ct in CONTACT_TYPES:
            r = rates.get((ax, ct))
            sel = (df["contact_type"] == ct).to_numpy()
            if r is None or not np.isfinite(vals[sel]).any():
                continue
            maps[(ax, ct)] = QuantileMap().fit(vals[sel], per_type_target(pooled, ax, r), axis=ax)
    return maps


def apply_calibration(pred: pd.DataFrame, df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=pred.index)
    for ax in ("x", "y", "z"):
        vals = pred[f"{ax}_hat"].to_numpy(float).copy()
        if (ax, "all") in maps:
            sel = np.isfinite(vals)
            vals[sel] = maps[(ax, "all")].transform(vals[sel])
        else:
            for ct in CONTACT_TYPES:
                if (ax, ct) not in maps:
                    continue
                sel = (df["contact_type"] == ct).to_numpy() & np.isfinite(vals)
                vals[sel] = maps[(ax, ct)].transform(vals[sel])
        out[f"{ax}_cal"] = vals
    return out


RELIABILITY_FLOOR = 50     # swings a player needs before their own mean is used
RELIABILITY_RATE_MIN = 100  # swings a player needs to enter the slope fit, as in compare_rates
MID_RATE = {"x": "centered_percent", "y": "on_time_percent", "z": "lined_up_percent"}


def apply_reliability(values, player_ids, k: float, min_swings: int = RELIABILITY_FLOOR):
    """Scale each swing about its own player's season mean by k.

    Per-swing error inflates every player's spread, the quantile map pins the
    pooled marginal, and what gives way is the between-player spread: a pitcher
    Savant has at 0.75 lined up comes out near 0.68 and one at 0.45 comes out near
    0.48. Narrowing each player about their own mean undoes that. It polarises the
    rates rather than shifting them, because a player centred inside the band keeps
    more of their swings and one centred outside keeps fewer, and no player's mean
    moves at all.

    A player under the floor is left alone. Their mean is too noisy to scale about,
    and Savant's leaderboard has no qualifier of its own to borrow: the endpoint is
    called with min=1 and returns players with a single swing.
    """
    v = np.asarray(values, float)
    ids = pd.Series(np.asarray(player_ids))
    g = pd.Series(v).groupby(ids)
    mu = g.transform("mean").to_numpy()
    n = g.transform("size").to_numpy()
    use = (n >= min_swings) & np.isfinite(mu)
    return np.where(use, mu + k * (v - mu), v)


def fit_reliability(pred: pd.DataFrame, df: pd.DataFrame, season: int = FIT_SEASON,
                    type_: str = "pitcher", min_swings: int = RELIABILITY_FLOOR,
                    savant: pd.DataFrame | None = None) -> dict:
    """One scalar per axis: the k that makes our per-player rate track Savant's
    one for one.

    pred carries the CALIBRATED values, so this runs after the quantile map and
    before categorize. k is chosen so the least squares slope of our rate on
    Savant's is 1 over the players with enough swings to be in the validation.
    Fit on the development season only and applied unchanged to the holdouts.
    """
    assert season == FIT_SEASON, "the reliability factor is fit on the development season only"
    if savant is None:
        savant = load_leaderboard(season, type_).dropna(subset=["id"]).astype({"id": int})
    ids = df[type_].to_numpy()
    out = {}
    for ax in ("x", "y", "z"):
        col = f"{ax}_cal"
        rate_col = MID_RATE[ax]
        if col not in pred or rate_col not in savant or not np.isfinite(pred[col]).any():
            continue
        v = pred[col].to_numpy(float)
        t = THRESH[ax]

        def slope(k: float) -> float:
            w = apply_reliability(v, ids, k, min_swings)
            ours = (pd.DataFrame({"id": ids, "r": np.abs(w) <= t, "ok": np.isfinite(w)})
                    .groupby("id").agg(r=("r", "mean"), n=("ok", "sum")).reset_index())
            m = ours.merge(savant[["id", rate_col, "n_swings"]], on="id", how="inner")
            m = m[m["n_swings"] >= RELIABILITY_RATE_MIN].dropna(subset=[rate_col])
            if len(m) < 3:
                return np.nan
            a = m[rate_col].to_numpy(float)
            b = m["r"].to_numpy(float)
            return float(np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1))

        lo, hi = 0.2, 3.0
        f_lo, f_hi = slope(lo) - 1.0, slope(hi) - 1.0
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            grid = np.linspace(lo, hi, 57)
            vals = np.array([slope(g) for g in grid])
            out[ax] = float(grid[int(np.nanargmin(np.abs(vals - 1.0)))])
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = slope(mid) - 1.0
            if not np.isfinite(f_mid):
                break
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
            if hi - lo < 1e-4:
                break
        out[ax] = float(0.5 * (lo + hi))
    return out


def selection_weights(values: np.ndarray, ids: np.ndarray, hist: pd.DataFrame,
                      axis: str) -> np.ndarray:
    """Inverse-selection weights for tail rows: n_region / 10.

    n_region is that player's published count of swings at or beyond the row's
    value in the tail direction, and Savant published the 10 worst, so a row
    drawn from a region holding n swings stands in for n / 10 of them.
    """
    h = hist[hist.axis == axis]
    by_id = {pid: (g["bin"].to_numpy(float), g["n"].to_numpy(float)) for pid, g in h.groupby("id")}
    b = to_bin(axis, values)
    w = np.ones(len(values))
    for i, (bi, pid) in enumerate(zip(b, ids)):
        hp = by_id.get(pid)
        if hp is None or not np.isfinite(bi):
            continue
        bins, ns = hp
        region = ns[bins >= bi] if values[i] >= 0 else ns[bins <= bi]
        w[i] = max(region.sum(), 1.0) / 10.0
    return w


def fallback_fit(axis: str, tails_feats: pd.DataFrame, hist: pd.DataFrame,
                 feature_cols: list, key: str = "batter"):
    import lightgbm as lgb
    f = tails_feats.dropna(subset=list(feature_cols) + [f"sav_{axis}"])
    w = selection_weights(f[f"sav_{axis}"].to_numpy(float), f[key].to_numpy(), hist, axis)
    ds = lgb.Dataset(f[feature_cols].to_numpy(float), f[f"sav_{axis}"].to_numpy(float), weight=w)
    return lgb.train({"objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
                      "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                      "bagging_freq": 1, "verbose": -1}, ds, 400)


def fallback_predict(model, feats: pd.DataFrame, feature_cols: list) -> np.ndarray:
    return model.predict(feats[list(feature_cols)].to_numpy(float))
