"""Heatmaps over the contact point and timing, and the validation figures.

Colors: one diverging pair with a neutral midpoint for run value (the quantity
has a real zero and a sign), and the same two hues as the only two series on the
comparison figures. Validated with the dataviz skill's checker against the light
surface: worst adjacent CVD delta E 21.1 (protan), normal-vision 28.7, contrast
above 3:1 on both. Category names on every axis are Jeremy's words.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .calibrate import BINS, to_bin  # noqa: E402

AXIS_LABEL = {
    "x": "along the bat (in): tied up  |  centered  |  flail",
    "y": "timing (ms): late  |  on time  |  early",
    "z": "above the swing plane (in): over  |  lined up  |  under",
}
THRESH = {"x": 4.0, "y": 7.0, "z": 2.0}
NEG, MID, POS = "#b2182b", "#efefee", "#2166ac"
CMAP = LinearSegmentedColormap.from_list("rv", [NEG, MID, POS])
GRID = "#d8d8d4"
DPI = 140   # embedded in the notebook as base64; 200 dpi pushes the .ipynb past 16 MB
INK = "#3d3d3a"


def _dress(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def _grid(cal, df, axes, value, min_n):
    a, b = axes
    t = pd.DataFrame({"ba": to_bin(a, cal[f"{a}_cal"].to_numpy(float)),
                      "bb": to_bin(b, cal[f"{b}_cal"].to_numpy(float)),
                      "v": df[value].to_numpy(float)}).dropna()
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(["ba", "bb"])["v"].agg(["mean", "size"]).reset_index()
    g.loc[g["size"] < min_n, "mean"] = np.nan
    return g.pivot(index="bb", columns="ba", values="mean")


def heatmap(cal, df, axes=("x", "y"), value="delta_run_exp", min_n=50, title=None,
            ax=None, vmax=None, cbar=True):
    a, b = axes
    piv = _grid(cal, df, axes, value, min_n)
    ax = ax or plt.gca()
    _dress(ax)
    if piv.empty or not np.isfinite(piv.to_numpy()).any():
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no cell reached the minimum count", ha="center", va="center",
                transform=ax.transAxes, color=INK)
        return ax
    lim = vmax or float(np.nanmax(np.abs(piv.to_numpy())))
    lim = lim if lim > 0 else 0.1
    x_edges = np.append(piv.columns.to_numpy(float), piv.columns.max() + BINS[a])
    y_edges = np.append(piv.index.to_numpy(float), piv.index.max() + BINS[b])
    m = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(piv.to_numpy()), cmap=CMAP,
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), shading="flat")
    for t in (-THRESH[a], THRESH[a]):
        ax.axvline(t, color=INK, lw=0.9, ls="--", alpha=0.8)
    for t in (-THRESH[b], THRESH[b]):
        ax.axhline(t, color=INK, lw=0.9, ls="--", alpha=0.8)
    ax.set_xlabel(AXIS_LABEL[a], fontsize=9, color=INK)
    ax.set_ylabel(AXIS_LABEL[b], fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    if cbar:
        cb = plt.colorbar(m, ax=ax)
        cb.set_label(value, fontsize=9, color=INK)
        cb.ax.tick_params(labelsize=8, colors=INK)
    return ax


def heatmap_grid(cal, df, value, out_png: Path, title: str, min_n=50) -> Path:
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, pair in zip(axs, (("x", "y"), ("x", "z"), ("y", "z"))):
        heatmap(cal, df, pair, value, min_n=min_n, ax=ax)
    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_png


def _ols(x: pd.Series, y: pd.Series):
    """Least-squares fit of ours on Savant, pairwise-complete.

    The slope is the number the scatter is really about: with a noisy per-swing
    estimate it sits below 1 no matter how well calibrated the league marginal
    is, because within-player error widens every player toward the league mean.
    Reporting it next to r keeps that visible instead of inferred from the cloud.
    """
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3:
        return None, None
    a, b = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    if np.ptp(a) == 0:
        return None, None
    slope, intercept = np.polyfit(a, b, 1)
    return float(slope), float(intercept)


def rate_scatter(ours: pd.DataFrame, sav: pd.DataFrame, rate: str, out_png: Path,
                 min_swings=100) -> Path:
    m = ours.merge(sav, on="id", suffixes=("_ours", "_sav"))
    swings = "n_swings_sav" if "n_swings_sav" in m else "n_swings"
    m = m[m[swings] >= min_swings]
    fig, ax = plt.subplots(figsize=(6, 6))
    _dress(ax)
    ax.scatter(m[f"{rate}_sav"], m[f"{rate}_ours"], s=12, alpha=0.55, color=POS,
               edgecolors="none", zorder=3)
    vals = pd.concat([m[f"{rate}_sav"], m[f"{rate}_ours"]]).dropna()
    lo, hi = float(vals.min()), float(vals.max())
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, zorder=4, label="perfect agreement")
    slope, intercept = _ols(m[f"{rate}_sav"], m[f"{rate}_ours"])
    if slope is not None:
        ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
                color=NEG, lw=1.4, ls="--", zorder=5,
                label=f"fit: slope {slope:.2f}")
    ax.set_xlabel(f"Savant {rate}", color=INK)
    ax.set_ylabel(f"ours {rate}", color=INK)
    r = m[f"{rate}_sav"].corr(m[f"{rate}_ours"])
    title = f"{rate}: r = {r:.3f}, n = {len(m)}"
    if slope is not None:
        title += f", slope = {slope:.3f}"
    ax.set_title(title, fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    return out_png


def _bulk_range(t: pd.DataFrame, axis: str, keep=0.999):
    """The bin range holding the central mass, on whichever side has counts.

    Savant's own published x histogram carries bins at -16,156 and +4,092 inches
    holding one swing each. Drawn to scale they collapse the entire distribution
    into a single pixel column, so the panel is limited to the bulk.
    """
    lo, hi = np.inf, -np.inf
    for col in ("n_sav", "n_ours"):
        if col not in t:
            continue
        h = t[["bin", col]].dropna().sort_values("bin")
        if h.empty or h[col].sum() <= 0:
            continue
        cum = h[col].cumsum() / h[col].sum()
        edge = (1.0 - keep) / 2.0
        inside = h[(cum > edge) & (cum - h[col] / h[col].sum() < 1.0 - edge)]
        if len(inside):
            lo = min(lo, float(inside["bin"].min()))
            hi = max(hi, float(inside["bin"].max()))
    if not np.isfinite(lo):
        return None
    pad = BINS[axis] * 2
    return lo - pad, hi + pad


def league_bins_figure(table: pd.DataFrame, out_png: Path, min_n=100) -> Path:
    """Counts on top, run value below, one column per axis.

    Never a twin y axis: the plan drew counts and run value on two scales in one
    frame, which makes the two lines' crossings meaningless. Two stacked panels
    share the bin axis and each keeps its own honest scale.

    The run-value panel drops bins under min_n swings on that side. A mean run
    value over three swings is noise, and drawn as a line it reads as signal.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 9), sharex="col")
    for col, axis in enumerate(("x", "y", "z")):
        t = table[table.axis == axis].sort_values("bin")
        w = BINS[axis]
        top, bot = axs[0][col], axs[1][col]
        _dress(top)
        _dress(bot)
        top.bar(t["bin"] + w * 0.05, t["n_sav"], width=w * 0.42, color=NEG, align="edge",
                label="Savant", zorder=3)
        top.bar(t["bin"] + w * 0.52, t["n_ours"], width=w * 0.42, color=POS, align="edge",
                label="ours", zorder=3)
        top.set_ylabel("swings in bin", fontsize=9, color=INK)
        top.set_title(f"axis {axis}", fontsize=11, color=INK)
        top.legend(frameon=False, fontsize=9)
        for ncol, rcol, color, lbl in (("n_sav", "rv_sav", NEG, "Savant"),
                                       ("n_ours", "rv_ours", POS, "ours")):
            q = t[t[ncol].fillna(0) >= min_n]
            bot.plot(q["bin"], q[rcol], color=color, lw=2, label=lbl, zorder=3)
        bot.axhline(0.0, color=INK, lw=0.8, alpha=0.6)
        bot.set_ylabel(f"mean delta_run_exp (bins of {min_n}+ swings)", fontsize=9, color=INK)
        bot.set_xlabel(AXIS_LABEL[axis], fontsize=9, color=INK)
        bot.legend(frameon=False, fontsize=9)
        rng = _bulk_range(t, axis)
        if rng:
            top.set_xlim(*rng)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    return out_png


def tails_scatter(pred: dict, truth: dict, out_png: Path, caveat: str) -> Path:
    """Our reconstructed value against Savant's, on the labeled whiff tails."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, axis in zip(axs, ("x", "y", "z")):
        p, t = np.asarray(pred[axis], float), np.asarray(truth[axis], float)
        ok = np.isfinite(p) & np.isfinite(t)
        _dress(ax)
        ax.scatter(t[ok], p[ok], s=8, alpha=0.35, color=POS, edgecolors="none", zorder=3)
        if ok.sum() > 2:
            lo = float(min(t[ok].min(), p[ok].min()))
            hi = float(max(t[ok].max(), p[ok].max()))
            ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, zorder=4, label="perfect agreement")
            r = float(np.corrcoef(t[ok], p[ok])[0, 1])
            slope, intercept = _ols(pd.Series(t[ok]), pd.Series(p[ok]))
            title = f"axis {axis}: r = {r:.3f}, n = {int(ok.sum())}"
            if slope is not None:
                ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
                        color=NEG, lw=1.4, ls="--", zorder=5,
                        label=f"fit: slope {slope:.2f}")
                title += f", slope = {slope:.3f}"
            ax.set_title(title, fontsize=11, color=INK)
            ax.legend(frameon=False, fontsize=9)
        ax.set_xlabel(f"Savant sav_{axis}", color=INK)
        ax.set_ylabel(f"ours {axis}_hat", color=INK)
    fig.suptitle(caveat, fontsize=10, color=INK)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)
    return out_png
