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
    # Savant's own sign and ordering. The word "ball" is load-bearing: the
    # NUMBER is the ball's height above the swing plane while the LABEL is what
    # the bat did, and they point opposite ways. A ball above the plane means
    # the bat passed under it, so "under" is the POSITIVE end. Savant's league
    # page publishes avg_z_over = -3.78 and avg_z_under = +2.99, and getting
    # under the ball is what lifts it: mean launch angle is +22 degrees at
    # z = +0.75 and -6 degrees at z = -0.75.
    "z": "ball above the swing plane (in): over  |  lined up  |  under",
}

# Every axis is drawn on Savant's own sign, so a figure and a validation table
# can never disagree. The machinery stays because the bin-mirroring rule in
# to_display_bin is easy to get wrong and this is where it is written down;
# flipping one entry to -1.0 is the whole change if an axis ever needs it.
DISPLAY_SIGN = {"x": 1.0, "y": 1.0, "z": 1.0}


def to_display(axis: str, values):
    """Savant's stored sign turned into the sign the figures are drawn in.

    For raw VALUES only. A bin LABEL is its lower edge, and negating a lower
    edge gives an upper edge, which lands the bin a full width away from the
    value it is supposed to hold. Bin a displayed value, or use to_display_bin.
    """
    return np.asarray(values, float) * DISPLAY_SIGN[axis]


def to_display_bin(axis: str, lower_edges):
    """Flip a bin named by its LOWER EDGE, keeping the bin over its own data.

    Bin [lo, lo+w) mirrors to (-lo-w, -lo], whose lower edge is -lo-w. Dropping
    the width shifted every z cell one whole inch, which put the peak of the
    xwOBA ridge in the wrong bin and made the 3D figures disagree with the
    hexbins, which flip the value and never had the bug.
    """
    lo = np.asarray(lower_edges, float)
    if DISPLAY_SIGN[axis] > 0:
        return lo
    return -lo - BINS[axis]
THRESH = {"x": 4.0, "y": 7.0, "z": 2.0}
NEG, MID, POS = "#b2182b", "#efefee", "#2166ac"
# Savant's polarity: cold blue at the bottom of the scale, hot red at the top.
CMAP = LinearSegmentedColormap.from_list("rv", [POS, MID, NEG])

# One fixed scale per quantity, shared by every panel and every figure that
# draws it. A per-panel scale fitted to its own data makes two panels look
# alike when their cells are a quarter of a run apart, and the eye cannot
# carry a reading from one figure to the next. Fixed limits cost some contrast
# inside a single panel and buy comparability everywhere.
SCALES = {
    "delta_run_exp": (-0.3, 0.0, 0.3),
    "estimated_woba_using_speedangle": (0.0, 0.6, 1.2),
}
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
    t = pd.DataFrame({"ba": to_bin(a, to_display(a, cal[f"{a}_cal"].to_numpy(float))),
                      "bb": to_bin(b, to_display(b, cal[f"{b}_cal"].to_numpy(float))),
                      "v": df[value].to_numpy(float)}).dropna()
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(["ba", "bb"])["v"].agg(["mean", "size"]).reset_index()
    g.loc[g["size"] < min_n, "mean"] = np.nan
    return g.pivot(index="bb", columns="ba", values="mean")


def heatmap(cal, df, axes=("x", "y"), value="delta_run_exp", min_n=50, title=None,
            ax=None, vmax=None, vcenter=0.0, cbar=True):
    """Mean of `value` over Savant's own rectangular bins.

    `vcenter` is where the diverging colormap puts its white. Zero is right for
    run value, which is signed about zero. Pass None for a quantity that is
    not, such as xwOBA: every xwOBA sits between 0.1 and 0.9, so centring at
    zero paints the whole panel in one half of the map and throws the red away.
    """
    a, b = axes
    piv = _grid(cal, df, axes, value, min_n)
    ax = ax or plt.gca()
    _dress(ax)
    if piv.empty or not np.isfinite(piv.to_numpy()).any():
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no cell reached the minimum count", ha="center", va="center",
                transform=ax.transAxes, color=INK)
        return ax
    centre, lo, hi = _scale_for(value, piv.to_numpy().ravel(), vcenter)
    if vmax:
        lo, hi = centre - vmax, centre + vmax
    x_edges = np.append(piv.columns.to_numpy(float), piv.columns.max() + BINS[a])
    y_edges = np.append(piv.index.to_numpy(float), piv.index.max() + BINS[b])
    m = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(piv.to_numpy()), cmap=CMAP,
                      norm=TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi),
                      shading="flat")
    # A fixed corner here, not "best": the mesh fills the panel, so the overlap
    # search that places the hexbin legends has nothing to find.
    panel_legend(ax, threshold_lines(ax, (a, b)), loc="upper left")
    ax.set_xlabel(AXIS_LABEL[a], fontsize=9, color=INK)
    ax.set_ylabel(AXIS_LABEL[b], fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    if cbar:
        cb = plt.colorbar(m, ax=ax)
        cb.set_label(value, fontsize=9, color=INK)
        cb.ax.tick_params(labelsize=8, colors=INK)
    return ax


def heatmap_grid(cal, df, value, out_png: Path, title: str, min_n=50,
                 vcenter=0.0) -> Path:
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, pair in zip(axs, (("x", "y"), ("x", "z"), ("y", "z"))):
        heatmap(cal, df, pair, value, min_n=min_n, ax=ax, vcenter=vcenter)
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


# A 34 in MLB bat with its sweet spot 6 in from the barrel end, which is where
# exit velocity actually peaks: 100.3 mph at x = +0.5, falling to 87 by +/-2.5
# and 75 by +/-6 (2025 balls in play). So the tip sits at x = +6 and the knob at
# x = -28, and the drawing is to scale on the x axis.
#
# It is worth knowing what the overlay reveals rather than hides: x is NOT
# symmetric. Nothing reaches the knob (0.0 percent of swings below -28, and the
# 1st percentile is only -8.5), while 8.6 percent of balls in play and 32.8
# percent of whiffs land PAST the tip. Contact cannot work further down than the
# hands, but a swing can miss by any distance off the end, which is what the
# flail tail is.
# The panels hexbin_grid draws, as (horizontal, vertical, one-scale). Shared
# with the tests on purpose: three tests once passed while describing a layout
# that had already been replaced, because each built its own panels.
# (horizontal, vertical, one-scale, carries a bat)
PANELS = (("x", "z", True, True), ("x", "y", False, True), ("y", "z", False, False))

# What Savant's own threshold on each axis is called, for the legend. The lines
# were unlabelled and three different dashed styles shared a single caption.
MID_NAME = {"x": "centered", "y": "on time", "z": "lined up"}
AXIS_UNIT = {"x": "in", "y": "ms", "z": "in"}
AXIS_OF = {"x": " along the bat", "y": "", "z": " of the swing plane"}

BAT_TIP_X, BAT_LENGTH = 6.0, 34.0
BAT_R, BALL_R = 1.30, 1.45      # barrel radius and a baseball's radius, inches
_BAT_PROFILE = (        # (inches from the knob end, radius in inches)
    (0.00, 0.10), (0.10, 0.80), (0.55, 0.98), (1.05, 0.90), (1.45, 0.58),
    (2.20, 0.48), (9.00, 0.46), (13.0, 0.50), (16.5, 0.62), (19.5, 0.85),
    (22.5, 1.12), (25.5, 1.27), (31.0, 1.30), (32.8, 1.26), (33.5, 1.12),
    (34.0, 0.10),
)
# Where the bat sits on a panel whose other axis is NOT inches. A fraction of
# the axes height, since milliseconds have no bat in them.
BAT_FRAC_HALF = 0.055


def bat_outline(n: int = 600):
    """(x along the bat, radius in inches) for one bat silhouette.

    Anchored so the sweet spot is x = 0, which is where exit velocity actually
    peaks: 100.3 mph at +0.5, 87 by 2.5 in either way, 75 by 6 (2025 balls in
    play). That puts the tip at +6 and the knob at -28.
    """
    s_pts = np.array([p[0] for p in _BAT_PROFILE], float)
    r_pts = np.array([p[1] for p in _BAT_PROFILE], float)
    s = np.linspace(0.0, BAT_LENGTH, n)
    return s - (BAT_LENGTH - BAT_TIP_X), np.interp(s, s_pts, r_pts)


def _isotropic_half(ax, fallback=BAT_FRAC_HALF):
    """Axes-fraction half-thickness that makes the bat look to scale.

    The panel's own aspect decides it: one data inch on x is some number of
    figure inches, and the bat's half-radius has to be that same number of
    figure inches expressed as a fraction of the panel's height.
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    x_lo, x_hi = ax.get_xlim()
    if w_in <= 0 or h_in <= 0 or x_hi <= x_lo:
        return fallback
    fig_in_per_data_in = w_in / (x_hi - x_lo)
    return float(BAT_R * fig_in_per_data_in / h_in)


def draw_bat(ax, axes_pair=("x", "y"), true_scale=False, color="#2f2a22"):
    """Put the bat on the panel, and report what it drew for the legend.

    `true_scale` says the panel's two axes share one scale, which only the x/z
    panel can: both of its axes are inches. There the bat goes on in DATA units,
    a real 34 by 2.6 in, and its own edges ARE the barrel. Everywhere else the
    other axis is milliseconds, so the silhouette is drawn isotropically and is
    a ruler for x alone; its thickness is drawing, not data.

    The dashed line is +/-(BAT_R + BALL_R) = 2.75 in, the furthest a ball's
    centre can sit and still touch the barrel. 98.5 percent of balls in play
    fall inside it, against 76.8 of fouls and 53.9 of whiffs, and nothing in the
    reconstruction was ever told a bat exists.
    """
    from matplotlib.lines import Line2D
    a, b = axes_pair
    reach = BAT_R + BALL_R
    strokes = ((4.0, "#ffffff", 0.85, 7), (2.0, color, 1.0, 8))
    keys = []
    if a == "x":
        x, r = bat_outline()
        if true_scale:
            up, dn, trans = r, -r, ax.transData
            lbl = f"bat, {BAT_LENGTH:.0f} x {2 * BAT_R:.1f} in to scale"
        else:
            lo, hi = ax.get_ylim()
            mid = (0.0 - lo) / (hi - lo) if hi > lo else 0.5
            k = _isotropic_half(ax) / r.max()
            up, dn, trans = mid + r * k, mid - r * k, ax.get_xaxis_transform()
            lbl = f"bat, {BAT_LENGTH:.0f} in long (thickness not to scale)"
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([up, dn[::-1]])
        for lw, c, al, z in strokes:
            ax.plot(xs, ys, color=c, lw=lw, alpha=al, solid_joinstyle="round",
                    zorder=z, transform=trans, clip_on=True)
        keys.append((Line2D([], [], color=color, lw=2.0), lbl))
    if b == "z":
        if not true_scale:
            for edge in (-BAT_R, BAT_R):
                for lw, c, al, z in strokes:
                    ax.axhline(edge, color=c, lw=lw, alpha=al, zorder=z)
            keys.append((Line2D([], [], color=color, lw=2.0),
                         f"barrel edge, +/-{BAT_R:.2f} in "
                         f"({2 * BAT_R:.1f} in deep)"))
        for t in (-reach, reach):
            ax.axhline(t, color=color, lw=1.1, ls=(0, (1, 1.8)), alpha=0.9, zorder=6)
        keys.append((Line2D([], [], color=color, lw=1.4, ls=(0, (1, 1.8))),
                     f"furthest a ball can still touch the barrel, {reach:.2f} in"))
    return keys


def threshold_lines(ax, axes_pair):
    """Draw Savant's own category boundaries and name each one.

    These were three unlabelled dashes sharing one caption, and the reader had
    no way to tell a Savant threshold from the bat's reach. Each line now comes
    back with the words Savant uses for the band it closes and the number it
    sits at, and the caller puts them in a legend.
    """
    from matplotlib.lines import Line2D
    keys = []
    for axis, draw in zip(axes_pair, (ax.axvline, ax.axhline)):
        for t in (-THRESH[axis], THRESH[axis]):
            draw(t, color=INK, lw=1.0, ls=(0, (6, 4)), alpha=0.85, zorder=3)
        keys.append((Line2D([], [], color=INK, lw=1.0, ls=(0, (6, 4))),
                     f"Savant calls it {MID_NAME[axis]}: within +/-"
                     f"{THRESH[axis]:g} {AXIS_UNIT[axis]}{AXIS_OF[axis]}"))
    return keys


def panel_legend(ax, keys, loc="best"):
    """One legend per panel, built from the lines that panel actually drew.

    `loc="best"` rather than a fixed corner: each panel's cloud has a different
    shape, and the hexbin is a PolyCollection carrying offsets, so matplotlib's
    own overlap search puts the box in whichever corner holds the fewest cells.
    """
    if not keys:
        return None
    handles, labels = zip(*keys)
    leg = ax.legend(handles, labels, loc=loc, fontsize=7.5, framealpha=0.85,
                    facecolor="white", edgecolor="#cccccc", borderpad=0.5,
                    labelspacing=0.45, handlelength=2.6)
    leg.set_zorder(12)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _diverging_limits(vals, vcenter=0.0, weights=None, keep=1.0):
    """Where the colormap puts its white, and how far each half reaches.

    A signed quantity keeps `vcenter` at zero and a SYMMETRIC reach, so a cell
    half a run better and one half a run worse get the same colour intensity.

    An unsigned one passes vcenter=None. White then moves to the mean and each
    half reaches only as far as the data does, because xwOBA lives between 0.00
    and 0.92: a symmetric reach around a mean of 0.366 would run the red half
    down to -0.18 and throw away a quarter of the colour range on values no
    swing can produce.
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0 if vcenter is None else float(vcenter)), -0.1, 0.1
    if vcenter is None:
        centre = float(np.average(vals, weights=weights) if weights is not None
                       else vals.mean())
        edge = (1.0 - keep) / 2.0
        lo, hi = ((float(vals.min()), float(vals.max())) if keep >= 1.0
                  else tuple(float(q) for q in np.quantile(vals, [edge, 1.0 - edge])))
        return centre, min(lo, centre - 1e-6), max(hi, centre + 1e-6)
    centre = float(vcenter)
    dev = np.abs(vals - centre)
    # A robust reach, not the maximum. Over all swings 91.8 percent of the run
    # value cells sit within a quarter of max|v - 0|, because a handful of
    # extreme cells set the scale; clipping at the keep quantile puts the median
    # cell at 40 percent of the bar instead of 14, and saturates the few.
    lim = float(np.quantile(dev, keep) if keep < 1.0 else dev.max()) or 0.1
    return centre, centre - lim, centre + lim


def _fade(values, keep=0.98, floor=0.03, span=0.97, power=1.8):
    """Opacity and a 0-to-1 weight per cell, rising with the VALUE itself.

    Monotone, not symmetric about a centre. The reader is looking for where the
    good contact is, so the best cells are opaque and full size and the worst
    fade out of the way. An earlier version ramped on distance from the league
    mean, which made the worst cells as loud as the best and buried the ridge
    inside an opaque shell of ordinary blue.

    Normalised on the data's own robust range rather than the fixed colour
    scale, so the reddest cell PRESENT reaches full opacity even when the scale
    leaves headroom above it (xwOBA tops out near 0.92 on a bar drawn to 1.2).
    Colour stays comparable between figures; opacity is a within-figure aid.
    """
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, floor), np.zeros(v.shape)
    edge = (1.0 - keep) / 2.0
    lo, hi = np.quantile(finite, [edge, 1.0 - edge])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    u = np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full(v.shape, 0.5)
    return floor + span * u ** power, u


def _scale_for(value, vals, vcenter=0.0, weights=None, keep=1.0):
    """The colour limits for `value`: its fixed scale when it has one.

    Falls back to _diverging_limits for a quantity SCALES does not name, so a
    new metric still draws something sensible instead of raising.
    """
    fixed = SCALES.get(value)
    if fixed is not None:
        lo, centre, hi = (float(f) for f in fixed)
        return centre, lo, hi
    return _diverging_limits(vals, vcenter, weights=weights, keep=keep)


def _bulk_extent(v: np.ndarray, axis: str, keep=0.998):
    """Axis limits holding the central mass, padded by one Savant bin.

    The reconstructed axes have long thin tails (a swing can be scored 60
    inches off the sweet spot). Letting them set the extent collapses the
    hexagons that hold 99 percent of the swings into a smear.
    """
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    edge = (1.0 - keep) / 2.0
    lo, hi = np.quantile(v, [edge, 1.0 - edge])
    pad = BINS[axis]
    return float(lo) - pad, float(hi) + pad


def hexbin(cal, df, axes=("x", "y"), value="delta_run_exp", min_n=50, title=None,
           ax=None, gridsize=34, vmax=None, vcenter=0.0, cbar=True, bat=True,
           limits=None, equal=False):
    """Mean of `value` over hexagonal cells of the reconstructed plane.

    Hexagons rather than the Savant rectangles because these panels are a
    surface to read, not a comparison against Savant's own binning. A hexagon
    has six equidistant neighbours where a square has four plus four diagonals,
    so a smooth surface reads as smooth instead of as a staircase, and the same
    cell count covers the plane with less empty area.

    `vcenter` is where the diverging colormap puts its white; None centres it on
    the swing-weighted mean of `value`, which is what a quantity like xwOBA
    needs and zero would waste half the colour range on.
    """
    a, b = axes
    ax = ax or plt.gca()
    _dress(ax)
    x = to_display(a, cal[f"{a}_cal"].to_numpy(float))
    y = to_display(b, cal[f"{b}_cal"].to_numpy(float))
    c = df[value].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    # limits, when given, are computed once per QUANTITY by hexbin_grid, so the
    # same axis cannot come out a little different on two panels.
    ex = (limits or {}).get(a) or _bulk_extent(x[ok], a)
    ey = (limits or {}).get(b) or _bulk_extent(y[ok], b)
    if bat and a == "x" and ex is not None:
        # Hold the whole bat, knob included, or the silhouette is a cone with
        # its handle off the page. gridsize rises with the widened extent so the
        # hexagons over the data stay the size they were.
        wide = min(ex[0], -(BAT_LENGTH - BAT_TIP_X) - 1.5)
        if wide < ex[0]:
            gridsize = int(round(gridsize * (ex[1] - wide) / (ex[1] - ex[0])))
            ex = (wide, ex[1])
    if ok.sum() < min_n or ex is None or ey is None:
        ax.set_title(title or "")
        ax.text(0.5, 0.5, "no cell reached the minimum count", ha="center", va="center",
                transform=ax.transAxes, color=INK)
        return ax
    hb = ax.hexbin(x[ok], y[ok], C=c[ok], reduce_C_function=np.mean, mincnt=min_n,
                   gridsize=gridsize, extent=(ex[0], ex[1], ey[0], ey[1]),
                   cmap=CMAP, linewidths=0.0, zorder=2)
    vals = hb.get_array()
    vals = np.asarray(vals[np.isfinite(vals)], float)
    if vals.size:
        centre, lo, hi = _scale_for(value, vals, vcenter)
        if vmax:
            lo, hi = centre - vmax, centre + vmax
        hb.set_norm(TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi))
    keys = threshold_lines(ax, (a, b))
    ax.set_xlim(*ex)
    ax.set_ylim(*ey)
    if equal:
        # One scale for both axes, which only x/z can have: inches on each. It
        # is what lets the bat go on in data units at a real 34 by 2.6 in.
        ax.set_aspect("equal", adjustable="box")
    if bat:
        keys += draw_bat(ax, axes_pair=(a, b), true_scale=equal)
    panel_legend(ax, keys)
    ax.set_xlabel(AXIS_LABEL[a], fontsize=9, color=INK)
    ax.set_ylabel(AXIS_LABEL[b], fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    if cbar:
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(value, fontsize=9, color=INK)
        cb.ax.tick_params(labelsize=8, colors=INK)
    return ax


def hexbin_grid(cal, df, value, out_png: Path, title: str, min_n=50, gridsize=34,
                vcenter=0.0, bat=True) -> Path:
    """Three planes of the contact point, one scale per quantity.

    Each range is computed ONCE from the whole column and handed to every panel
    that carries it, so x, y and z look identical wherever they appear.

    The x/z panel additionally holds its two axes to ONE scale, because both are
    inches. That is what makes its bat a real 34 by 2.6 in shape rather than a
    drawing, and it makes the panel short and wide, so it takes a row of its own.
    """
    limits = {}
    for axis in ("x", "y", "z"):
        v = cal[f"{axis}_cal"].to_numpy(float)
        limits[axis] = _bulk_extent(v[np.isfinite(v)], axis)
    if bat:
        fig = plt.figure(figsize=(20, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5],
                              hspace=0.22, wspace=0.18)
        cells = (gs[0, :], gs[1, 0], gs[1, 1])
        panels = [(fig.add_subplot(c), (a, b), eq, has_bat)
                  for c, (a, b, eq, has_bat) in zip(cells, PANELS)]
    else:
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        panels = [(ax, (a, b), False, False) for ax, (a, b, _, _) in zip(axs, PANELS)]
    for ax, pair, eq, has_bat in panels:
        hexbin(cal, df, pair, value, min_n=min_n, ax=ax, gridsize=gridsize,
               vcenter=vcenter, bat=bat and has_bat, limits=limits, equal=eq)
    fig.suptitle(title, fontsize=13, color=INK)
    if not bat:
        fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_png


def cells3d(cal, df, value="estimated_woba_using_speedangle", min_n=25):
    """Mean of `value` per (x, y, z) cell on Savant's own bin grid.

    Returns bin centres, the mean, and the count, so the caller can decide how
    to draw it. Shared by the static figure and the interactive one, which is
    the point: two views of one table cannot disagree.
    """
    t = pd.DataFrame({"bx": to_bin("x", to_display("x", cal["x_cal"].to_numpy(float))),
                      "by": to_bin("y", to_display("y", cal["y_cal"].to_numpy(float))),
                      "bz": to_bin("z", to_display("z", cal["z_cal"].to_numpy(float))),
                      "v": df[value].to_numpy(float)}).dropna()
    if t.empty:
        return pd.DataFrame(columns=["bx", "by", "bz", "mean", "n"])
    g = t.groupby(["bx", "by", "bz"])["v"].agg(["mean", "size"]).reset_index()
    g = g.rename(columns={"size": "n"})
    return g[g["n"] >= min_n].reset_index(drop=True)


def cloud3d(cal, df, out_png: Path, title: str,
            value="estimated_woba_using_speedangle", min_n=25, keep=0.98,
            vcenter=None) -> Path:
    """The three axes at once, one marker per cell, coloured by `value`.

    Four viewing angles, because a single static angle of a solid cloud hides
    its own interior. Opacity rises with how far the cell sits from the league
    mean, so the ordinary middle fades out and the structure is what remains.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    g = cells3d(cal, df, value, min_n)
    fig = plt.figure(figsize=(15, 11.5))
    if g.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no cell reached the minimum count", ha="center",
                va="center", color=INK)
        fig.savefig(out_png, dpi=DPI, facecolor="white")
        plt.close(fig)
        return Path(out_png)
    centre, lo, hi = _scale_for(value, g["mean"], vcenter, weights=g["n"], keep=keep)
    norm = TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi)
    # Colour comes off the fixed scale so the figure is comparable across
    # figures; opacity and size come off the data, rising with the value.
    alpha, u = _fade(g["mean"], keep=keep)
    # Alpha is baked into the face colours rather than passed as `alpha=`. A 3D
    # collection re-pairs its alpha array against its EDGE colours too, and with
    # edgecolors="none" that is 551 alphas against 0 colours, which raises.
    rgba = CMAP(norm(g["mean"].to_numpy(float)))
    rgba[:, 3] = alpha
    size = 4 + 56 * u ** 1.8
    views = ((22, -60), (22, 30), (60, -45), (8, -90))
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(g["bx"], g["by"], g["bz"], c=rgba, s=size, depthshade=False)
        ax.set_xlabel("x: along the bat (in)", fontsize=8, color=INK, labelpad=1)
        ax.set_ylabel("y: timing (ms)", fontsize=8, color=INK, labelpad=1)
        ax.set_zlabel("z: ball above the plane (in), under is positive",
                      fontsize=8, color=INK, labelpad=1)
        ax.tick_params(labelsize=7, colors=INK)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9, color=INK)
        ax.set_box_aspect((1, 1, 0.8))
    # A 3D axes reserves a wide internal margin for its own tick labels, so the
    # default spacing leaves four small cubes in a large empty figure.
    fig.subplots_adjust(left=0.0, right=0.87, top=0.91, bottom=0.04,
                        wspace=0.0, hspace=0.02)
    # c=rgba leaves the scatter with no mappable, so the colorbar gets its own.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02, fraction=0.03)
    cb.set_label(value, fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)
    fig.suptitle(f"{title}\n one marker per {BINS['x']:.0f} in x {BINS['y']:.0f} ms x "
                 f"{BINS['z']:.0f} in cell of Savant's grid: {len(g):,} cells of at least "
                 f"{min_n} swings.\nFixed scale {lo:g} to {hi:g}, white at {centre:g}; "
                 f"size and opacity rise with {value}",
                 fontsize=12, color=INK)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_png


def cloud3d_html(cal, df, out_html: Path, title: str,
                 value="estimated_woba_using_speedangle", min_n=25, keep=0.98,
                 vcenter=None) -> Path:
    """The same cells as cloud3d, rotatable, as one self-contained HTML file.

    plotly.js is embedded rather than linked, so the page opens on a phone with
    no internet at all. It costs about 3.5 MB, which is the price of the LAN
    rule. Both this and the static figure read cells3d, so the two cannot
    disagree about what is in a cell.
    """
    import plotly.graph_objects as go

    g = cells3d(cal, df, value, min_n)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    if g.empty:
        out_html.write_text("<p>no cell reached the minimum count</p>", encoding="utf-8")
        return out_html
    centre, lo, hi = _scale_for(value, g["mean"], vcenter, weights=g["n"], keep=keep)
    # plotly spaces a colorscale evenly over [cmin, cmax], so an asymmetric
    # reach needs white placed at the fraction the centre actually sits at.
    mid = (centre - lo) / (hi - lo) if hi > lo else 0.5
    scale = [[0.0, POS], [round(float(mid), 6), MID], [1.0, NEG]]
    # Same encoding as the static figure: size and opacity rise with the value.
    # scatter3d takes only a SCALAR marker.opacity, so per-cell alpha has to be
    # baked into rgba strings; that leaves the trace with no colour axis, so a
    # second empty trace carries the colourbar.
    alpha, u = _fade(g["mean"], keep=keep)
    size = 2 + 16 * u ** 1.8
    rgba = CMAP(TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi)(g["mean"].to_numpy(float)))
    fill = [f"rgba({int(r*255)},{int(gr*255)},{int(b*255)},{a:.3f})"
            for (r, gr, b, _), a in zip(rgba, alpha)]
    hover = [f"x {bx:+.0f} in<br>y {by:+.0f} ms<br>z {bz:+.0f} in"
             f"<br>{value} {m:.3f}<br>{int(n):,} swings"
             for bx, by, bz, m, n in zip(g["bx"], g["by"], g["bz"], g["mean"], g["n"])]
    fig = go.Figure(go.Scatter3d(
        x=g["bx"], y=g["by"], z=g["bz"], mode="markers", text=hover, hoverinfo="text",
        showlegend=False,
        marker=dict(size=size, color=fill, line=dict(width=0))))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=0.1, color=[lo], colorscale=scale, cmin=lo, cmax=hi,
                    showscale=True,
                    colorbar=dict(title=dict(text=value, side="right"), thickness=14))))
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>one marker per {BINS['x']:.0f} in x "
                        f"{BINS['y']:.0f} ms x {BINS['z']:.0f} in cell of Savant's grid: "
                        f"{len(g):,} cells of at least {min_n} swings. Fixed scale "
                        f"{lo:g} to {hi:g}, white at {centre:g}; size and opacity "
                        f"rise with {value}</sub>"),
        scene=dict(xaxis_title="x: along the bat (in)", yaxis_title="y: timing (ms)",
                   zaxis_title="z: ball above the plane (in), under is positive",
                   aspectmode="cube"),
        paper_bgcolor="white", margin=dict(l=0, r=0, t=70, b=0), height=820)
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    return out_html


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
        t = table[table.axis == axis].copy()
        t["bin"] = to_display_bin(axis, t["bin"].to_numpy(float))
        t = t.sort_values("bin")
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
