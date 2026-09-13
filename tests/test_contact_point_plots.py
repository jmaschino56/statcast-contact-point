import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import plots  # noqa: E402


def _fake(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.DataFrame({"x_cal": rng.normal(0, 6, n), "y_cal": rng.normal(0, 10, n),
                        "z_cal": rng.normal(0, 2.5, n)})
    df = pd.DataFrame({"delta_run_exp": rng.normal(0, 0.1, n),
                       "contact_type": rng.choice(["in_play", "foul", "whiff"], n),
                       "estimated_woba_using_speedangle": rng.uniform(0, 1, n)})
    return cal, df


def test_heatmap_grid_writes_png(tmp_path):
    cal, df = _fake()
    out = plots.heatmap_grid(cal, df, "delta_run_exp", tmp_path / "hm.png", "test")
    assert out.exists() and out.stat().st_size > 10_000


def test_heatmap_grid_survives_nan_values(tmp_path):
    """Unrecovered or unreconstructed swings arrive as NaN and must not blank the figure."""
    cal, df = _fake()
    cal.loc[cal.sample(frac=0.4, random_state=1).index, "z_cal"] = np.nan
    df.loc[df.sample(frac=0.2, random_state=2).index, "delta_run_exp"] = np.nan
    out = plots.heatmap_grid(cal, df, "delta_run_exp", tmp_path / "hm_nan.png", "test nan")
    assert out.exists() and out.stat().st_size > 10_000


def test_league_bins_figure_writes_png(tmp_path):
    rows = []
    for ax, w in (("x", 2.0), ("y", 2.0), ("z", 1.0)):
        for b in np.arange(-20, 21, w):
            rows.append({"axis": ax, "bin": b, "n_ours": 500 + b, "rv_ours": b * 0.001,
                         "n_sav": 520 + b, "rv_sav": b * 0.0011})
    out = plots.league_bins_figure(pd.DataFrame(rows), tmp_path / "bins.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_rate_scatter_writes_png(tmp_path):
    rng = np.random.default_rng(4)
    ours = pd.DataFrame({"id": range(60), "centered_percent": rng.uniform(0.5, 0.8, 60),
                         "n_swings": rng.integers(120, 900, 60)})
    sav = ours.copy()
    sav["centered_percent"] = sav["centered_percent"] + rng.normal(0, 0.02, 60)
    out = plots.rate_scatter(ours, sav, "centered_percent", tmp_path / "sc.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_ols_recovers_a_known_shrunken_slope():
    """The fit line has to show shrinkage, which is the whole reason it is drawn."""
    rng = np.random.default_rng(11)
    sav = pd.Series(rng.uniform(0.4, 0.8, 400))
    ours = 0.60 + 0.55 * (sav - 0.60)          # shrunk toward the mean by construction
    slope, intercept = plots._ols(sav, ours)
    assert abs(slope - 0.55) < 0.01
    assert abs(intercept - (0.60 - 0.55 * 0.60)) < 0.01


def test_ols_is_pairwise_complete_and_survives_a_degenerate_axis():
    a = pd.Series([0.5, 0.6, None, 0.7, 0.8])
    b = pd.Series([0.5, 0.62, 0.64, None, 0.79])
    slope, _ = plots._ols(a, b)
    assert slope is not None
    flat, _ = plots._ols(pd.Series([0.5] * 10), pd.Series(range(10), dtype=float))
    assert flat is None


def test_rate_scatter_labels_the_fit_line(tmp_path):
    rng = np.random.default_rng(5)
    sav_rate = rng.uniform(0.45, 0.80, 80)
    ours = pd.DataFrame({"id": range(80),
                         "centered_percent": 0.6 + 0.7 * (sav_rate - 0.6),
                         "n_swings": rng.integers(150, 900, 80)})
    sav = pd.DataFrame({"id": range(80), "centered_percent": sav_rate,
                        "n_swings": ours["n_swings"]})
    out = plots.rate_scatter(ours, sav, "centered_percent", tmp_path / "sc.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_tails_scatter_draws_the_fit(tmp_path):
    rng = np.random.default_rng(7)
    truth = {a: rng.normal(0, 10, 300) for a in ("x", "y", "z")}
    pred = {a: 0.6 * truth[a] + rng.normal(0, 2, 300) for a in ("x", "y", "z")}
    out = plots.tails_scatter(pred, truth, tmp_path / "tails.png", "caveat")
    assert out.exists() and out.stat().st_size > 10_000


def test_league_bins_figure_limits_the_axis_to_the_bulk():
    """Savant's own x histogram has bins at -16156 and 4092 holding one swing."""
    rows = []
    for b in np.arange(-20, 21, 2.0):
        rows.append({"axis": "x", "bin": b, "n_ours": 5000.0, "rv_ours": b * 0.001,
                     "n_sav": 5000.0, "rv_sav": b * 0.0011})
    for b in (-16156.0, 4092.0):
        rows.append({"axis": "x", "bin": b, "n_ours": np.nan, "rv_ours": np.nan,
                     "n_sav": 1.0, "rv_sav": -0.05})
    t = pd.DataFrame(rows)
    lo, hi = plots._bulk_range(t, "x")
    assert -30 < lo < -20 and 20 < hi < 30, (lo, hi)


def test_league_bins_figure_drops_sparse_bins_from_the_run_value_line(tmp_path):
    rows = []
    for ax, w in (("x", 2.0), ("y", 2.0), ("z", 1.0)):
        for b in np.arange(-20, 21, w):
            rows.append({"axis": ax, "bin": b, "n_ours": 900.0, "rv_ours": b * 0.001,
                         "n_sav": 900.0, "rv_sav": b * 0.0011})
        rows.append({"axis": ax, "bin": 40.0, "n_ours": 3.0, "rv_ours": 9.9,
                     "n_sav": 3.0, "rv_sav": 9.9})
    out = plots.league_bins_figure(pd.DataFrame(rows), tmp_path / "b.png", min_n=100)
    assert out.exists() and out.stat().st_size > 10_000


def test_committed_notebook_keeps_tables_and_drops_images():
    """Spec section 8: the notebook is committed with outputs cleared except the
    scorecard and validation tables. The figures are delivered through the media
    album, so the embedded PNGs are what makes the committed file large."""
    import nbformat
    import build_notebook

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("x")]
    nb.cells[0].outputs = [
        nbformat.v4.new_output("display_data", data={"image/png": "AAAA",
                                                     "text/plain": "<Figure>"}),
        nbformat.v4.new_output("display_data", data={"text/html": "<table/>",
                                                     "text/plain": "a b"}),
        nbformat.v4.new_output("stream", name="stdout", text="scorecard"),
    ]
    build_notebook.strip_images(nb)
    kinds = [set(o.get("data", {})) for o in nb.cells[0].outputs if o.output_type != "stream"]
    assert {"image/png"} not in kinds, "the PNG output must be gone"
    assert len(nb.cells[0].outputs) == 2, nb.cells[0].outputs
    assert nb.cells[0].outputs[0]["data"]["text/html"] == "<table/>"
    assert nb.cells[0].outputs[1]["text"] == "scorecard"


def test_every_notebook_code_cell_compiles():
    """A cell source is a Python string inside build_notebook.py, so an escape meant
    for the cell needs two backslashes in the file. Written with one, a print("\\n...")
    becomes a real newline inside a quoted string and the cell is a syntax error that
    only papermill finds, an hour into a run. Two cells shipped that way."""
    import build_notebook

    broken = []
    for i, (kind, src) in enumerate(build_notebook.CELLS):
        if kind == "md":
            continue
        try:
            compile(src, f"<cell {i}>", "exec")
        except SyntaxError as e:
            broken.append((i, str(e)))
    assert not broken, broken


def test_diverging_limits_stays_symmetric_for_a_signed_quantity():
    """Run value is signed about zero, so equal magnitudes must get equal colour."""
    v = np.array([-0.4, -0.1, 0.0, 0.2, 0.9])
    centre, lo, hi = plots._diverging_limits(v, 0.0)
    assert centre == 0.0
    assert lo == -hi
    assert hi == 0.9


def test_diverging_limits_never_reaches_past_an_unsigned_quantity():
    """The shipped xwOBA heatmap centred a diverging map at zero.

    Every xwOBA sits above zero, so the whole panel landed in the blue half and
    the colourbar ran down to -0.8 where no swing can go. White belongs at the
    mean and neither half may reach past the data.
    """
    v = np.array([0.10, 0.30, 0.35, 0.40, 0.92])
    centre, lo, hi = plots._diverging_limits(v, None)
    assert abs(centre - v.mean()) < 1e-9
    assert lo == v.min() and hi == v.max()
    assert lo > 0.0, "the red half reached past the smallest xwOBA that exists"


def test_diverging_limits_weights_the_centre_by_cell_count():
    v = np.array([0.2, 0.8])
    plain, _, _ = plots._diverging_limits(v, None)
    weighted, _, _ = plots._diverging_limits(v, None, weights=np.array([9.0, 1.0]))
    assert abs(plain - 0.5) < 1e-9
    assert abs(weighted - 0.26) < 1e-9


def test_hexbin_never_puts_white_below_the_smallest_possible_xwoba(tmp_path):
    """The original defect: white at zero and a bar reaching -0.8.

    Whatever chooses the limits, an unsigned quantity must not be drawn with
    half its colour range below the smallest value it can take.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake()
    fig, ax = plt.subplots()
    plots.hexbin(cal, df, ("x", "y"), "estimated_woba_using_speedangle",
                 min_n=5, ax=ax, cbar=False)
    hb = [c for c in ax.collections if hasattr(c, "get_offsets")][0]
    assert hb.norm.vmin >= 0.0, f"the cold half reached down to {hb.norm.vmin}"
    assert hb.norm.vcenter > 0.0, f"white sat at {hb.norm.vcenter}"
    plt.close(fig)


def test_hexbin_grid_writes_png(tmp_path):
    cal, df = _fake()
    out = plots.hexbin_grid(cal, df, "delta_run_exp", tmp_path / "hx.png", "test")
    assert out.exists() and out.stat().st_size > 10_000


def test_hexbin_grid_survives_nan_values(tmp_path):
    cal, df = _fake()
    cal.loc[cal.sample(frac=0.4, random_state=1).index, "z_cal"] = np.nan
    df.loc[df.sample(frac=0.2, random_state=2).index, "delta_run_exp"] = np.nan
    out = plots.hexbin_grid(cal, df, "delta_run_exp", tmp_path / "hx_nan.png", "nan")
    assert out.exists() and out.stat().st_size > 10_000


def test_cells3d_drops_cells_below_the_minimum_count():
    cal, df = _fake(n=20000, seed=3)
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=25)
    assert not g.empty
    assert int(g["n"].min()) >= 25
    loose = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=1)
    assert len(loose) > len(g)


def test_cells3d_lands_on_savants_bin_grid():
    """The 3D cells must be the same cells the 2D panels and Savant use."""
    from cp_lib.calibrate import to_bin
    cal, df = _fake(n=20000, seed=4)
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=5)
    for axis, col in (("x", "bx"), ("y", "by"), ("z", "bz")):
        want = set(np.unique(to_bin(axis, cal[f"{axis}_cal"].to_numpy(float))))
        assert set(g[col]) <= want, f"{axis} cells left Savant's grid"


def test_cloud3d_writes_png(tmp_path):
    """Also pins the RGBA path: a per-point alpha= array raises on a 3D scatter
    whose edgecolors are 'none', because the alphas get paired with 0 colours."""
    cal, df = _fake(n=20000, seed=5)
    out = plots.cloud3d(cal, df, tmp_path / "cloud.png", "test",
                        value="estimated_woba_using_speedangle", min_n=10)
    assert out.exists() and out.stat().st_size > 50_000


def test_cloud3d_survives_an_empty_cell_table(tmp_path):
    cal, df = _fake(n=200, seed=6)
    out = plots.cloud3d(cal, df, tmp_path / "empty.png", "test", min_n=10_000)
    assert out.exists()


def test_cloud3d_html_is_self_contained(tmp_path):
    """LAN only: the page has to open on a phone with no internet at all."""
    cal, df = _fake(n=20000, seed=7)
    out = plots.cloud3d_html(cal, df, tmp_path / "cloud.html", "test",
                             value="estimated_woba_using_speedangle", min_n=10)
    html = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 1_000_000, "plotly.js was linked, not embedded"
    assert "Plotly.newPlot" in html
    assert '<script src="https://' not in html and "<script src='https://" not in html


def test_cloud3d_html_places_white_at_the_centre_of_an_asymmetric_range(tmp_path):
    """plotly spaces a colorscale evenly over [cmin, cmax].

    With white pinned at 0.5 and an asymmetric reach the neutral colour lands
    somewhere that is not the league mean, and the page disagrees with the PNG.
    """
    cal, df = _fake(n=20000, seed=8)
    # A deterministic function of x, so the skew survives averaging inside a
    # cell. Drawing per row instead lets the cell means concentrate on the mean
    # and the fixture's own guard below then refuses the test.
    df["unscaled_metric"] = np.exp(-np.abs(cal["x_cal"]) / 1.0)
    # keep is passed on both sides so the test pins the placement rule, not the
    # current default.
    out = plots.cloud3d_html(cal, df, tmp_path / "c.html", "t",
                             value="unscaled_metric", min_n=10, keep=0.98)
    g = plots.cells3d(cal, df, "unscaled_metric", 10)
    centre, lo, hi = plots._diverging_limits(g["mean"], None, weights=g["n"], keep=0.98)
    want = (centre - lo) / (hi - lo)
    assert abs(want - 0.5) > 0.02, "fixture is symmetric, the test proves nothing"
    m = re.search(r'\[\[0(?:\.0)?,\s*"' + re.escape(plots.POS)
                  + r'"\],\s*\[([0-9.]+),\s*"' + re.escape(plots.MID) + r'"\]',
                  out.read_text(encoding="utf-8"))
    assert m, "could not find the diverging colorscale in the page"
    assert abs(float(m.group(1)) - want) < 1e-3, (
        f"white sits at {m.group(1)} of the range, the mean is at {want:.4f}")


def test_diverging_limits_clips_a_signed_reach_at_the_keep_quantile():
    """A few extreme cells must not wash out the rest.

    Over all swings 91.8 percent of the run value cells sat inside a quarter of
    max|v|, so the figure was a faint pink blur with one blue streak. Clipping
    the reach at the keep quantile puts the median cell at 40 percent of the
    bar instead of 14.
    """
    v = np.concatenate([np.full(99, 0.05), np.array([5.0])])
    _, lo_max, hi_max = plots._diverging_limits(v, 0.0, keep=1.0)
    _, lo_clip, hi_clip = plots._diverging_limits(v, 0.0, keep=0.90)
    assert hi_max == 5.0, "keep=1.0 should still reach the extreme"
    assert hi_clip < 0.2, f"the outlier still set the reach ({hi_clip})"
    assert lo_clip == -hi_clip, "a signed quantity must stay symmetric"


def test_the_colormap_runs_cold_to_hot():
    """Savant's polarity: blue at the bottom of the scale, red at the top."""
    lo = plots.CMAP(0.0)[:3]
    hi = plots.CMAP(1.0)[:3]
    assert lo[2] > lo[0], f"the low end is not blue: {lo}"
    assert hi[0] > hi[2], f"the high end is not red: {hi}"


def test_the_fixed_scales_are_the_ones_that_were_asked_for():
    assert plots.SCALES["estimated_woba_using_speedangle"] == (0.0, 0.6, 1.2)
    assert plots.SCALES["delta_run_exp"] == (-0.3, 0.0, 0.3)


def test_a_fixed_scale_ignores_the_data_it_is_given():
    """The point of a fixed scale is that two panels can be compared.

    A scale fitted per panel makes a quiet panel and a violent one look alike.
    """
    quiet = np.array([0.55, 0.60, 0.65])
    violent = np.array([0.01, 0.60, 1.19])
    for vals in (quiet, violent):
        centre, lo, hi = plots._scale_for("estimated_woba_using_speedangle", vals)
        assert (lo, centre, hi) == (0.0, 0.6, 1.2)


def test_an_unnamed_metric_still_gets_limits():
    vals = np.array([-0.4, 0.0, 0.9])
    centre, lo, hi = plots._scale_for("some_new_metric", vals, vcenter=0.0)
    assert centre == 0.0 and lo == -hi and hi > 0


def test_every_panel_of_a_hexbin_grid_shares_one_scale(tmp_path):
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=11)
    norms = []
    for pair in (("x", "y"), ("x", "z"), ("y", "z")):
        fig, ax = plt.subplots()
        plots.hexbin(cal, df, pair, "estimated_woba_using_speedangle",
                     min_n=5, ax=ax, cbar=False)
        hb = [c for c in ax.collections if hasattr(c, "get_offsets")][0]
        norms.append((hb.norm.vmin, hb.norm.vcenter, hb.norm.vmax))
        plt.close(fig)
    assert norms[0] == norms[1] == norms[2] == (0.0, 0.6, 1.2), norms


def test_fade_rises_with_the_value_and_is_monotone():
    """Opacity tracks the value itself, not distance from a centre.

    A symmetric ramp made the worst cells as loud as the best and buried the
    ridge inside an opaque shell of ordinary blue. The best cell must be fully
    opaque and the worst nearly invisible.
    """
    vals = np.linspace(0.0, 1.0, 200)
    alpha, u = plots._fade(vals)
    assert np.all(np.diff(alpha) >= -1e-12), "opacity is not monotone in the value"
    assert alpha[-1] > 0.95, f"the top of the data is not opaque: {alpha[-1]:.3f}"
    assert alpha[0] < 0.10, f"the bottom of the data is not faint: {alpha[0]:.3f}"
    assert u[0] == 0.0 and u[-1] == 1.0


def test_fade_reaches_full_opacity_even_when_the_scale_leaves_headroom():
    """xwOBA tops out near 0.92 on a bar drawn to 1.2.

    Normalising opacity on the fixed scale would cap the reddest cell present
    at about two thirds opaque, so it normalises on the data instead.
    """
    vals = np.linspace(0.05, 0.92, 300)
    alpha, _ = plots._fade(vals)
    assert alpha.max() > 0.95, alpha.max()


def test_fade_takes_no_weights_or_anchor():
    """Nothing about the fade can be steered by a caller's colour scale."""
    import inspect
    names = set(inspect.signature(plots._fade).parameters)
    assert names == {"values", "keep", "floor", "span", "power"}, names


def test_cloud3d_renders_a_ridge_more_opaque_than_its_surroundings(tmp_path):
    rng = np.random.default_rng(21)
    n = 40000
    cal = pd.DataFrame({"x_cal": rng.normal(0, 6, n), "y_cal": rng.normal(0, 10, n),
                        "z_cal": rng.normal(0, 2.5, n)})
    v = np.where(np.abs(cal["x_cal"]) < 2.0, 0.80, 0.30)
    df = pd.DataFrame({"estimated_woba_using_speedangle": v})
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", 10)
    ridge = (g["mean"] > 0.7).to_numpy()
    assert ridge.any() and (~ridge).any(), "fixture lost its ridge"
    alpha, _ = plots._fade(g["mean"])
    assert alpha[ridge].mean() > alpha[~ridge].mean()
    out = plots.cloud3d(cal, df, tmp_path / "c.png", "t",
                        value="estimated_woba_using_speedangle", min_n=10)
    assert out.exists() and out.stat().st_size > 50_000


def test_the_interactive_page_uses_the_fixed_scale(tmp_path):
    cal, df = _fake(n=20000, seed=22)
    out = plots.cloud3d_html(cal, df, tmp_path / "c.html", "t",
                             value="estimated_woba_using_speedangle", min_n=10)
    # The embedded plotly.js mentions "cmin" in its own source, so read the
    # figure payload at the last newPlot call, and do not assume key order.
    html = out.read_text(encoding="utf-8")
    payload = html[html.rfind("Plotly.newPlot"):]
    lo = re.search(r'"cmin":\s*([0-9.eE+-]+)', payload)
    hi = re.search(r'"cmax":\s*([0-9.eE+-]+)', payload)
    assert lo and hi, "no cmin/cmax in the figure payload"
    assert (float(lo.group(1)), float(hi.group(1))) == (0.0, 1.2), (lo.group(1), hi.group(1))


def test_every_axis_is_drawn_on_savants_own_sign():
    """No axis is negated, so a figure and a validation table cannot disagree."""
    assert plots.DISPLAY_SIGN == {"x": 1.0, "y": 1.0, "z": 1.0}
    v = np.array([-3.0, 0.0, 2.0])
    for axis in ("x", "y", "z"):
        assert list(plots.to_display(axis, v)) == [-3.0, 0.0, 2.0]


def test_under_is_the_positive_end_of_z_like_savant():
    """Savant's number is the BALL's height, their word is what the BAT did.

    A ball above the swing plane means the bat passed under it, so "under" is
    positive: their 2025 league page publishes avg_z_over = -3.78 and
    avg_z_under = +2.99, and every one of their per-swing rows labelled Over
    carries a negative value under both batter hands.
    """
    assert "over  |  lined up  |  under" in plots.AXIS_LABEL["z"]
    assert "ball above the swing plane" in plots.AXIS_LABEL["z"]


def test_cells3d_keeps_z_on_savants_sign():
    from cp_lib.calibrate import to_bin
    cal, df = _fake(n=20000, seed=31)
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=5)
    stored = set(np.unique(to_bin("z", cal["z_cal"].to_numpy(float))))
    assert set(g["bz"]) <= stored


def test_a_flipped_bin_still_contains_its_own_value():
    """Negating a bin LABEL is not the same as negating its data.

    to_bin names a bin by its LOWER EDGE, so [lo, lo+w) mirrors to a bin whose
    lower edge is -lo-w. Dropping the width shifted every z cell one whole inch
    and moved the peak of the xwOBA ridge into the wrong bin, while the hexbins,
    which flip the value, showed it correctly. The two disagreed by a bin.
    """
    from cp_lib.calibrate import to_bin, BINS
    rng = np.random.default_rng(5)
    # Production draws every axis unflipped, so force the negated branch here:
    # the rule has to stay correct for whoever flips an axis next.
    for sign in (1.0, -1.0):
        for axis in ("x", "y", "z"):
            saved = plots.DISPLAY_SIGN[axis]
            plots.DISPLAY_SIGN[axis] = sign
            try:
                v = rng.normal(0, 5, 4000)
                drawn = plots.to_display(axis, v)
                label = to_bin(axis, drawn)
                w = BINS[axis]
                assert np.all(label <= drawn + 1e-9), (axis, sign)
                assert np.all(drawn < label + w + 1e-9), (axis, sign)
            finally:
                plots.DISPLAY_SIGN[axis] = saved


def test_to_display_bin_agrees_with_binning_the_flipped_value():
    from cp_lib.calibrate import to_bin
    rng = np.random.default_rng(6)
    for sign in (1.0, -1.0):
        for axis in ("x", "y", "z"):
            saved = plots.DISPLAY_SIGN[axis]
            plots.DISPLAY_SIGN[axis] = sign
            try:
                v = rng.normal(0, 5, 4000)
                from_value = to_bin(axis, plots.to_display(axis, v))
                from_label = plots.to_display_bin(axis, to_bin(axis, v))
                assert np.allclose(from_value, from_label), (axis, sign)
            finally:
                plots.DISPLAY_SIGN[axis] = saved


def test_cells3d_and_hexbin_agree_about_where_the_z_ridge_is():
    """The 3D path bins, the hexbin path does not. They must not disagree."""
    rng = np.random.default_rng(7)
    n = 60000
    cal = pd.DataFrame({"x_cal": rng.normal(0, 6, n), "y_cal": rng.normal(0, 10, n),
                        "z_cal": rng.normal(0, 2.5, n)})
    # Peak at z = +1.0, deliberately NOT on a bin edge. It lives in the bin
    # whose lower edge is +0.5, and the 3D path must agree with the raw value.
    v = np.exp(-((cal["z_cal"] - 1.0) ** 2) / 2.0)
    df = pd.DataFrame({"estimated_woba_using_speedangle": v})
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=20)
    prof = g.groupby("bz").apply(lambda t: np.average(t["mean"], weights=t["n"]))
    peak = float(prof.idxmax())
    assert peak == 0.5, f"the peak landed at {peak}, expected +0.5"


def test_the_bat_is_drawn_to_the_x_scale():
    """x is inches along the bat, so the overlay has to share its units."""
    x, r = plots.bat_outline()
    assert abs(x.max() - plots.BAT_TIP_X) < 1e-9, "the tip is not at x = +6"
    assert abs((x.max() - x.min()) - plots.BAT_LENGTH) < 1e-9, "not 34 in long"
    assert abs(x.min() + 28.0) < 1e-9, "the knob is not at x = -28"
    assert 1.2 < r.max() < 1.35, f"barrel radius {r.max()} is not a legal bat"
    handle = r[(x > -25) & (x < -18)]
    assert handle.max() < 0.6, "the handle is as thick as the barrel"


def test_a_bat_panel_widens_its_x_range_to_hold_the_whole_bat():
    """Without this the handle runs off the page and it reads as a cone."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=41)
    fig, ax = plt.subplots()
    plots.hexbin(cal, df, ("x", "y"), "delta_run_exp", min_n=5, ax=ax, cbar=False)
    assert ax.get_xlim()[0] <= -28.0, ax.get_xlim()
    plt.close(fig)

    fig, ax = plt.subplots()
    plots.hexbin(cal, df, ("x", "y"), "delta_run_exp", min_n=5, ax=ax, cbar=False,
                 bat=False)
    assert ax.get_xlim()[0] > -28.0, "bat=False still widened the range"
    plt.close(fig)


def _bat_lines(ax):
    """The two strokes draw_bat lays down, found by their point count."""
    return [l for l in ax.lines if len(l.get_xdata()) == 1200]


def _flat_lines(ax):
    """y values of the horizontal rules on the panel."""
    return {round(float(l.get_ydata()[0]), 4) for l in ax.lines
            if len(l.get_xdata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]}


def test_the_full_bat_profile_goes_only_where_x_is_an_axis():
    """Only x runs along the bat, so only x panels can carry its profile."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=42)
    counts = {}
    for pair in (("x", "y"), ("x", "z"), ("y", "z")):
        fig, ax = plt.subplots()
        plots.hexbin(cal, df, pair, "delta_run_exp", min_n=5, ax=ax, cbar=False)
        counts[pair] = len(_bat_lines(ax))
        plt.close(fig)
    assert counts[("x", "y")] == 2 and counts[("x", "z")] == 2, counts
    assert counts[("y", "z")] == 0, "a bat profile was drawn along milliseconds"


def test_the_shipped_timing_panel_carries_no_bat_furniture():
    """Neither of its axes runs along the bat, so the bat came off it.

    It is the panel where the silhouette taught the least and cost the most
    room, and every line it still draws has to be a Savant threshold.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=46)
    a, b, equal, bat = [p for p in plots.PANELS if p[:2] == ("y", "z")][0]
    assert bat is False, "the shipped y/z panel asked for a bat"
    fig, ax = _panel(cal, df, (a, b), equal, bat)
    reach = plots.BAT_R + plots.BALL_R
    flat = _flat_lines(ax)
    for gone in (plots.BAT_R, -plots.BAT_R, reach, -reach):
        assert round(gone, 4) not in flat, (gone, flat)
    assert _bat_lines(ax) == []
    assert flat == {round(-plots.THRESH["z"], 4), round(plots.THRESH["z"], 4)}, flat
    plt.close(fig)


def _panel(cal, df, pair, equal, bat=True):
    """One panel built exactly the way hexbin_grid builds it."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    plots.hexbin(cal, df, pair, "delta_run_exp", min_n=5, ax=ax, cbar=False,
                 equal=equal, bat=bat)
    return fig, ax


def _legend_labels(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def test_the_shipped_panels_are_the_ones_the_tests_check():
    assert plots.PANELS == (("x", "z", True, True), ("x", "y", False, True),
                            ("y", "z", False, False))


def test_only_the_x_z_panel_holds_its_two_axes_to_one_scale():
    """Both of its axes are inches. No other pair can share a scale."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=44)
    for a, b, equal, bat in plots.PANELS:
        fig, ax = _panel(cal, df, (a, b), equal, bat)
        locked = ax.get_aspect() == 1.0
        assert locked == equal, (a, b, equal, ax.get_aspect())
        if locked:
            assert (a, b) == ("x", "z"), (a, b)
        plt.close(fig)


def test_on_the_one_scale_panel_the_bat_is_a_real_34_by_2_6_inch_shape():
    """This is the panel where the silhouette IS the barrel.

    Drawn isotropically instead it came out 0.28 in thick on the z axis, which
    is what made it look wrong against the 2.75 in reach lines.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=44)
    fig, ax = _panel(cal, df, ("x", "z"), True)
    line = _bat_lines(ax)[0]
    assert line.get_transform() == ax.transData, "the bat is not in data units"
    y = np.asarray(line.get_ydata(), float)
    x = np.asarray(line.get_xdata(), float)
    assert abs(y.max() - plots.BAT_R) < 1e-3, (y.max(), plots.BAT_R)
    assert abs((x.max() - x.min()) - plots.BAT_LENGTH) < 1e-9
    plt.close(fig)


def test_the_barrel_is_drawn_once():
    """Its own outline is the barrel on the one-scale panel.

    A separate pair of barrel lines there would double the edge; the other z
    panel has no outline, so it needs them.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=44)
    reach = plots.BAT_R + plots.BALL_R
    fig, ax = _panel(cal, df, ("x", "z"), True)
    flat = _flat_lines(ax)
    assert round(plots.BAT_R, 4) not in flat, "barrel drawn twice on the bat panel"
    assert round(reach, 4) in flat and round(-reach, 4) in flat, flat
    plt.close(fig)

    # The shipped figure no longer asks for this, but a caller that does still
    # gets a barrel edge-on rather than a silhouette in milliseconds.
    fig, ax = _panel(cal, df, ("y", "z"), False, bat=True)
    flat = _flat_lines(ax)
    for want in (plots.BAT_R, -plots.BAT_R, reach, -reach):
        assert round(want, 4) in flat, (want, flat)
    plt.close(fig)

    fig, ax = _panel(cal, df, ("x", "y"), False)
    assert round(plots.BAT_R, 4) not in _flat_lines(ax), "a barrel depth in milliseconds"
    plt.close(fig)


def test_the_silhouette_is_isotropic_wherever_the_axes_cannot_share_a_scale():
    """There the other axis is milliseconds, so the thickness is drawing, not
    data, and it is sized to look right rather than to mean something."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=49)
    for a, b, equal, bat in plots.PANELS:
        if equal or a != "x":
            continue
        fig, ax = _panel(cal, df, (a, b), equal, bat)
        line = _bat_lines(ax)[0]
        assert line.get_transform() != ax.transData, (a, b)
        y = np.asarray(line.get_ydata(), float)
        assert 0.0 <= y.min() and y.max() <= 1.0, (a, b, y.min(), y.max())
        plt.close(fig)


def test_on_the_x_y_panel_the_bat_height_is_not_in_data_units():
    """The other axis is milliseconds, which no bat has a thickness in."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=45)
    fig, ax = plt.subplots()
    plots.hexbin(cal, df, ("x", "y"), "delta_run_exp", min_n=5, ax=ax, cbar=False)
    line = _bat_lines(ax)[0]
    assert line.get_transform() != ax.transData, "milliseconds were read as inches"
    y = np.asarray(line.get_ydata(), float)
    assert 0.0 <= y.min() and y.max() <= 1.0, "not an axes fraction"
    assert y.max() - y.min() < 2.2 * plots.BAT_FRAC_HALF + 1e-9
    plt.close(fig)


def test_the_timing_panel_bat_is_drawn_isotropic():
    """Its height is not a data quantity, so it is sized to LOOK to scale.

    A fixed fraction left it about twice too thick beside the equal-aspect
    panel, which is what made it read stubby.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=48)
    fig, ax = plt.subplots(figsize=(8, 5))
    plots.hexbin(cal, df, ("x", "y"), "delta_run_exp", min_n=5, ax=ax, cbar=False)
    line = _bat_lines(ax)[0]
    y = np.asarray(line.get_ydata(), float)
    x = np.asarray(line.get_xdata(), float)
    pos = ax.get_position()
    w_in = pos.width * fig.get_figwidth()
    h_in = pos.height * fig.get_figheight()
    x_lo, x_hi = ax.get_xlim()
    length_in = (x.max() - x.min()) / (x_hi - x_lo) * w_in     # drawn, figure inches
    thick_in = (y.max() - y.min()) * h_in
    assert abs(length_in / thick_in - plots.BAT_LENGTH / (2 * plots.BAT_R)) < 0.6, (
        length_in / thick_in, plots.BAT_LENGTH / (2 * plots.BAT_R))
    plt.close(fig)


def test_every_quantity_gets_one_range_across_the_whole_figure():
    """x, y and z each appear on two panels and must look identical on both.

    The range is computed once from the whole column and handed to each panel,
    rather than recomputed per panel: the per-panel finite masks differ, so two
    panels drawing the same column came out 53.4 and 53.5 in wide.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=50)
    # DIFFERENT columns go missing, which is what makes the per-panel finite
    # masks differ and the recomputed extents drift. With a fixture that has no
    # NaN at all the masks are identical and this test cannot fail.
    rng = np.random.default_rng(52)
    for axis, frac in (("y", 0.25), ("z", 0.30)):
        idx = rng.choice(len(cal), int(frac * len(cal)), replace=False)
        cal.loc[cal.index[idx], f"{axis}_cal"] = np.nan
    limits = {}
    for axis in ("x", "y", "z"):
        v = cal[f"{axis}_cal"].to_numpy(float)
        limits[axis] = plots._bulk_extent(v[np.isfinite(v)], axis)
    seen = {}
    fig, axs = plt.subplots(1, 3)
    for ax, pair in zip(axs, (("x", "z"), ("x", "y"), ("y", "z"))):
        plots.hexbin(cal, df, pair, "delta_run_exp", min_n=5, ax=ax, cbar=False,
                     limits=limits)
        for q, rng in zip(pair, (ax.get_xlim(), ax.get_ylim())):
            seen.setdefault(q, []).append(tuple(round(float(r), 6) for r in rng))
    plt.close(fig)
    for q, ranges in seen.items():
        assert len(ranges) == 2, (q, ranges)
        assert len(set(ranges)) == 1, f"{q} differs between panels: {ranges}"


def test_hexbin_grid_writes_a_png_with_the_bat_layout(tmp_path):
    cal, df = _fake(n=20000, seed=51)
    out = plots.hexbin_grid(cal, df, "delta_run_exp", tmp_path / "g.png", "t", min_n=5)
    assert out.exists() and out.stat().st_size > 20_000


def _rules(ax):
    """Every straight reference line on the panel, as the number it sits at.

    An axvline is two points with one x; an axhline two points with one y. The
    bat profile is 1200 points, so it never lands here.
    """
    out = set()
    for l in ax.lines:
        x = np.asarray(l.get_xdata(), float)
        y = np.asarray(l.get_ydata(), float)
        if x.size != 2:
            continue
        if x[0] == x[1]:
            out.add(round(abs(float(x[0])), 4))
        elif y[0] == y[1]:
            out.add(round(abs(float(y[0])), 4))
    return out


def _legend_numbers(ax):
    import re
    return {round(float(v), 4)
            for t in _legend_labels(ax)
            for v in re.findall(r"\d+(?:\.\d+)?", t)}


def test_every_reference_line_on_a_panel_is_named_with_its_number():
    """Three kinds of dashed line shared one free-text caption, and the reader
    had no way to tell a Savant threshold from the barrel's reach.

    This is the invariant, not a fixture measurement: whatever rules a panel
    draws, each one's value has to appear in that panel's own legend.
    """
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=51)
    for a, b, equal, bat in plots.PANELS:
        fig, ax = _panel(cal, df, (a, b), equal, bat)
        drawn, named = _rules(ax), _legend_numbers(ax)
        assert drawn, (a, b)
        assert drawn <= named, (a, b, sorted(drawn - named))
        plt.close(fig)


def test_the_legend_names_the_bat_exactly_when_the_panel_draws_one():
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=52)
    for a, b, equal, bat in plots.PANELS:
        fig, ax = _panel(cal, df, (a, b), equal, bat)
        said = [t for t in _legend_labels(ax) if t.startswith("bat,")]
        assert len(said) == (1 if bat and a == "x" else 0), (a, b, said)
        assert len(said) == len(_bat_lines(ax)) // 2, (a, b, said)
        plt.close(fig)


def test_each_savant_threshold_is_named_with_the_word_savant_uses():
    """A number alone does not tell the reader what the band is called."""
    import matplotlib.pyplot as plt
    cal, df = _fake(n=20000, seed=53)
    for a, b, equal, bat in plots.PANELS:
        fig, ax = _panel(cal, df, (a, b), equal, bat)
        text = " | ".join(_legend_labels(ax))
        for axis in (a, b):
            assert plots.MID_NAME[axis] in text, (a, b, axis, text)
            assert f"{plots.THRESH[axis]:g} {plots.AXIS_UNIT[axis]}" in text, (axis, text)
        plt.close(fig)


def test_every_tails_panel_asked_for_actually_draws(tmp_path):
    """Filtering to status == "recovered" shipped a 3-panel figure with 2 blank.

    A blank panel reads as broken, not as the deliberate distinction it was.
    The panel count follows `axes`, and every panel carries its points.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    t = {a: rng.normal(0, 5, 400) for a in ("x", "y", "z")}
    pred = {a: t[a] * 0.9 + rng.normal(0, 1, 400) for a in ("x", "y", "z")}
    from PIL import Image
    widths = {}
    for axes in (("y",), ("x", "z"), ("x", "y", "z")):
        before = plt.get_fignums()
        out = plots.tails_scatter(pred, t, tmp_path / f"t{len(axes)}.png", "c",
                                  axes=axes, status={a: "fallback" for a in axes})
        assert out.exists() and out.stat().st_size > 5000, axes
        assert plt.get_fignums() == before, "a figure was left open"
        widths[len(axes)] = Image.open(out).size[0]
    # The panel count is the thing under test, and only the drawn width shows
    # it once the figure is closed. Each panel is the same 6 in wide, so the
    # image width has to scale with the count. Asserting the file merely exists
    # passes happily when `axes` is ignored and three panels are always drawn.
    assert widths[1] < widths[2] < widths[3], widths
    for n in (2, 3):
        assert abs(widths[n] / widths[1] - n) < 0.25, (n, widths)


def test_the_tails_title_says_which_claim_each_panel_makes():
    """y is a recovered formula. x and z are the best linear candidate and are
    NOT what ships for contact swings. Unlabelled, the reader reads one claim."""
    rec = plots.tails_title("y", "recovered", 0.975, 21328, 0.951)
    fall = plots.tails_title("x", "fallback", 0.844, 21328)
    assert "recovered formula" in rec and "best linear candidate" in fall
    assert "recovered" not in fall, "a modeled axis claimed a recovered formula"
    assert "0.951" in rec and "r2 = 0.712" in fall, (rec, fall)
    bare = plots.tails_title("x", None, 0.844, 10)
    assert "(" not in bare, bare


def test_the_rate_grid_and_the_single_scatter_share_one_drawing():
    """Two copies of the drawing is two places for the slope to be computed
    differently, and that slope below 1 is a headline result."""
    import inspect
    single = inspect.getsource(plots.rate_scatter)
    grid = inspect.getsource(plots.rate_scatter_grid)
    assert "_rate_panel" in single and "_rate_panel" in grid
    assert "ax.scatter" not in single and "ax.scatter" not in grid


def test_the_rate_grid_draws_one_panel_per_rate(tmp_path):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)
    rates = ("on_time_percent", "centered_percent", "lined_up_percent")
    ours = pd.DataFrame({"id": np.arange(300), "n_swings": 200,
                         **{r: rng.uniform(0.4, 0.8, 300) for r in rates}})
    sav = pd.DataFrame({"id": np.arange(300), "n_swings": 200,
                        **{r: rng.uniform(0.4, 0.8, 300) for r in rates}})
    before = plt.get_fignums()
    out = plots.rate_scatter_grid(ours, sav, rates, tmp_path / "g.png", "t")
    assert out.exists() and out.stat().st_size > 5000
    assert plt.get_fignums() == before, "a figure was left open"
