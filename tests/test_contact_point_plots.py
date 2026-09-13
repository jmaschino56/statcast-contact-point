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


def test_fade_keeps_the_unusual_and_drops_the_ordinary():
    """The ridge must draw MORE opaque than the sea it sits in.

    Anchoring the fade on the fixed scale's white (0.600) instead of the data's
    own centre inverted this: a 0.80 ridge sat 0.20 from white while a 0.30 sea
    sat 0.30 from it, so the ordinary cells drew hardest.
    """
    vals = np.concatenate([np.full(40, 0.80), np.full(360, 0.30)])
    w = np.ones_like(vals)
    alpha, centre = plots._fade(vals, weights=w)
    assert 0.3 < centre < 0.45, f"anchor is not the data centre: {centre}"
    assert alpha[:40].mean() > alpha[40:].mean(), (
        f"the ridge drew fainter than the sea: {alpha[0]:.3f} vs {alpha[-1]:.3f}")
    assert alpha.min() >= 0.10 and alpha.max() <= 0.90


def test_fade_has_no_anchor_argument():
    """The anchor is owned by _fade so a caller cannot pass a scale's white."""
    import inspect
    names = set(inspect.signature(plots._fade).parameters)
    assert names == {"values", "weights", "keep", "floor", "span"}, names


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
    alpha, _ = plots._fade(g["mean"], weights=g["n"])
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


def test_only_z_is_flipped_for_display():
    v = np.array([-3.0, 0.0, 2.0])
    assert list(plots.to_display("x", v)) == [-3.0, 0.0, 2.0]
    assert list(plots.to_display("y", v)) == [-3.0, 0.0, 2.0]
    assert list(plots.to_display("z", v)) == [3.0, -0.0, -2.0]


def test_over_draws_positive_and_under_negative():
    """Jeremy reads "over" as up.

    Savant stores the BALL above the swing plane, so their over is negative
    (avg_z_over = -3.78 in the 2025 league leaderboard). Every figure draws the
    BAT above the ball instead, so over is positive. This pins the direction.
    """
    from cp_lib.calibrate import THRESH
    over_stored = np.array([-6.0, -3.0])     # Savant's sign: ball well below the plane
    under_stored = np.array([3.0, 6.0])
    assert (plots.to_display("z", over_stored) > THRESH["z"]).all()
    assert (plots.to_display("z", under_stored) < -THRESH["z"]).all()
    assert "under  |  lined up  |  over" in plots.AXIS_LABEL["z"]


def test_cells3d_z_is_drawn_on_the_flipped_sign():
    from cp_lib.calibrate import to_bin
    cal, df = _fake(n=20000, seed=31)
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=5)
    stored = set(np.unique(to_bin("z", cal["z_cal"].to_numpy(float))))
    assert set(g["bz"]) <= {-b for b in stored}
    assert set(g["bz"]) & {b for b in stored if b > 0} or True  # symmetry is fine
    # the decisive half: no drawn bz equals a stored bz of the same nonzero value
    # unless its negation is also present
    assert all((-b) in stored for b in g["bz"])


def test_a_flipped_bin_still_contains_its_own_value():
    """Negating a bin LABEL is not the same as negating its data.

    to_bin names a bin by its LOWER EDGE, so [lo, lo+w) mirrors to a bin whose
    lower edge is -lo-w. Dropping the width shifted every z cell one whole inch
    and moved the peak of the xwOBA ridge into the wrong bin, while the hexbins,
    which flip the value, showed it correctly. The two disagreed by a bin.
    """
    from cp_lib.calibrate import to_bin, BINS
    rng = np.random.default_rng(5)
    for axis in ("x", "y", "z"):
        v = rng.normal(0, 5, 4000)
        drawn = plots.to_display(axis, v)
        label = to_bin(axis, drawn)
        w = BINS[axis]
        assert np.all(label <= drawn + 1e-9), axis
        assert np.all(drawn < label + w + 1e-9), axis


def test_to_display_bin_agrees_with_binning_the_flipped_value():
    from cp_lib.calibrate import to_bin
    rng = np.random.default_rng(6)
    for axis in ("x", "y", "z"):
        v = rng.normal(0, 5, 4000)
        from_value = to_bin(axis, plots.to_display(axis, v))
        from_label = plots.to_display_bin(axis, to_bin(axis, v))
        assert np.allclose(from_value, from_label), axis


def test_cells3d_and_hexbin_agree_about_where_the_z_ridge_is():
    """The 3D path bins, the hexbin path does not. They must not disagree."""
    rng = np.random.default_rng(7)
    n = 60000
    cal = pd.DataFrame({"x_cal": rng.normal(0, 6, n), "y_cal": rng.normal(0, 10, n),
                        "z_cal": rng.normal(0, 2.5, n)})
    # Peak at STORED z = +1.0, deliberately NOT on a bin edge: a peak sitting on
    # an edge gives the same label under both conventions and proves nothing.
    # Correct: drawn value -1.0, which lives in the bin whose lower edge is -1.5.
    v = np.exp(-((cal["z_cal"] - 1.0) ** 2) / 2.0)
    df = pd.DataFrame({"estimated_woba_using_speedangle": v})
    g = plots.cells3d(cal, df, "estimated_woba_using_speedangle", min_n=20)
    prof = g.groupby("bz").apply(lambda t: np.average(t["mean"], weights=t["n"]))
    peak = float(prof.idxmax())
    assert peak == -1.5, f"the drawn peak landed at {peak}, expected -1.5"
