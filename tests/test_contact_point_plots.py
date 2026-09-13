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
