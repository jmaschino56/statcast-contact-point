import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import calibrate as c  # noqa: E402
from cp_lib import savant_truth as st  # noqa: E402


def test_to_bin_lands_on_savants_grid():
    """These expected values changed with the bin convention. They were written as
    plain multiples of the width, which is the grid you get by reading Savant's bin
    label as a lower edge. Savant labels the CENTRE, so its x bins are [-5, -3),
    [-3, -1), [-1, 1) and so on, and a value of 0 sits in the bin labelled 0 whose
    lower edge is -1."""
    assert list(c.to_bin("x", np.array([-4.1, -4.0, 0.0, 1.9, 2.0]))) == [-5, -5, -1, 1, 1]
    assert list(c.to_bin("z", np.array([-0.5, 0.0, 0.99, 1.0]))) == [-0.5, -0.5, 0.5, 0.5]
    assert np.isnan(c.to_bin("y", np.array([np.nan]))[0]), "NaN must survive binning"


def test_quantile_map_matches_target_histogram():
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 1, 20000)                       # wrong scale and center
    edges = np.arange(-20, 21, 2)
    target = pd.Series({b: n for b, n in zip(edges, np.exp(-((edges - 4) / 8.0) ** 2))})
    qm = c.QuantileMap().fit(pred, target)
    out = qm.transform(pred)
    got = pd.Series(c.to_bin("x", out)).value_counts(normalize=True).sort_index()
    want = (target / target.sum())
    common = got.index.intersection(want.index)
    assert np.abs(got[common] - want[common]).sum() < 0.08
    assert np.all(np.diff(qm.transform(np.linspace(-3, 3, 50))) >= 0)


def test_quantile_map_passes_nan_through():
    qm = c.QuantileMap().fit(np.array([0.0, 1.0, 2.0, 3.0]), pd.Series({0: 1, 2: 1}))
    out = qm.transform(np.array([1.0, np.nan]))
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_fallback_weights_undo_tail_selection():
    """The fixture's bins are lower edges on Savant's grid, so they are odd on y.
    They used to be plain multiples of the width, which no value of to_bin can land
    on any more."""
    hist = pd.DataFrame({"id": [1] * 5, "axis": ["y"] * 5, "bin": [-21.0, -11.0, -1.0, 9.0, 19.0],
                         "n": [5, 20, 100, 20, 5]})
    w = c.selection_weights(np.array([20.5, -20.0]), np.array([1, 1]), hist, "y")
    assert abs(w[0] - 5 / 10) < 1e-9 and abs(w[1] - 5 / 10) < 1e-9


def test_selection_weights_unknown_player_is_neutral():
    hist = pd.DataFrame({"id": [1], "axis": ["y"], "bin": [0], "n": [10]})
    w = c.selection_weights(np.array([5.0]), np.array([99]), hist, "y")
    assert w[0] == 1.0


def test_wasserstein_is_zero_against_our_own_histogram():
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 5, 4000)
    df = pd.DataFrame({"pitcher": np.full(4000, 7)})
    mine = pd.Series(c.to_bin("x", vals)).value_counts().sort_index()
    hist = pd.DataFrame({"id": 7, "axis": "x", "bin": mine.index, "n": mine.to_numpy()})
    w1 = c.wasserstein_per_player(vals, df, hist, "x", "pitcher")
    assert abs(w1.loc[7]) < 1e-9


def _pred_and_df(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    ct = rng.choice(["in_play", "foul", "whiff"], n, p=[0.35, 0.4, 0.25])
    centre = np.where(ct == "in_play", 0.0, np.where(ct == "foul", 6.0, 14.0))
    pred = pd.DataFrame({"x_hat": rng.normal(centre, 3, n),
                         "y_hat": rng.normal(centre, 3, n),
                         "z_hat": rng.normal(centre, 3, n)})
    return pred, pd.DataFrame({"contact_type": ct})


def _hist():
    edges = np.arange(-30, 31, 2)
    rows = []
    for ax in ("x", "y", "z"):
        for b, n in zip(edges, np.exp(-((edges - 6) / 9.0) ** 2) * 1000):
            rows.append({"id": 1, "axis": ax, "bin": b, "n": float(n)})
    return pd.DataFrame(rows)


def test_per_type_calibration_gives_each_contact_type_its_own_rates():
    """This test used to assert the opposite, and it encoded a superseded
    expectation rather than a fact: a map per contact type erased the contrast
    only because every type was fit onto the all-swing marginal. Given each type
    its own target built from the split CSV's rates, the map reproduces those
    rates, which is what spec section 6 item 3 asks for.
    """
    pred, df = _pred_and_df()
    rates = {("y", "in_play"): {"lo": 0.05, "mid": 0.90, "hi": 0.05},
             ("y", "foul"): {"lo": 0.25, "mid": 0.50, "hi": 0.25},
             ("y", "whiff"): {"lo": 0.45, "mid": 0.20, "hi": 0.35}}
    maps = c.fit_calibration(pred, df, 2025, hist=_hist(), rates=rates)
    assert set(maps) == {("y", "in_play"), ("y", "foul"), ("y", "whiff")}, set(maps)
    out = c.apply_calibration(pred, df, maps)
    for ct, want in (("in_play", 0.90), ("foul", 0.50), ("whiff", 0.20)):
        v = out.loc[(df["contact_type"] == ct).to_numpy(), "y_cal"].to_numpy()
        assert abs(float(np.mean(np.abs(v) <= 7.0)) - want) < 0.06, (ct, want, v.mean())


def test_pooled_axes_keeps_one_map_for_the_axis_it_names():
    pred, df = _pred_and_df()
    rates = {("y", ct): {"lo": 0.3, "mid": 0.4, "hi": 0.3} for ct in c.CONTACT_TYPES}
    maps = c.fit_calibration(pred, df, 2025, hist=_hist(), rates=rates, pooled_axes=("y",))
    assert set(maps) == {("y", "all")}, set(maps)


def test_calibration_refuses_a_holdout_season():
    pred, df = _pred_and_df()
    try:
        c.fit_calibration(pred, df, 2024, hist=_hist())
    except AssertionError:
        return
    raise AssertionError("fit_calibration must refuse to fit on a holdout season")


def test_quantile_map_ignores_a_singleton_outlier_bin():
    """Savant's own x histogram has bins at -16156 and +4092 holding one swing."""
    edges = list(range(-20, 21, 2))
    target = pd.Series({b: 1000.0 for b in edges})
    target[-16156] = 1.0
    target[4092] = 1.0
    rng = np.random.default_rng(7)
    pred = rng.normal(0, 5, 20000)
    out = c.QuantileMap().fit(pred, target).transform(pred)
    assert out.min() > -40 and out.max() < 40, (out.min(), out.max())
    kept = c.trim_tails(target)
    assert -16156 not in kept.index and 4092 not in kept.index
    assert len(kept) == len(edges)


def test_band_fractions_split_savants_bins_the_way_label_of_does():
    """A bin's share of each category must agree with discover.label_of, which is
    what the validation labels with.

    The expected values here changed once. They were written when Savant's bin
    labels were read as lower edges, which made this "bin -3 is over, bin -2 is
    lined up, bin 2 is under" and put the only straddle on y. Savant labels a bin
    by its CENTRE, so on z label -2 covers [-2.5, -1.5) and splits half over and
    half lined up, label 2 covers [1.5, 2.5) and splits half lined up and half
    under, and only label -3 and beyond is wholly over. y stops straddling: label
    -8 covers [-9, -7), which is entirely late. band_fractions itself did not
    change; what changed is which interval a Savant label names.
    """
    edge = lambda ax, label: st._lower_edge([ax], [label])[0]
    def f(ax, label):
        return c.band_fractions(ax, np.array([edge(ax, label)]))[0].tolist()
    assert f("z", -3) == [1.0, 0.0, 0.0]
    assert f("z", -2) == [0.5, 0.5, 0.0]
    assert f("z", -1) == [0.0, 1.0, 0.0]
    assert f("z", 2) == [0.0, 0.5, 0.5]
    assert f("z", 3) == [0.0, 0.0, 1.0]
    assert f("y", -8) == [1.0, 0.0, 0.0], "7 ms lands on a bin edge once labels are centres"
    assert f("y", -6) == [0.0, 1.0, 0.0]
    assert f("y", 8) == [0.0, 0.0, 1.0]
    assert f("x", -6) == [1.0, 0.0, 0.0]
    assert f("x", -4) == [0.5, 0.5, 0.0]
    assert f("x", 4) == [0.0, 0.5, 0.5]


def test_per_type_target_hits_the_requested_rates_and_keeps_the_total():
    edges = np.arange(-20, 21, 2.0)
    pooled = pd.Series({b: 100.0 for b in edges})
    want = {"lo": 0.30, "mid": 0.50, "hi": 0.20}
    t = c.per_type_target(pooled, "y", want)
    assert abs(t.sum() - pooled.sum()) < 1e-6, "total swings must be preserved"
    lo, mid, hi = c.band_masses(t, "y") / t.sum()
    assert abs(lo - 0.30) < 1e-9 and abs(mid - 0.50) < 1e-9 and abs(hi - 0.20) < 1e-9
    # within-band shape comes from the pooled histogram: a flat pool stays flat
    inside = t[(t.index > -6) & (t.index < 6)]
    assert inside.nunique() == 1, inside.to_dict()


def test_per_type_target_refuses_a_band_the_pool_cannot_supply():
    pooled = pd.Series({0.0: 100.0})          # everything on time, no tails at all
    try:
        c.per_type_target(pooled, "y", {"lo": 0.3, "mid": 0.5, "hi": 0.2})
    except ValueError as e:
        assert "lo" in str(e)
    else:
        raise AssertionError("a target band with no pooled mass must raise, not divide by zero")


CACHE = ROOT / "notebooks" / "contact_point" / "data" / "savant"


def _pooled_rate_error():
    """Max absolute gap per axis between the pooled 2025 pitcher histogram's band
    masses and the same leaderboard's swing-weighted category rates."""
    from cp_lib import savant_truth as st

    h = st.load_histograms(2025, "pitcher")
    lb = st.load_leaderboard(2025, "pitcher").dropna(subset=["id"])
    w = lb["n_swings"].to_numpy(float)
    out = {}
    for ax in ("x", "y", "z"):
        g = h[h.axis == ax].groupby("bin")["n"].sum().sort_index()
        sav = np.array([float(np.average(lb[col], weights=w)) for col in c.BAND_RATES[ax]])
        m = c.band_masses(g, ax)
        out[ax] = float(np.abs(m / m.sum() - sav).max())
    return out


@pytest.mark.skipif(not (CACHE / "done_2025.json").exists(), reason="needs the 2025 Savant cache")
def test_the_histogram_reproduces_savants_own_pooled_rates():
    """Savant agreeing with Savant, with no free parameter. Its histogram labels a
    bin by its CENTRE, and the loader turns that into the lower edge every consumer
    here assumes. Read correctly, the pooled pitcher histogram's three band masses
    ARE the pooled leaderboard's three category rates. Measured: 0.0000 on y,
    0.0032 on z, 0.0165 on x, the last carrying the known 0.15 percent count gap
    between that histogram and n_swings."""
    err = _pooled_rate_error()
    assert err["y"] < 0.001, err
    assert err["z"] < 0.01, err
    assert err["x"] < 0.02, err


@pytest.mark.skipif(not (CACHE / "done_2025.json").exists(), reason="needs the 2025 Savant cache")
def test_reading_the_labels_as_lower_edges_misses_those_rates(monkeypatch):
    """The check above only means something if the wrong convention fails it. Read
    as lower edges, the same three gaps are 0.0734, 0.0317 and 0.0864."""
    from cp_lib import savant_truth as st

    monkeypatch.setattr(st, "BIN_WIDTH", {k: 0.0 for k in st.BIN_WIDTH})
    err = _pooled_rate_error()
    assert err["y"] > 0.02 and err["z"] > 0.05 and err["x"] > 0.05, err


@pytest.mark.skipif(not (CACHE / "done_2025.json").exists(), reason="needs the 2025 Savant cache")
def test_our_bins_land_on_the_same_grid_savants_do():
    """Our binning and Savant's have to name the same intervals or every merge on
    bin silently finds nothing in common.

    This is not hypothetical. Shifting the loaders to lower edges without moving
    to_bin put Savant's x grid on the odd numbers and ours on the even ones, with
    no overlap at all, and the only visible symptom was the per-player Wasserstein
    distance more than doubling.
    """
    from cp_lib import savant_truth as st

    h = st.load_histograms(2025, "pitcher")
    for ax in ("x", "y", "z"):
        theirs = set(np.round(h[h.axis == ax]["bin"].unique().astype(float), 6))
        v = np.linspace(-15, 15, 601)
        ours = set(np.round(np.unique(c.to_bin(ax, v)), 6))
        inside = {b for b in ours if -14 < b < 14}
        assert inside <= theirs, (ax, sorted(inside - theirs)[:6], sorted(theirs)[:6])


def test_per_type_target_is_never_negative():
    """A histogram cannot hold minus four swings, and the 3x3 solve produced exactly
    that: for in-play z, whose requested low band is 0.0045 against a pooled 0.139,
    it returned a negative scale and 45 negative bins. trim_tails then walked a
    cumulative sum that was no longer monotone, kept 5 bins of 50, and the realised
    lined-up rate came out 0.911 against the 0.975 asked for.
    """
    edges = np.arange(-20.5, 20.5, 1.0)
    pooled = pd.Series({b: 100.0 for b in edges})
    t = c.per_type_target(pooled, "z", {"lo": 0.0045, "mid": 0.9751, "hi": 0.0204})
    assert (t >= 0).all(), t[t < 0].to_dict()
    lo, mid, hi = c.band_masses(t, "z") / t.sum()
    assert abs(mid - 0.9751) < 1e-6 and abs(lo - 0.0045) < 1e-6, (lo, mid, hi)
    kept = c.trim_tails(t)
    assert len(kept) > 0.8 * len(t), f"trim_tails kept only {len(kept)} of {len(t)}"


def _reliability_fixture(shrink=1.0, n_players=40, n_swings=200, seed=0):
    """Players whose true z spread differs, so their lined-up rates differ. shrink
    pulls every swing toward its own player's mean, which polarises the rates."""
    rng = np.random.default_rng(seed)
    ids, vals = [], []
    spreads = np.linspace(0.8, 4.0, n_players)
    centres = np.linspace(-1.0, 1.0, n_players)
    for p, (sd, mu) in enumerate(zip(spreads, centres)):
        v = rng.normal(mu, sd, n_swings)
        v = v.mean() + shrink * (v - v.mean())
        ids += [p] * n_swings
        vals += list(v)
    ids = np.asarray(ids)
    cal = pd.DataFrame({"z_cal": np.asarray(vals), "x_cal": np.nan, "y_cal": np.nan})
    df = pd.DataFrame({"pitcher": ids})
    rate = (pd.DataFrame({"id": ids, "in": np.abs(vals) <= 2.0})
            .groupby("id")["in"].mean().rename("lined_up_percent").reset_index())
    rate["n_swings"] = n_swings
    return cal, df, rate


def test_reliability_k_is_one_when_the_spread_already_matches():
    """Savant's per-player rates here ARE ours, so there is nothing to undo."""
    cal, df, rate = _reliability_fixture()
    k = c.fit_reliability(cal, df, 2025, "pitcher", savant=rate)
    assert abs(k["z"] - 1.0) < 0.02, k


def test_reliability_k_exceeds_one_when_the_within_player_spread_is_shrunk():
    """Pulling every swing halfway to its own player's mean polarises the rates:
    a player centred inside the band gets more of them, one centred outside gets
    fewer, so the between-player spread of the rate grows and the correction has to
    widen it back. On the real data the sign is the other way, because per-swing
    error widens rather than narrows."""
    _, _, truth = _reliability_fixture()
    cal, df, _ = _reliability_fixture(shrink=0.5)
    k = c.fit_reliability(cal, df, 2025, "pitcher", savant=truth)
    assert k["z"] > 1.2, k


def test_reliability_leaves_players_under_the_floor_alone():
    v = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    ids = np.array([1, 1, 1, 1, 2, 2])          # player 1 has 4 swings, player 2 has 2
    out = c.apply_reliability(v, ids, 0.5, min_swings=3)
    assert list(out[4:]) == [0.0, 10.0], "a player under the floor must be untouched"
    assert list(out[:4]) == [2.5, 7.5, 2.5, 7.5], list(out[:4])


def test_reliability_keeps_every_players_mean():
    """The correction scales about each player's own mean, so no player's mean
    moves. That is what keeps it from being a recentring in disguise."""
    cal, df, _ = _reliability_fixture()
    out = c.apply_reliability(cal["z_cal"].to_numpy(), df["pitcher"].to_numpy(), 0.7)
    before = pd.Series(cal["z_cal"].to_numpy()).groupby(df["pitcher"]).mean()
    after = pd.Series(out).groupby(df["pitcher"]).mean()
    assert np.allclose(before.to_numpy(), after.to_numpy())
