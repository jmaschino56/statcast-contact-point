"""Per-batter traits, the MLB player lookup and the Savant swing-path slate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import traits as t  # noqa: E402


def _swings(batter, n, tilt, plate_z=None, season=2025):
    rng = np.random.default_rng(batter)
    return pd.DataFrame({
        "batter": [batter] * n, "season": [season] * n,
        "swing_path_tilt": rng.normal(tilt, 2.0, n),
        "attack_angle": rng.normal(10.0, 3.0, n),
        "bat_speed": rng.normal(70.0, 4.0, n),
        "intercept_ball_minus_batter_pos_x_inches": rng.normal(30.0, 3.0, n),
        "intercept_ball_minus_batter_pos_y_inches": rng.normal(24.0, 3.0, n),
        "plate_z": rng.normal(2.5, 0.6, n) if plate_z is None else plate_z,
        "sz_top": [3.4] * n, "sz_bot": [1.6] * n,
    })


def test_traits_use_only_the_season_they_are_asked_for():
    """A holdout season must never see the fit season's swings. The trait table is
    keyed by batter alone, so a frame carrying two seasons has to be filtered, not
    grouped over."""
    a = _swings(1, 400, tilt=20.0, season=2025)
    b = _swings(1, 400, tilt=40.0, season=2024)
    both = pd.concat([a, b], ignore_index=True)
    only_2024 = t.batter_traits(both, 2024)
    assert abs(float(only_2024.loc[1, "tilt_mean"]) - 40.0) < 1.0, only_2024.loc[1].to_dict()
    assert abs(float(t.batter_traits(both, 2025).loc[1, "tilt_mean"]) - 20.0) < 1.0


def test_traits_fall_back_to_the_league_slope_under_the_floor():
    big = _swings(1, 300, tilt=20.0)
    small = _swings(2, 10, tilt=20.0)
    tr = t.batter_traits(pd.concat([big, small], ignore_index=True), 2025, min_swings=100)
    assert np.isfinite(tr.loc[2, "aa_on_plate_z"]), "a short sample gets the league slope"
    assert tr.loc[2, "aa_on_plate_z"] == tr.attrs["league_aa_on_plate_z"]
    assert tr.loc[1, "aa_on_plate_z"] != tr.attrs["league_aa_on_plate_z"]
    assert tr.loc[2, "n_swings"] == 10


def test_deviation_columns_centre_each_batter_on_zero():
    df = pd.concat([_swings(1, 300, 20.0), _swings(2, 300, 40.0)], ignore_index=True)
    out = t.add_trait_features(df, t.batter_traits(df, 2025))
    for b, g in out.groupby("batter"):
        assert abs(float(g["tilt_dev"].mean())) < 1e-9, b


@pytest.mark.parametrize("raw,inches", [("6' 2\"", 74), ("5' 11\"", 71), ("6' 0\"", 72),
                                        ("6'2\"", 74), (None, None), ("", None)])
def test_height_parser(raw, inches):
    got = t.parse_height(raw)
    assert (got is None and inches is None) or got == inches, (raw, got)


CACHE = ROOT / "notebooks" / "contact_point" / "data" / "savant"


@pytest.mark.skipif(not (CACHE / "done_2025.json").exists(), reason="needs the 2025 cache")
def test_tails_features_reproduce_the_committed_geometry_columns():
    """The rebuild must land on the same swings as the table Task 4 fitted the
    verdicts on. It is easy to get wrong: join_tails hands back a fresh index, and
    indexing the season frame by it takes the wrong rows without raising."""
    old = pd.read_parquet(ROOT / "notebooks/contact_point/data/reports/tails_feats_2025.parquet")
    new = t.tails_features(2025, 0.0, 1.0)
    assert len(new) == len(old)
    for c in ("icpt_x", "icpt_y", "plane_z_at_icpt", "sav_x", "sav_z"):
        a = np.sort(old[c].dropna().to_numpy())
        b = np.sort(new[c].dropna().to_numpy())
        assert len(a) == len(b) and np.abs(a - b).max() < 1e-9, c


def test_fallback_columns_refuse_the_selection_column():
    f = pd.DataFrame({"icpt_x": [1.0], "miss_distance": [2.0], "bt_n_swings": [3.0],
                      "z_from_miss": [4.0], "bt_aa_mean": [5.0], "sp_avg_foot_sep0": [6.0],
                      "zone": [7.0], "stand": ["R"]})
    cols = t.fallback_columns(f, ["icpt_x"])
    assert "miss_distance" not in cols and "bt_n_swings" not in cols
    assert "z_from_miss" not in cols and "stand" not in cols
    assert set(cols) == {"icpt_x", "bt_aa_mean", "sp_avg_foot_sep0", "zone"}, cols


def test_only_the_axes_that_gained_get_the_wide_feature_set():
    """Measured on both holdouts, the new features help x and hurt z: the batter
    lined-up correlation falls 0.745 to 0.708 on 2024 and 0.726 to 0.714 on 2026
    when z is given them. So z is kept on geometry."""
    f = pd.DataFrame({"icpt_x": [1.0], "bt_aa_mean": [2.0], "zone": [3.0],
                      "sp_avg_foot_sep0": [4.0]})
    assert t.fallback_columns(f, ["icpt_x"], axis="x") == ["icpt_x", "zone", "bt_aa_mean",
                                                           "sp_avg_foot_sep0"]
    assert t.fallback_columns(f, ["icpt_x"], axis="z") == ["icpt_x"]
    assert t.fallback_columns(f, ["icpt_x"]) == t.fallback_columns(f, ["icpt_x"], axis="x")
