import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import validate as v  # noqa: E402


def _frame():
    df = pd.DataFrame({"pitcher": [1, 1, 1, 2], "batter": [9, 9, 9, 9],
                       "contact_type": ["in_play", "whiff", "foul", "whiff"],
                       "delta_run_exp": [0.3, -0.1, 0.0, -0.2]})
    cal = pd.DataFrame({"x_cal": [0.0, 10.0, -6.0, 0.0],
                        "y_cal": [1.0, 20.0, 0.0, 0.0],
                        "z_cal": [0.5, -5.0, 0.0, 0.0]})
    return df, cal


def test_categorize_perfect_and_flawed():
    df, cal = _frame()
    cats = v.categorize(cal, df)
    assert list(cats["perfect"]) == [True, False, False, False]
    assert list(cats["flawed"]) == [False, True, False, False]


def test_perfect_needs_a_ball_in_play_not_just_contact():
    """Savant reports perfect_percent as exactly 0 for fouls."""
    df = pd.DataFrame({"pitcher": [1, 1], "batter": [9, 9],
                       "contact_type": ["in_play", "foul"], "delta_run_exp": [0.3, 0.0]})
    cal = pd.DataFrame({"x_cal": [0.0, 0.0], "y_cal": [0.0, 0.0], "z_cal": [0.0, 0.0]})
    cats = v.categorize(cal, df)
    assert list(cats["perfect"]) == [True, False]


def test_player_rates_columns_match_savant():
    df, cal = _frame()
    rates = v.player_rates(df, v.categorize(cal, df), "pitcher")
    for c in ("tied_up_percent", "centered_percent", "flailed_percent", "early_percent",
              "on_time_percent", "late_percent", "over_percent", "lined_up_percent",
              "under_percent", "perfect_percent", "flawed_percent", "n_swings",
              "whiff_rate", "avg_x_tied_up", "avg_x_flail"):
        assert c in rates.columns, c
    assert rates.loc[rates.id == 1, "n_swings"].item() == 3
    assert abs(rates.loc[rates.id == 1, "flailed_percent"].item() - 1 / 3) < 1e-9


def test_bucket_means_are_computed_when_values_are_passed():
    df, cal = _frame()
    cats = v.categorize(cal, df)
    rates = v.player_rates(df, cats, "pitcher", cal)
    # pitcher 1: one Flail at x = 10, one Tiedup at x = -6
    assert abs(rates.loc[rates.id == 1, "avg_x_flail"].item() - 10.0) < 1e-9
    assert abs(rates.loc[rates.id == 1, "avg_x_tied_up"].item() - (-6.0)) < 1e-9


def test_compare_rates_is_exact_on_identical_input():
    df, cal = _frame()
    rates = v.player_rates(df, v.categorize(cal, df), "pitcher")
    cmp_ = v.compare_rates(rates, rates, min_swings=1)
    # the six bucket means are NaN here (player_rates was called without values),
    # so they carry a NaN mae by design; every rate that has data must be exact
    assert (cmp_["mae"].dropna() == 0).all()
    assert set(cmp_.loc[cmp_["mae"].isna(), "rate"]) == set(v.MEAN_COLS)


def test_compare_rates_survives_all_nan_columns():
    """player_rates without values leaves the six bucket means NaN; that must not raise."""
    df, cal = _frame()
    rates = v.player_rates(df, v.categorize(cal, df), "pitcher")
    assert rates["avg_x_flail"].isna().all()
    cmp_ = v.compare_rates(rates, rates, min_swings=1)
    assert "avg_x_flail" in set(cmp_["rate"])
    assert cmp_.loc[cmp_.rate == "avg_x_flail", "players"].item() == 0


def test_scorecard_has_one_row_per_table():
    df, cal = _frame()
    rates = v.player_rates(df, v.categorize(cal, df), "pitcher")
    cmp_ = v.compare_rates(rates, rates, min_swings=1)
    sc = v.scorecard({(2025, "pitcher rates"): cmp_, (2024, "pitcher rates"): cmp_})
    assert len(sc) == 2
    assert list(sc["in_sample"]) == [False, True]


def test_scorecard_flags_the_table_the_calibration_was_fit_to():
    """The per-type map is fit to the 2025 pitcher split CSV's own category rates,
    so that one table is not evidence: it is the fit target. In sample is not
    enough to say so, because the 2025 pitcher rates table is in sample too and is
    still a real comparison."""
    df, cal = _frame()
    rates = v.player_rates(df, v.categorize(cal, df), "pitcher")
    cmp_ = v.compare_rates(rates, rates, min_swings=1)
    res = {(2025, "pitcher by contact type"): cmp_, (2025, "pitcher rates"): cmp_,
           (2024, "pitcher by contact type"): cmp_}
    sc = v.scorecard(res, fit_target=(2025, "pitcher by contact type"))
    flag = dict(zip(zip(sc.season, sc.table), sc.calibration_target))
    assert flag[(2025, "pitcher by contact type")] is True
    assert flag[(2025, "pitcher rates")] is False
    assert flag[(2024, "pitcher by contact type")] is False
    assert list(v.scorecard(res)["calibration_target"]) == [False, False, False]
