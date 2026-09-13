import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import discover as d  # noqa: E402


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "icpt_x": rng.normal(30, 8, n), "icpt_y": rng.normal(20, 10, n),
        "icpt_x_body": rng.normal(30, 8, n), "attack_direction": rng.normal(0, 10, n),
        "ball_in_per_ms": np.full(n, 1.5), "z_above_plane": rng.normal(0, 4, n),
        "stand": rng.choice(["R", "L"], n),
    })
    # planted truth. x is a clean affine map of icpt_x_body. y needs BOTH icpt_y and
    # attack_direction: the attack_direction term carries 36 percent of the variance,
    # so icpt_y alone cannot clear the 0.8 r2 gate and the ladder must climb to the
    # two-feature candidate. z has no explanation in the candidate set.
    f["sav_x"] = 1.1 * f["icpt_x_body"] - 33 + rng.normal(0, 0.5, n)
    f["sav_y"] = 0.8 * f["icpt_y"] - 0.6 * f["attack_direction"] - 15 + rng.normal(0, 0.5, n)
    f["sav_z"] = rng.normal(0, 4, n)
    return f


def test_labels_use_savant_thresholds():
    assert list(d.label_of("x", np.array([-5, 0, 5]))) == ["Tiedup", "Centered", "Flail"]
    assert list(d.label_of("y", np.array([-8, 0, 8]))) == ["Late", "OnTime", "Early"]
    assert list(d.label_of("z", np.array([-3, 0, 3]))) == ["Over", "Linedup", "Under"]


def test_labels_sit_exactly_on_savant_thresholds():
    """Probe the band edges, not just points far outside them.

    Values well past the threshold get the same label under any threshold that is
    smaller, so the test above passes with x at 4, y at 7 or z at 2 changed to
    anything lower. These assertions pin the actual numbers.
    """
    for axis, t, (lo, mid, hi) in (("x", 4.0, ("Tiedup", "Centered", "Flail")),
                                   ("y", 7.0, ("Late", "OnTime", "Early")),
                                   ("z", 2.0, ("Over", "Linedup", "Under"))):
        probe = np.array([-t - 0.01, -t, -t + 0.01, t - 0.01, t, t + 0.01])
        assert list(d.label_of(axis, probe)) == [lo, mid, mid, mid, mid, hi], axis


def test_label_of_marks_a_missing_value_unknown():
    assert list(d.label_of("z", np.array([np.nan, 0.0]))) == [d.UNKNOWN, "Linedup"]


def test_ladder_recovers_planted_x_and_flags_z():
    f = _synthetic()
    vx = d.verdict(d.run_ladder(f, "x"))
    assert vx.status == "recovered" and vx.best.r2 > 0.95 and vx.best.all_swing
    vz = d.verdict(d.run_ladder(f, "z"))
    assert vz.status == "fallback"


def test_ladder_climbs_to_the_candidate_that_carries_both_terms():
    f = _synthetic()
    v = d.verdict(d.run_ladder(f, "y"))
    assert v.status == "recovered"
    assert set(v.best.coef or next(iter(v.best.stand_models.values()))["coef"]) == {
        "icpt_y", "attack_direction"}
    single = [r for r in v.ladder if r.name == "icpt_y_raw"][0]
    assert single.r2 < 0.8, "the single-feature candidate must not clear the gate here"


def test_apply_reproduces_fit():
    f = _synthetic()
    res = d.verdict(d.run_ladder(f, "y")).best
    pred = d.apply(res, f)
    assert np.corrcoef(pred, f["sav_y"])[0, 1] > 0.97


def test_apply_round_trips_a_by_stand_fit():
    f = _synthetic()
    cand = [c for c in d.CANDIDATES["y"] if c.name == "icpt_y_by_stand"][0]
    res = d.fit_candidate(f, "y", cand)
    assert res.by_stand and set(res.stand_models) == {"R", "L"}
    pred = d.apply(res, f)
    assert not np.isnan(pred).any()
    assert np.corrcoef(pred, f["sav_y"])[0, 1] > 0.7


def test_verdict_on_an_empty_ladder_does_not_raise():
    v = d.verdict([])
    assert v.status == "fallback" and v.best is None


def test_step5_candidates_are_on_the_ladder():
    """Task 4 Step 5's expensive hypotheses, one per axis that fell back."""
    names = {ax: {c.name for c in d.CANDIDATES[ax]} for ax in ("x", "y", "z")}
    assert "x_full_geometry" in names["x"]
    assert {"z_zone_relative", "z_miss_decomp"} <= names["z"]
    from cp_lib.geometry import FEATURE_COLUMNS
    for ax, cands in d.CANDIDATES.items():
        for c in cands:
            for feat in c.features:
                assert feat in FEATURE_COLUMNS or feat == "miss_distance", (ax, c.name, feat)
