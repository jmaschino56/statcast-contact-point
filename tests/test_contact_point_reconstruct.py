import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import reconstruct as r  # noqa: E402


def _df(n=3):
    return pd.DataFrame({
        "vx0": [0.0] * n, "vy0": [-139.0] * n, "vz0": [-5.0] * n,
        "ax": [0.0] * n, "ay": [25.0] * n, "az": [-32.17] * n,
        "plate_x": [0.0] * n, "plate_z": [2.5] * n, "stand": ["R"] * n,
        "attack_angle": [10.0] * n, "attack_direction": [0.0] * n, "swing_path_tilt": [30.0] * n,
        "intercept_ball_minus_batter_pos_x_inches": [30.0] * n,
        "intercept_ball_minus_batter_pos_y_inches": [24.0] * n,
        "bat_speed": [72.0] * n, "release_speed": [95.0] * n,
    })


def _verdicts():
    return {
        "x": r.FormulaSpec(status="recovered", all_swing=True, coef={"icpt_x_body": 1.0},
                           intercept=-30.0, by_stand=False, stand_models={}),
        "y": r.FormulaSpec(status="recovered", all_swing=False, coef={"miss_distance": 1.0},
                           intercept=0.0, by_stand=False, stand_models={}),
        "z": r.FormulaSpec(status="fallback", all_swing=False, coef={}, intercept=0.0,
                           by_stand=False, stand_models={}),
    }


def test_ev_ceiling_matches_the_deck():
    assert abs(r.ev_ceiling(77.0, 47.0) - 105.52) < 0.01


def test_x_from_ev_deficit_is_monotone_signed_and_bounded():
    d = np.array([0.0, 2.0, 6.0, 12.0, 40.0])
    tip = r.x_from_ev_deficit(d, toward_handle=np.zeros(5, bool))
    hand = r.x_from_ev_deficit(d, toward_handle=np.ones(5, bool))
    assert np.all(np.diff(tip) >= 0) and tip[0] == 0 and tip[-1] <= 14
    assert np.all(hand <= 0) and abs(hand[2]) < tip[2]   # handle loses EV faster


def test_z_from_launch_sign_and_clip():
    z = r.z_from_launch(np.array([30.0, 10.0, -20.0]), np.array([10.0, 10.0, 10.0]),
                        np.array([True, True, True]))
    assert z[0] > 0 and abs(z[1]) < 1e-9 and z[2] < 0
    assert np.all(np.abs(z) <= 2.75)


def test_z_from_launch_clips_wider_for_fouls():
    la = np.array([300.0, 300.0])
    z = r.z_from_launch(la, np.array([0.0, 0.0]), np.array([True, False]))
    assert abs(z[0] - 2.75) < 1e-9 and abs(z[1] - 6.0) < 1e-9


def test_reconstruct_routes_by_source():
    df = _df()
    df["miss_distance"] = [5.0, np.nan, np.nan]
    df["contact_type"] = ["whiff", "in_play", "foul"]
    df["launch_speed"] = [np.nan, 100.0, 60.0]
    df["launch_angle"] = [np.nan, 20.0, 45.0]
    out = r.reconstruct(df, 2025, _verdicts(), y_com_ft=0.0)
    assert list(out["source_x"]) == ["formula"] * 3
    assert list(out["source_y"]) == ["formula", "inversion", "inversion"]
    # a fallback axis is routed like a whiff-only one: the whiff waits for the
    # fallback model, the contact swings take the outcome inversion now
    assert list(out["source_z"]) == ["fallback", "inversion", "inversion"]
    assert np.isnan(out["z_hat"].iloc[0])
    assert np.isfinite(out["z_hat"].iloc[1])
    assert abs(out["x_hat"].iloc[0] - 0.0) < 1e-9


def test_reconstruct_marks_contact_swings_with_no_launch_data():
    """A foul with no launch data has no outcome to invert; it must not read as centered."""
    df = _df()
    df["miss_distance"] = [np.nan] * 3
    df["contact_type"] = ["foul"] * 3
    df["launch_speed"] = [95.0, np.nan, np.nan]
    df["launch_angle"] = [20.0, np.nan, 15.0]
    v = _verdicts()
    v["x"] = r.FormulaSpec("recovered", False, {"icpt_x_body": 1.0}, -30.0, False, {})
    v["z"] = r.FormulaSpec("recovered", False, {"z_above_plane": 1.0}, 0.0, False, {})
    out = r.reconstruct(df, 2025, v, y_com_ft=0.0)
    assert list(out["source_x"]) == ["inversion", "no_outcome", "no_outcome"]
    assert np.isnan(out["x_hat"].iloc[1]) and np.isnan(out["x_hat"].iloc[2])
    assert list(out["source_z"]) == ["inversion", "no_outcome", "inversion"]
    assert out.attrs["reconstruct_stats"]["no_outcome_x"] == 2


def test_x_from_ev_deficit_keeps_rising_past_the_published_curve():
    """The Driveline curve's last published point is 40 mph of deficit at 14 inches
    on the tip side. Clamping there put 12,221 of 2025's swings on exactly 14
    inches, a spike Savant's histogram does not have. The quantile map sets the
    final scale, so what this function has to preserve is rank order, not inches."""
    d = np.array([35.0, 40.0, 45.0, 60.0, 90.0, 200.0])
    tip = r.x_from_ev_deficit(d, np.zeros(len(d), bool))
    assert np.all(np.diff(tip) > 0), tip
    hand = r.x_from_ev_deficit(d, np.ones(len(d), bool))
    assert np.all(np.diff(hand) < 0), hand          # handle side runs negative
    assert abs(float(r.x_from_ev_deficit(np.array([40.0]), np.array([False]))[0]) - 14.0) < 1e-9
    assert float(r.x_from_ev_deficit(np.array([0.0]), np.array([False]))[0]) == 0.0


def test_a_contact_swing_with_no_outcome_falls_back_to_the_all_swing_formula():
    """A foul with no exit velocity has nothing to invert, so x used to stay NaN
    and the swing was reported Unknown rather than centered. One foul ball in eight
    is like that, and it is the whole of the last unclosed validation cell.

    The x inversion needs exit velocity but the discovery ladder's own winner for
    x does not: x_full_geometry is an all-swing formula. It is weaker evidence than
    the inversion, so it is used only where the inversion has nothing, and it gets
    its own provenance rather than being called a formula result.
    """
    df = _df(2)
    df["contact_type"] = ["in_play", "foul"]
    df["miss_distance"] = [np.nan, np.nan]
    df["launch_speed"] = [95.0, np.nan]        # the foul has no exit velocity
    df["launch_angle"] = [15.0, np.nan]
    verdicts = {
        "x": r.FormulaSpec(status="fallback", all_swing=True, coef={"icpt_x_body": 1.0},
                           intercept=-30.0, by_stand=False, stand_models={}),
        "y": r.FormulaSpec(status="recovered", all_swing=True, coef={"icpt_y": 1.0},
                           intercept=0.0, by_stand=False, stand_models={}),
        # z's real winner is by stand and whiff-only, so it must NOT be used here
        "z": r.FormulaSpec(status="fallback", all_swing=False, coef={}, intercept=0.0,
                           by_stand=True,
                           stand_models={"R": {"coef": {"icpt_x_body": 1.0}, "intercept": 0.0},
                                         "L": {"coef": {"icpt_x_body": 1.0}, "intercept": 0.0}}),
    }
    out = r.reconstruct(df, 2025, verdicts, 0.0, tilt_sign=1.0)
    src = list(out["source_x"])
    assert src == ["inversion", "geometry"], src
    assert np.isfinite(out["x_hat"].to_numpy()).all(), out["x_hat"].tolist()
    # z's best candidate is whiff-only, so it has nothing to offer and stays NaN
    assert list(out["source_z"]) == ["inversion", "no_outcome"], list(out["source_z"])
    assert np.isnan(out["z_hat"].to_numpy()[1])
