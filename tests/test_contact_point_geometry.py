import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import geometry as g  # noqa: E402


def _pitch(season=2025):
    # a 95 mph fastball thrown straight down the middle with gravity only
    return pd.DataFrame({
        "vx0": [0.0], "vy0": [-139.0], "vz0": [-5.0], "ax": [0.0], "ay": [25.0], "az": [-32.17],
        "plate_x": [0.0], "plate_z": [2.5], "stand": ["R"],
        "attack_angle": [10.0], "attack_direction": [0.0], "swing_path_tilt": [30.0],
        "intercept_ball_minus_batter_pos_x_inches": [30.0],
        "intercept_ball_minus_batter_pos_y_inches": [24.0],
        "bat_speed": [72.0], "miss_distance": [np.nan],
    })


def test_plate_reference_moves_in_2026():
    assert abs(g.plate_y_ft(2025) - 17 / 12) < 1e-9
    assert abs(g.plate_y_ft(2026) - 17 / 24) < 1e-9


def test_ball_state_reproduces_plate_columns():
    df = _pitch()
    st = g.ball_state_at_y(df, 2025, g.plate_y_ft(2025))
    assert abs(st["bx"].iloc[0] - 0.0) < 1e-6
    assert abs(st["bz"].iloc[0] - 2.5) < 1e-6
    assert 85 < st["speed_mph"].iloc[0] < 95
    assert st["descent_deg"].iloc[0] < 0  # ball is descending at the plate


def test_time_to_y_is_monotone_toward_the_plate():
    df = _pitch()
    ts = [g.time_to_y(df, y)[0] for y in (50.0, 30.0, 10.0, 1.417)]
    assert ts[0] == 0.0 or abs(ts[0]) < 1e-9
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_bat_frame_is_orthonormal_and_handed():
    fr = g.bat_frame(_pitch())
    v = fr[["v_x", "v_y", "v_z"]].to_numpy()[0]
    n = fr[["n_x", "n_y", "n_z"]].to_numpy()[0]
    b = fr[["b_x", "b_y", "b_z"]].to_numpy()[0]
    for u in (v, n, b):
        assert abs(np.linalg.norm(u) - 1) < 1e-9
    assert abs(v @ n) < 1e-9 and abs(v @ b) < 1e-9 and abs(n @ b) < 1e-9
    assert n[2] > 0            # normal points up-ish
    assert b[0] > 0            # RHB barrel points toward positive x (across the plate)
    assert abs(np.degrees(np.arcsin(v[2])) - 10.0) < 1e-6   # attack angle preserved


def test_bat_frame_mirrors_between_stands():
    df = pd.concat([_pitch(), _pitch().assign(stand="L")], ignore_index=True)
    df.loc[:, "attack_direction"] = 12.0
    fr = g.bat_frame(df)
    r = fr.iloc[0]
    l = fr.iloc[1]
    assert r["pull_sign"] == -1.0 and l["pull_sign"] == 1.0
    # same swing description, mirrored batters: x components flip, y and z do not
    for name in ("v", "n", "b"):
        assert abs(r[f"{name}_x"] + l[f"{name}_x"]) < 1e-9, name
        assert abs(r[f"{name}_y"] - l[f"{name}_y"]) < 1e-9, name
        assert abs(r[f"{name}_z"] - l[f"{name}_z"]) < 1e-9, name


def test_candidate_features_has_every_column():
    feats = g.candidate_features(_pitch(), 2025)
    for c in g.FEATURE_COLUMNS:
        assert c in feats.columns, c


def test_step5_features_and_the_all_swing_subset_exist():
    """These are what the fallback model is fit on, and no other test touches them."""
    for c in ("bz_minus_zone_mid", "zone_height", "z_from_miss", "icpt_y_over_ballspeed"):
        assert c in g.FEATURE_COLUMNS, c
    assert "z_from_miss" not in g.ALL_SWING_FEATURE_COLUMNS
    assert set(g.ALL_SWING_FEATURE_COLUMNS) == set(g.FEATURE_COLUMNS) - g.WHIFF_ONLY_FEATURES
    df = _pitch()
    df["sz_top"], df["sz_bot"], df["miss_distance"] = 3.4, 1.6, 5.0
    f = g.candidate_features(df, 2025)
    assert np.isfinite(f["bz_minus_zone_mid"].iloc[0])
    assert abs(f["zone_height"].iloc[0] - 21.6) < 1e-6
    assert abs(f["z_from_miss"].iloc[0]) <= 5.0
    # a frame with no strike zone columns still builds, with NaN in those slots
    f2 = g.candidate_features(_pitch(), 2025)
    assert np.isnan(f2["bz_minus_zone_mid"].iloc[0]) and np.isnan(f2["z_from_miss"].iloc[0])
    assert np.isfinite(f2["icpt_y_over_ballspeed"].iloc[0])
