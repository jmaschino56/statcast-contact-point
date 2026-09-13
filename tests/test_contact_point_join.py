import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cp_lib import join  # noqa: E402


def _swings():
    return pd.DataFrame({
        "game_pk": [1, 1, 1, 2],
        "batter": [10, 10, 10, 11],
        "pitcher": [20, 20, 20, 21],
        "pitch_type": ["FF", "SL", "SL", "FF"],
        "description": ["swinging_strike", "foul", "hit_into_play", "missed_bunt"],
        "bat_speed": [72.0, 70.0, 75.0, 30.0],
        "miss_distance": [4.1234567, np.nan, np.nan, 9.0],
    })


def test_load_shape_columns_are_derived():
    df = join.derive_columns(_swings(), season=2025)
    assert list(df["contact_type"]) == ["whiff", "foul", "in_play", "whiff"]
    assert list(df["is_bunt"]) == [False, False, False, True]
    assert (df["season"] == 2025).all()


def test_swing_filter_rules_nest():
    df = join.derive_columns(_swings(), season=2025)
    a = join.swing_filter(df, "bat_speed")
    b = join.swing_filter(df, "bat_speed_nobunt")
    c = join.swing_filter(df, "bat_speed_nobunt_typical")
    assert a.sum() == 4 and b.sum() == 3 and c.sum() == 3
    assert (b <= a).all() and (c <= b).all()


def test_join_tails_exact_then_nearest():
    df = join.derive_columns(_swings(), season=2025)
    tails = pd.DataFrame({
        "play_id": ["p1", "p2"],
        "game_pk": [1, 2],
        "batter": [10, 11],
        "pitcher": [20, 21],
        "pitch_type": ["FF", "FF"],
        "miss_distance_inches": [4.1234567, 9.0004],
        "sav_x": [1.0, 2.0], "sav_y": [0.0, 0.0], "sav_z": [0.0, 0.0],
    })
    out = join.join_tails(df, tails)
    assert list(out["join_method"]) == ["exact", "nearest"]
    assert out.loc[out.play_id == "p1", "description"].item() == "swinging_strike"


def test_join_tails_survives_shared_columns_and_fanout():
    """Real tails carry game_date, season and pitch_type; real swing rows can share a join key."""
    df = join.derive_columns(_swings(), season=2025)
    df["game_date"] = pd.to_datetime("2025-05-01").date()
    # a second swing row sharing (game_pk, batter, pitcher, miss_distance) with row 0
    dup = df.iloc[[0]].copy()
    dup["pitch_number"] = 2
    df = pd.concat([df, dup], ignore_index=True)
    tails = pd.DataFrame({
        "play_id": ["p1"],
        "game_pk": [1], "batter": [10], "pitcher": [20],
        "pitch_type": ["FF"], "game_date": [pd.to_datetime("2025-05-01").date()],
        "season": [2025], "miss_distance_inches": [4.1234567],
        "sav_x": [1.0], "sav_y": [0.0], "sav_z": [0.0],
    })
    out = join.join_tails(df, tails)
    assert len(out) == 1, "the exact merge must not fan out past one row per play_id"
    for c in ("game_date", "season", "pitch_type"):
        assert f"{c}_x" not in out.columns and f"{c}_y" not in out.columns, list(out.columns)
        assert list(out.columns).count(c) == 1, list(out.columns)
    assert out["description"].item() == "swinging_strike"
    assert out.attrs["join_stats"]["fanout_rows_dropped"] == 1


def test_join_tails_reports_unmatched():
    df = join.derive_columns(_swings(), season=2025)
    tails = pd.DataFrame({
        "play_id": ["p9"], "game_pk": [99], "batter": [10], "pitcher": [20],
        "pitch_type": ["FF"], "miss_distance_inches": [1.0],
        "sav_x": [1.0], "sav_y": [0.0], "sav_z": [0.0],
    })
    out = join.join_tails(df, tails)
    assert list(out["join_method"]) == ["unmatched"]
    assert out.attrs["join_stats"]["unmatched"] == 1


def test_denominator_report_scores_each_rule(monkeypatch):
    df = join.derive_columns(_swings(), season=2025)
    fake = pd.DataFrame({"id": [10, 11], "n_swings": [3, 1]})
    monkeypatch.setattr(join, "load_leaderboard", lambda season, type_, split=None: fake)
    rep = join.denominator_report(df, 2025, "batter")
    assert set(rep["rule"]) == {"bat_speed", "bat_speed_nobunt", "bat_speed_nobunt_typical"}
    assert rep.loc[rep.rule == "bat_speed", "total_ours"].item() == 4


def test_foul_tip_counts_as_a_whiff_like_savant():
    """Savant's bat_contact_code 9 includes foul tips: measured on 2025, our whiff
    rate is 0.2346 without them and 0.2565 with them, against Savant's 0.2569."""
    df = pd.DataFrame({
        "description": ["foul_tip", "bunt_foul_tip", "foul", "hit_into_play",
                        "swinging_strike", "swinging_strike_blocked"],
        "bat_speed": [70.0] * 6, "miss_distance": [np.nan] * 6,
        "game_pk": [1] * 6, "batter": [1] * 6, "pitcher": [1] * 6, "pitch_type": ["FF"] * 6,
    })
    out = join.derive_columns(df, season=2025)
    assert list(out["contact_type"]) == ["whiff", "whiff", "foul", "in_play", "whiff", "whiff"]


def test_residual_report_gives_the_per_player_mismatch_distribution(monkeypatch):
    """Spec section 3 asks for the residual mismatch distribution, not just the
    median relative error the rule was chosen on. _swings() gives batter 10 three
    swings under the plain bat_speed rule and batter 11 one; against a leaderboard
    claiming 2 and 1 the mismatches are exactly [1, 0]."""
    df = join.derive_columns(_swings(), season=2025)
    fake = pd.DataFrame({"id": [10, 11], "n_swings": [2, 1]})
    monkeypatch.setattr(join, "load_leaderboard", lambda season, type_, split=None: fake)
    res = join.residual_report(df, 2025, "batter", "bat_speed")
    assert res["players"] == 2
    assert res["total_ours"] == 4 and res["total_savant"] == 3
    assert res["median_diff"] == 0.5
    assert res["exact_match"] == 0.5
    assert res["p05"] == 0.05 and res["p95"] == 0.95
