"""The pulled parquet's schema, and the player lookup that goes with it."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "notebooks" / "contact_point" / "data"

NEW_COLUMNS = [
    "non_magnus_vertical_acceleration", "non_magnus_horizontal_acceleration",
    "magnus_vertical_break", "non_magnus_vertical_break",
    "seam_shifted_wake_x", "seam_shifted_wake_y",
    "vaa_vs_expected", "haa_vs_expected", "expected_vaa",
    "release_spin_rate", "spin_axis", "induced_vb", "hb", "zone",
    "plate_z_normalized", "hit_location", "launch_speed_angle", "hyper_speed",
    "swing_take", "arm_angle",
]


def test_pull_asks_for_the_new_columns():
    from cp_lib import pull

    missing = [c for c in NEW_COLUMNS if c not in pull.COLUMNS]
    assert not missing, missing


@pytest.mark.parametrize("season", [2024, 2025, 2026])
def test_the_pulled_parquet_carries_them(season):
    """A column asked for but not written means the pull ran on the old list."""
    import pyarrow.parquet as pq

    p = DATA / f"swings_{season}.parquet"
    if not p.exists():
        pytest.skip(f"{p.name} has not been pulled")
    names = set(pq.read_schema(p).names)
    missing = [c for c in NEW_COLUMNS if c not in names]
    assert not missing, (season, missing)
