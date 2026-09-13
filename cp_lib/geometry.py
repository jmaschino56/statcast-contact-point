"""Ball trajectory evaluation, the bat frame at contact, and candidate features.

Units: feet and seconds internally; inches, mph and ms at the boundaries.
Nothing here is fit. Every number is a closed-form function of public fields.

Statcast coordinates: x positive toward the catcher's right (first-base side),
y positive from home plate toward the pitcher, z up, feet. The 9-parameter fit
gives vx0, vy0, vz0, ax, ay, az at y = 50 ft; plate_x and plate_z are the ball
position at plate_y_ft(season), which moved from the front of the plate through
2025 to the middle of the plate in 2026.

Sign conventions measured on 2025 and recorded in
data/reports/attack_direction_sign.txt. See bat_frame's own comment block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FT_PER_S_TO_MPH = 3600.0 / 5280.0
IN_PER_FT = 12.0
Y_REF = 50.0
TILT_SIGN = +1.0   # barrel below the hands; see data/reports/attack_direction_sign.txt

FEATURE_COLUMNS = [
    "icpt_x", "icpt_y", "icpt_x_body", "icpt_x_field", "pull_sign",
    "attack_direction", "attack_direction_pull", "attack_angle", "swing_path_tilt", "bat_speed",
    "ball_speed_mph", "ball_in_per_ms", "descent_deg",
    "bx_icpt", "bz_icpt", "bx_plate", "bz_plate",
    "plane_z_at_icpt", "z_above_plane", "lateral_from_com_ft", "along_bat_from_com",
    "t_icpt_minus_plate_ms", "attack_minus_descent",
    # Task 4 Step 5 additions. Still closed-form, still all-swing except z_from_miss.
    "bz_minus_zone_mid", "zone_height", "z_from_miss", "icpt_y_over_ballspeed",
]

# z_from_miss is the only feature that needs miss_distance, so it is null on every
# contact swing. A model that has to score every swing must not use it.
WHIFF_ONLY_FEATURES = {"z_from_miss"}
ALL_SWING_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in WHIFF_ONLY_FEATURES]


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """A float column, or all-NaN when the frame does not carry it."""
    if name in df.columns:
        return df[name].to_numpy(float)
    return np.full(len(df), np.nan)


def plate_y_ft(season: int) -> float:
    return 17 / 24 if season >= 2026 else 17 / 12


def time_to_y(df: pd.DataFrame, y_ft) -> np.ndarray:
    """Seconds after the y = 50 reference at which the ball reaches y_ft (y decreasing toward the plate)."""
    vy, ay = df["vy0"].to_numpy(float), df["ay"].to_numpy(float)
    c = Y_REF - np.asarray(y_ft, dtype=float)
    disc = np.sqrt(np.maximum(vy * vy - 2.0 * ay * c, 0.0))
    return (-vy - disc) / ay


def initial_xz(df: pd.DataFrame, season: int):
    t = time_to_y(df, plate_y_ft(season))
    x0 = df["plate_x"].to_numpy(float) - df["vx0"].to_numpy(float) * t - 0.5 * df["ax"].to_numpy(float) * t * t
    z0 = df["plate_z"].to_numpy(float) - df["vz0"].to_numpy(float) * t - 0.5 * df["az"].to_numpy(float) * t * t
    return x0, z0


def ball_state_at_y(df: pd.DataFrame, season: int, y_ft) -> pd.DataFrame:
    x0, z0 = initial_xz(df, season)
    t = time_to_y(df, y_ft)
    vx0, vy0, vz0 = (df[c].to_numpy(float) for c in ("vx0", "vy0", "vz0"))
    ax, ay, az = (df[c].to_numpy(float) for c in ("ax", "ay", "az"))
    bvx, bvy, bvz = vx0 + ax * t, vy0 + ay * t, vz0 + az * t
    speed = np.sqrt(bvx ** 2 + bvy ** 2 + bvz ** 2)
    return pd.DataFrame({
        "bx": x0 + vx0 * t + 0.5 * ax * t * t,
        "by": np.broadcast_to(np.asarray(y_ft, float), t.shape).astype(float),
        "bz": z0 + vz0 * t + 0.5 * az * t * t,
        "bvx": bvx, "bvy": bvy, "bvz": bvz,
        "speed_mph": speed * FT_PER_S_TO_MPH,
        "descent_deg": np.degrees(np.arctan2(bvz, -bvy)),
        "t": t,
    }, index=df.index)


def bat_frame(df: pd.DataFrame, tilt_sign: float = TILT_SIGN) -> pd.DataFrame:
    """Unit vectors at contact. pull_sign: -1 for RHB (pull = negative x), +1 for LHB.

    v is the sweet spot's velocity direction, n the swing-plane normal (up-ish),
    b the bat axis pointing toward the barrel (across the plate, away from the
    batter's body). attack_direction is signed toward pull, so its field-x
    component carries pull_sign; measured on 2025 balls in play, see
    data/reports/attack_direction_sign.txt.

    tilt_sign controls which way swing_path_tilt leans the plane: -1 puts the
    barrel above the hands, +1 below. Measured, not assumed; see the same report.
    """
    pull = np.where(df["stand"].to_numpy() == "L", 1.0, -1.0)
    aa = np.radians(df["attack_angle"].to_numpy(float))
    ad = np.radians(df["attack_direction"].to_numpy(float))
    tilt = np.radians(df["swing_path_tilt"].to_numpy(float))
    vx = np.cos(aa) * np.sin(ad) * pull
    vy = np.cos(aa) * np.cos(ad)
    vz = np.sin(aa)
    v = np.stack([vx, vy, vz], axis=1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    up = np.array([0.0, 0.0, 1.0])
    up_perp = up - (v @ up)[:, None] * v
    up_perp /= np.linalg.norm(up_perp, axis=1, keepdims=True)
    side = np.cross(v, up_perp)                       # horizontal-ish, perpendicular to v
    # barrel side is across the plate from the batter, i.e. -pull_sign in field x
    barrel_side = np.where((side[:, 0] * -pull) > 0, 1.0, -1.0)[:, None] * side
    n = np.cos(tilt)[:, None] * up_perp + tilt_sign * np.sin(tilt)[:, None] * barrel_side
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    b = np.cross(n, v)
    b *= np.where((b[:, 0] * -pull) > 0, 1.0, -1.0)[:, None]   # point toward the barrel
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    n = np.cross(v, b)                                # re-orthogonalize; keep up-ish
    n *= np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]
    out = pd.DataFrame(index=df.index)
    for name, arr in (("v", v), ("n", n), ("b", b)):
        out[f"{name}_x"], out[f"{name}_y"], out[f"{name}_z"] = arr[:, 0], arr[:, 1], arr[:, 2]
    out["pull_sign"] = pull
    return out


def candidate_features(df: pd.DataFrame, season: int, y_com_ft: float = 0.0,
                       tilt_sign: float = None) -> pd.DataFrame:
    """Every cheap and mid-cost candidate quantity discovery will try, one row per swing.

    y_com_ft places the batter's center of mass on the field y axis so the
    intercept depth (inches, relative to the batter) can be turned into a field y.
    Discovery grid-searches it; 0.0 is the plate's point.
    """
    fr = bat_frame(df, TILT_SIGN if tilt_sign is None else tilt_sign)
    icpt_x = df["intercept_ball_minus_batter_pos_x_inches"].to_numpy(float)
    icpt_y = df["intercept_ball_minus_batter_pos_y_inches"].to_numpy(float)
    y_icpt_ft = y_com_ft + icpt_y / IN_PER_FT
    at_icpt = ball_state_at_y(df, season, y_icpt_ft)
    at_plate = ball_state_at_y(df, season, plate_y_ft(season))
    pull = fr["pull_sign"].to_numpy()
    nx, ny, nz = fr["n_x"].to_numpy(), fr["n_y"].to_numpy(), fr["n_z"].to_numpy()
    dx = at_plate["bx"].to_numpy() - at_icpt["bx"].to_numpy()
    dy = at_plate["by"].to_numpy() - at_icpt["by"].to_numpy()
    dz = at_plate["bz"].to_numpy() - at_icpt["bz"].to_numpy()
    z_above = (dx * nx + dy * ny + dz * nz) * IN_PER_FT
    plane_z_at_icpt = at_icpt["bz"].to_numpy() - (dx * nx + dy * ny) / np.where(np.abs(nz) < 1e-6, 1e-6, nz)
    lateral_from_com_ft = np.abs(icpt_x) / IN_PER_FT
    bx, by_, bz = fr["b_x"].to_numpy(), fr["b_y"].to_numpy(), fr["b_z"].to_numpy()
    along_bat_from_com = (icpt_x * -pull) * bx + icpt_y * by_   # inches, batter as the origin
    sz_top, sz_bot = _col(df, "sz_top"), _col(df, "sz_bot")
    zone_mid = (sz_top + sz_bot) / 2.0
    miss = _col(df, "miss_distance")
    z_from_miss = np.sign(z_above) * np.minimum(np.abs(z_above), miss)
    ball_in_per_ms = at_icpt["speed_mph"].to_numpy() * 5280 * 12 / 3600 / 1000
    return pd.DataFrame({
        "icpt_x": icpt_x, "icpt_y": icpt_y,
        # the column is already body-relative and never negative (measured, 2025):
        # icpt_x_body IS icpt_x, and the field-x displacement is the separate feature
        "icpt_x_body": icpt_x, "icpt_x_field": icpt_x * -pull, "pull_sign": pull,
        "attack_direction": df["attack_direction"].to_numpy(float),
        "attack_direction_pull": df["attack_direction"].to_numpy(float) * pull,
        "attack_angle": df["attack_angle"].to_numpy(float),
        "swing_path_tilt": df["swing_path_tilt"].to_numpy(float),
        "bat_speed": df["bat_speed"].to_numpy(float),
        "ball_speed_mph": at_icpt["speed_mph"].to_numpy(),
        "ball_in_per_ms": at_icpt["speed_mph"].to_numpy() * 5280 * 12 / 3600 / 1000,
        "descent_deg": at_icpt["descent_deg"].to_numpy(),
        "bx_icpt": at_icpt["bx"].to_numpy(), "bz_icpt": at_icpt["bz"].to_numpy(),
        "bx_plate": at_plate["bx"].to_numpy(), "bz_plate": at_plate["bz"].to_numpy(),
        "plane_z_at_icpt": plane_z_at_icpt, "z_above_plane": z_above,
        "lateral_from_com_ft": lateral_from_com_ft, "along_bat_from_com": along_bat_from_com,
        "t_icpt_minus_plate_ms": (at_icpt["t"].to_numpy() - at_plate["t"].to_numpy()) * 1000.0,
        "attack_minus_descent": df["attack_angle"].to_numpy(float) - at_icpt["descent_deg"].to_numpy(),
        # where the ball sat in the strike zone at the intercept depth. The swing
        # plane's height tracks the zone, and the zone is the only published proxy
        # for the batter's own scale (no batter heights were cached).
        "bz_minus_zone_mid": (at_icpt["bz"].to_numpy() - zone_mid) * IN_PER_FT,
        "zone_height": (sz_top - sz_bot) * IN_PER_FT,
        # Task 4 Step 5 hypothesis 3: z as a component of the 3D miss vector
        "z_from_miss": z_from_miss,
        # the ball's travel time over the contact depth: the one quantity the
        # discovery ladder showed Savant's own y is built on
        "icpt_y_over_ballspeed": icpt_y / np.where(ball_in_per_ms <= 0, np.nan, ball_in_per_ms),
    }, index=df.index)
