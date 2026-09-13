"""Per-batter signal that is not on the pitch row.

Three sources, none of which is a Savant label:

  1. Traits we compute ourselves from the pulled swings, PER SEASON. A holdout
     season uses its own swings and never the fit season's, which is why every
     function here takes the season and filters rather than grouping over
     whatever it was handed.
  2. The MLB Stats API's people endpoint: height, weight, bat side, birth date.
     Cached once; it is biography and does not change with the season.
  3. Savant's swing-path leaderboard, which publishes the batter's own position
     in the box, foot separation and foot angle, and a swing plane angle. Read
     the docstring on `swing_path` before using it: the endpoint ignores the
     season parameter.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from . import DATA
from .savant_truth import BASE, cached_get

TRAIT_FLOOR = 100          # swings a batter needs before their own slope is used
PLAYERS_PARQUET = DATA / "players.parquet"
MANIFEST = DATA / "traits_manifest.jsonl"
BATCH = 100
STATS_API = "https://statsapi.mlb.com/api/v1/people"

MEAN_SD = {
    "tilt": "swing_path_tilt",
    "aa": "attack_angle",
    "bs": "bat_speed",
    "ix": "intercept_ball_minus_batter_pos_x_inches",
    "iy": "intercept_ball_minus_batter_pos_y_inches",
}


def _note(rec: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rec["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# 1. traits from our own swings, one season at a time
# ----------------------------------------------------------------------------
def batter_traits(df: pd.DataFrame, season: int, min_swings: int = TRAIT_FLOOR) -> pd.DataFrame:
    """Mean and spread of each batter's swing, for ONE season.

    The slope of attack angle on pitch height is the one trait that needs a fit
    rather than a moment, and a batter under the floor gets the league slope
    instead of their own noisy one. The league slope is on the frame's attrs so a
    caller can see what a short sample was given.
    """
    d = df[df["season"] == season] if "season" in df else df
    g = d.groupby("batter")
    out = pd.DataFrame(index=g.size().index)
    out["n_swings"] = g.size()
    for short, col in MEAN_SD.items():
        if col not in d:
            continue
        out[f"{short}_mean"] = g[col].mean()
        out[f"{short}_sd"] = g[col].std()
    for col, name in (("sz_top", "sz_top_med"), ("sz_bot", "sz_bot_med")):
        if col in d:
            out[name] = g[col].median()

    league = _slope(d["plate_z"].to_numpy(float), d["attack_angle"].to_numpy(float))
    slopes = {}
    for b, gg in d.groupby("batter"):
        if len(gg) >= min_swings:
            s = _slope(gg["plate_z"].to_numpy(float), gg["attack_angle"].to_numpy(float))
            slopes[b] = league if not np.isfinite(s) else s
        else:
            slopes[b] = league
    out["aa_on_plate_z"] = pd.Series(slopes)
    out.attrs["league_aa_on_plate_z"] = league
    out.attrs["season"] = season
    out.index.name = "batter"
    return out


def _slope(x, y) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    v = np.var(x[m], ddof=1)
    return float(np.cov(x[m], y[m], ddof=1)[0, 1] / v) if v > 0 else np.nan


def add_trait_features(df: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Join the traits on and add each swing's deviation from its batter's mean."""
    out = df.copy()
    j = tr.reindex(out["batter"].to_numpy())
    for c in tr.columns:
        out[f"bt_{c}"] = j[c].to_numpy()
    for short, col in MEAN_SD.items():
        if col in out and f"{short}_mean" in tr:
            out[f"{short}_dev"] = out[col].to_numpy(float) - j[f"{short}_mean"].to_numpy()
    return out


# ----------------------------------------------------------------------------
# 2. MLB Stats API biography
# ----------------------------------------------------------------------------
def parse_height(raw) -> int | None:
    """\"6' 2\\\"\" -> 74 inches. Savant and the Stats API both write it this way."""
    if raw is None or not isinstance(raw, str):
        return None
    m = re.match(r"\s*(\d+)\s*'\s*(\d+)?\s*\"?\s*$", raw)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2) or 0)


def fetch_players(ids, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "contact-point research")
    ids = sorted({int(i) for i in ids if pd.notna(i)})
    rows = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        r = s.get(STATS_API, params={"personIds": ",".join(map(str, chunk))}, timeout=60)
        r.raise_for_status()
        for p in r.json().get("people", []):
            rows.append({
                "batter": int(p["id"]),
                "height_in": parse_height(p.get("height")),
                "weight_lb": p.get("weight"),
                "bats": (p.get("batSide") or {}).get("code"),
                "throws": (p.get("pitchHand") or {}).get("code"),
                "birth_date": p.get("birthDate"),
            })
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if len(df):
        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    return df


def load_players(ids=None) -> pd.DataFrame:
    """Cached biography, fetched once. Pass ids to top up a cache that is short."""
    have = pd.read_parquet(PLAYERS_PARQUET) if PLAYERS_PARQUET.exists() else pd.DataFrame()
    if ids is None:
        return have
    want = sorted({int(i) for i in ids if pd.notna(i)})
    known = set(have["batter"]) if len(have) else set()
    missing = [i for i in want if i not in known]
    if missing:
        got = fetch_players(missing)
        have = pd.concat([have, got], ignore_index=True) if len(have) else got
        have = have.drop_duplicates("batter")
        have.to_parquet(PLAYERS_PARQUET, index=False)
        _note({"source": "statsapi_people", "asked": len(missing), "got": len(got),
               "cache_rows": len(have)})
    return have


def add_player_features(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    j = players.set_index("batter").reindex(out["batter"].to_numpy())
    out["pl_height_in"] = pd.to_numeric(j["height_in"], errors="coerce").to_numpy()
    out["pl_weight_lb"] = pd.to_numeric(j["weight_lb"], errors="coerce").to_numpy()
    if "birth_date" in j and "game_date" in out:
        age = ((pd.to_datetime(out["game_date"].to_numpy())
                - pd.to_datetime(j["birth_date"].to_numpy())).days / 365.25)
        out["pl_age"] = age
    return out


# ----------------------------------------------------------------------------
# 3. Savant swing path
# ----------------------------------------------------------------------------
SWING_PATH_URL = (f"{BASE}/leaderboard/bat-tracking/swing-path-attack-angle"
                  "?season=2025&type=batter&minSwings=1&csv=true")
SWING_PATH_HTML = (f"{BASE}/leaderboard/bat-tracking/swing-path-attack-angle"
                   "?season=2025&type=batter&minSwings=1")


def swing_path() -> pd.DataFrame:
    """Savant's per-batter swing-path slate: 652 batters, one row each.

    It carries what nothing else here does: the batter's own x and y position in
    the box, foot separation and foot angle, the swing's plane angle, and the
    intercept depth measured from the PLATE as well as from the batter, which is
    the offset between the two reference frames.

    NOT PER SEASON, and the endpoint does not say so. `season=2024`,
    `season=2026`, `season%5B%5D=2025` and no season at all all return the same
    652 rows with the same values to the last decimal. The slate is 2026, the
    current season: it covers 1.000 of our 2026 swings, 0.923 of 2025 and 0.827
    of 2024, so a 2024 batter is on it only if they are still playing now. That
    makes it a batter trait with no season attached, and it cannot be used the way
    `batter_traits` is. `minSwings=1` widens it from the default 202 rows to 652;
    `min=1`, which widens the timing leaderboard, does nothing here.
    """
    csv_p = cached_get(SWING_PATH_URL, "csv")
    df = pd.read_csv(csv_p, encoding="utf-8-sig").dropna(subset=["id"])
    df = df.astype({"id": int}).rename(columns={"id": "batter"})
    html = cached_get(SWING_PATH_HTML, "swingpath_html").read_text(encoding="utf-8")
    m = re.search(r"\bdata\s*=\s*(\[\{.*?\}\]);", html, re.S)
    if m:
        rich = pd.DataFrame(json.loads(m.group(1)))
        keep = ["id", "avg_plane_vertical_angle", "avg_batter_x_position",
                "avg_batter_y_position", "avg_foot_sep0", "avg_foot_angle0",
                "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter",
                "rate_ideal_attack_angle", "attack_direction_pullopp"]
        rich = rich[[c for c in keep if c in rich]].dropna(subset=["id"])
        rich = rich.astype({"id": int}).rename(columns={"id": "batter"})
        df = df.merge(rich, on="batter", how="left", suffixes=("", "_rich"))
    return df


SWING_PATH_FEATURES = [
    "avg_plane_vertical_angle", "avg_batter_x_position", "avg_batter_y_position",
    "avg_foot_sep0", "avg_foot_angle0", "avg_intercept_y_vs_plate",
    "avg_intercept_y_vs_batter", "rate_ideal_attack_angle", "swing_tilt",
    "ideal_attack_angle_rate",
]


def add_swing_path_features(df: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in SWING_PATH_FEATURES if c in sp]
    j = sp.drop_duplicates("batter").set_index("batter").reindex(out["batter"].to_numpy())
    for c in cols:
        out[f"sp_{c}"] = pd.to_numeric(j[c], errors="coerce").to_numpy()
    if "avg_intercept_y_vs_plate" in cols and "avg_intercept_y_vs_batter" in cols:
        out["sp_frame_offset"] = (out["sp_avg_intercept_y_vs_batter"]
                                  - out["sp_avg_intercept_y_vs_plate"])
    return out


# ----------------------------------------------------------------------------
# everything together
# ----------------------------------------------------------------------------
# Per-pitch columns from the Task 14 re-pull that a whiff can carry. Outcome
# columns (hit_location, launch_speed_angle, hyper_speed) are deliberately absent:
# a whiff has none of them, and the whiff model is what these features are for.
PITCH_FEATURES = [
    "non_magnus_vertical_acceleration", "non_magnus_horizontal_acceleration",
    "magnus_vertical_break", "non_magnus_vertical_break",
    "seam_shifted_wake_x", "seam_shifted_wake_y",
    "vaa_vs_expected", "haa_vs_expected", "expected_vaa",
    "release_spin_rate", "induced_vb", "hb", "zone", "plate_z_normalized",
    "arm_angle", "spin_axis_num",
    "release_extension", "effective_speed", "release_pos_x", "release_pos_z",
    "balls", "strikes",
]


def extended_features(df: pd.DataFrame, feats: pd.DataFrame, season: int,
                      players: pd.DataFrame | None = None,
                      sp: pd.DataFrame | None = None,
                      tr: pd.DataFrame | None = None) -> pd.DataFrame:
    """The geometry features plus everything Task 14 added, aligned to df's rows.

    Traits are computed on `season` only unless a table is passed in. Nothing here
    reads a Savant per-swing label.
    """
    out = feats.copy()
    out.index = df.index
    src = df.copy()
    if "spin_axis" in src:
        src["spin_axis_num"] = pd.to_numeric(src["spin_axis"], errors="coerce")
    for c in PITCH_FEATURES:
        if c in src:
            out[c] = pd.to_numeric(src[c], errors="coerce").to_numpy()
    if "season" not in src:
        src["season"] = season
    if tr is None:
        tr = batter_traits(src, season)
    base = src[["batter"] + [c for c in MEAN_SD.values() if c in src]]
    joined = add_trait_features(base, tr)
    for c in joined.columns:
        if c.startswith("bt_") or c.endswith("_dev"):
            out[c] = joined[c].to_numpy()
    if players is not None and len(players):
        pf = add_player_features(src[["batter"] + (["game_date"] if "game_date" in src else [])],
                                 players)
        for c in ("pl_height_in", "pl_weight_lb", "pl_age"):
            if c in pf:
                out[c] = pf[c].to_numpy()
    if sp is not None and len(sp):
        spf = add_swing_path_features(src[["batter"]], sp)
        for c in spf.columns:
            if c.startswith("sp_"):
                out[c] = spf[c].to_numpy()
    return out


def numeric_feature_columns(frame: pd.DataFrame, exclude=()) -> list:
    """Every column a LightGBM fit can take: numeric, not all null, not excluded."""
    bad = set(exclude) | {"sav_x", "sav_y", "sav_z", "batter", "pitcher", "lists"}
    out = []
    for c in frame.columns:
        if c in bad or not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        if frame[c].notna().any():
            out.append(c)
    return out


def tails_features(season: int, y_com_ft: float, tilt_sign: float,
                   players: pd.DataFrame | None = None,
                   sp: pd.DataFrame | None = None) -> pd.DataFrame:
    """The labeled whiff tails with every feature, ready for the fallback fit.

    join_tails returns TAIL rows with the swing merged in, on a fresh index, so
    the features are built from the joined frame. Indexing the season frame by
    that index takes the wrong swings, silently: it moved the baseline pitcher
    category error from 0.0204 to 0.0273 with the feature set unchanged.
    """
    from . import join as _join
    from .geometry import candidate_features
    from .savant_truth import load_tails

    d = _join.load_swings(season)
    d = d[_join.swing_filter(d, "bat_speed_nobunt")].reset_index(drop=True)
    j = _join.join_tails(d, load_tails(season))
    lab = j[j["join_method"] != "unmatched"].copy().reset_index(drop=True)
    f = candidate_features(lab, season, y_com_ft=y_com_ft, tilt_sign=tilt_sign)
    f["stand"] = lab["stand"].to_numpy()
    if "miss_distance" in lab:
        f["miss_distance"] = lab["miss_distance"].to_numpy(float)
    f["icpt_y_over_ballspeed"] = f["icpt_y"] / f["ball_in_per_ms"].replace(0, np.nan)
    out = extended_features(lab, f, season, players=players, sp=sp)
    for c in ("sav_x", "sav_y", "sav_z", "batter", "pitcher", "contact_type"):
        if c in lab:
            out[c] = lab[c].to_numpy()
    out.attrs["join_stats"] = j.attrs.get("join_stats", {})
    return out


# Features the fallback model may use. miss_distance is banned twice over: it is
# whiff-only, and it is what Savant selected the tails on, so a model given it
# learns the selection. bt_n_swings is a sample size, not a swing trait.
FALLBACK_BANNED = {"miss_distance", "z_from_miss", "bt_n_swings"}


# The widened feature set helps x and hurts z, measured on both holdouts. On x
# every centered scatter improves: 2024 pitcher slope 0.738 to 0.755 with r 0.795
# to 0.799, 2026 pitcher 0.715 to 0.738 with r 0.748 to 0.757, and the batter error
# falls from 0.0504 to 0.0489 and 0.0504 to 0.0476. On z the batter lined-up
# scatter goes the other way on BOTH holdouts, r 0.745 to 0.708 in 2024 and 0.726
# to 0.714 in 2026, with the error rising in step. So z keeps the geometry
# features and x takes everything.
WIDENED_AXES = ("x",)


def fallback_columns(frame: pd.DataFrame, base: list, axis: str | None = None) -> list:
    """The base geometry features, plus the Task 14 features on the axes they help.

    Pass the axis to get that axis's list. Without one the widened list comes back,
    which is what a caller wants when it is only counting features.
    """
    if axis is not None and axis not in WIDENED_AXES:
        return [c for c in base if c in frame and frame[c].notna().any()]
    extra = ([c for c in PITCH_FEATURES]
             + [c for c in frame.columns if c.startswith(("bt_", "pl_", "sp_"))]
             + [c for c in frame.columns if c.endswith("_dev")])
    out = []
    for c in list(base) + extra:
        if (c in frame and c not in out and c not in FALLBACK_BANNED
                and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()):
            out.append(c)
    return out
