"""Swing loading, the swing filter that matches Savant's n_swings, and the tail join."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import DATA
from .savant_truth import load_leaderboard  # re-exported so tests can monkeypatch it

# Savant's bat_contact_code 9 (whiff) includes foul tips. Measured on 2025: our
# whiff rate is 0.2346 without them and 0.2565 with them, against Savant's own
# published 0.2569. A foul tip is a ball that grazes the bat and is caught, and
# Savant's bat tracking scores it as no meaningful contact.
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "swinging_pitchout",
         "foul_tip", "bunt_foul_tip"}
RULES = ("bat_speed", "bat_speed_nobunt", "bat_speed_nobunt_typical")
TYPICAL_MIN_BAT_SPEED = 40.0
JOIN_KEYS = ["game_pk", "batter", "pitcher", "miss_key"]


def derive_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = df.copy()
    d = df["description"].fillna("")
    df["contact_type"] = np.where(
        d == "hit_into_play", "in_play", np.where(d.isin(WHIFF), "whiff", "foul"))
    df["is_bunt"] = d.str.contains("bunt")
    df["season"] = season
    return df


def load_swings(season: int) -> pd.DataFrame:
    return derive_columns(pd.read_parquet(DATA / f"swings_{season}.parquet"), season)


def swing_filter(df: pd.DataFrame, rule: str) -> pd.Series:
    if rule not in RULES:
        raise ValueError(rule)
    m = df["bat_speed"].notna()
    if rule.endswith("nobunt") or rule.endswith("typical"):
        m &= ~df["is_bunt"]
    if rule.endswith("typical"):
        m &= df["bat_speed"] >= TYPICAL_MIN_BAT_SPEED
    return m


def denominator_report(df: pd.DataFrame, season: int, type_: str) -> pd.DataFrame:
    key = "pitcher" if type_ == "pitcher" else "batter"
    sav = load_leaderboard(season, type_)[["id", "n_swings"]].dropna(subset=["id"]).astype({"id": int})
    rows = []
    for rule in RULES:
        ours = df[swing_filter(df, rule)].groupby(key).size().rename("ours").reset_index()
        m = sav.merge(ours, left_on="id", right_on=key, how="left").fillna({"ours": 0})
        rel = (m["ours"] - m["n_swings"]).abs() / m["n_swings"].clip(lower=1)
        rows.append({"rule": rule, "type": type_, "players": len(m),
                     "median_abs_rel_err": float(rel.median()),
                     "p90_abs_rel_err": float(rel.quantile(0.9)),
                     "total_ours": int(m["ours"].sum()),
                     "total_savant": int(m["n_swings"].sum())})
    return pd.DataFrame(rows)


def residual_report(df: pd.DataFrame, season: int, type_: str, rule: str) -> dict:
    """The per-player mismatch left over after a rule is chosen.

    denominator_report scores the rules against each other; this says how far off
    the winner still is, player by player, which is what spec section 3 asks to be
    printed. Differences are absolute, so p05 and p95 bound the size of the miss
    rather than its direction.
    """
    key = "pitcher" if type_ == "pitcher" else "batter"
    sav = load_leaderboard(season, type_)[["id", "n_swings"]].dropna(subset=["id"]).astype({"id": int})
    ours = df[swing_filter(df, rule)].groupby(key).size().rename("ours").reset_index()
    m = sav.merge(ours, left_on="id", right_on=key, how="left").fillna({"ours": 0})
    d = (m["ours"] - m["n_swings"]).abs()
    return {"rule": rule, "type": type_, "players": int(len(m)),
            "total_ours": int(m["ours"].sum()), "total_savant": int(m["n_swings"].sum()),
            "median_diff": float(d.median()), "p05": float(d.quantile(0.05)),
            "p95": float(d.quantile(0.95)), "exact_match": float((d == 0).mean())}


def choose_swing_filter(df: pd.DataFrame, season: int) -> str:
    rep = pd.concat([denominator_report(df, season, t) for t in ("pitcher", "batter")])
    return str(rep.groupby("rule")["median_abs_rel_err"].mean().idxmin())


def join_tails(df: pd.DataFrame, tails: pd.DataFrame) -> pd.DataFrame:
    """Tail rows left-joined to our swing rows.

    Exact on (game_pk, batter, pitcher, round(miss_distance, 6)); tails that miss
    fall back to (game_pk, batter, pitcher, pitch_type) with the nearest miss
    distance within 1e-3. Real tails carry game_date, season and pitch_type, which
    our swing rows also carry, so the swing-side copies of any shared non-key column
    are dropped before merging rather than being suffixed. Two swing rows can share
    a join key (same game, same matchup, the same rounded miss distance), so the
    exact merge is deduplicated to one row per play_id and the fan-out is counted.
    """
    left = tails.copy()
    left["miss_key"] = left["miss_distance_inches"].round(6)
    right = df.copy()
    right["miss_key"] = right["miss_distance"].round(6)
    shared = [c for c in right.columns if c in left.columns and c not in JOIN_KEYS]
    right = right.drop(columns=shared)

    exact = left.merge(right, on=JOIN_KEYS, how="inner")
    fanout = int(len(exact) - exact["play_id"].nunique())
    exact = exact.drop_duplicates(subset=["play_id"], keep="first")
    exact["join_method"] = "exact"

    todo = left[~left["play_id"].isin(exact["play_id"])]
    near_keys = ["game_pk", "batter", "pitcher", "pitch_type"]
    right_near = df.drop(columns=[c for c in shared if c not in near_keys], errors="ignore")
    cand = todo.drop(columns=["miss_key"]).merge(right_near, on=near_keys, how="inner")
    if len(cand):
        cand = cand[(cand["miss_distance"] - cand["miss_distance_inches"]).abs() <= 1e-3]
    if len(cand):
        cand = cand.assign(_d=(cand["miss_distance"] - cand["miss_distance_inches"]).abs())
        cand = cand.sort_values("_d").drop_duplicates(subset=["play_id"], keep="first").drop(columns=["_d"])
        cand["join_method"] = "nearest"
    matched = set(exact["play_id"]) | set(cand["play_id"] if len(cand) else [])
    missed = todo[~todo["play_id"].isin(matched)].drop(columns=["miss_key"]).copy()
    missed["join_method"] = "unmatched"

    parts = [p for p in (exact.drop(columns=["miss_key"], errors="ignore"), cand, missed) if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else exact.drop(columns=["miss_key"], errors="ignore")
    out.attrs["join_stats"] = {
        "tails": int(len(left)),
        "exact": int((out["join_method"] == "exact").sum()),
        "nearest": int((out["join_method"] == "nearest").sum()),
        "unmatched": int((out["join_method"] == "unmatched").sum()),
        "fanout_rows_dropped": fanout,
        "join_rate": float((out["join_method"] != "unmatched").mean()) if len(out) else 0.0,
    }
    return out
