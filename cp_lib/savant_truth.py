"""Baseball Savant swing-timing ground truth: fetch once, cache by URL, load tidy.

Four sources, all public but undocumented beyond the leaderboard itself:

  1. Leaderboard CSV (per player-season rates and bucket means), plain and split
     by contact type (bat_contact_code: 2 = in play, 4 = foul, 9 = whiff).
  2. The leaderboard page HTML, which embeds TIMING_BINNED_DATA_LEAGUE (league
     histograms with avg run value per bin) and
     TIMING_leagueAverageDataFilteredGrouped (league category counts).
  3. Per-player histograms over ALL swings:
     /leaderboard/bat-tracking/swing-timing-distribution/player/{id}
  4. Per-player whiff tails (the 10 worst per list, four lists):
     /leaderboard/services/swing-timing?id={id}

`season` must be sent as the array form `season[]=YYYY`; the scalar form is
ignored and returns the current season.

    .venv/bin/python -m cp_lib.savant_truth fetch --seasons 2025 2024 2026
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = DATA / "savant"
MANIFEST = CACHE / "manifest.jsonl"
BASE = "https://baseballsavant.mlb.com"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128 Safari/537.36")
THROTTLE_S = 0.5
TYPES = ("pitcher", "batter")
CONTACT_CODES = {2: "in_play", 4: "foul", 9: "whiff"}

_session = requests.Session()
_session.headers["User-Agent"] = UA


# ----------------------------------------------------------------------------
# URLs
# ----------------------------------------------------------------------------
def leaderboard_url(season: int, type_: str, split: str | None = None,
                    csv_: bool = True, min_: str = "1") -> str:
    u = (f"{BASE}/leaderboard/bat-tracking/swing-timing-miss-distance"
         f"?season%5B%5D={season}&type={type_}&min={min_}")
    if split:
        u += f"&split%5B%5D={split}"
    if csv_:
        u += "&csv=true"
    return u


def histogram_url(pid: int, season: int, type_: str) -> str:
    return (f"{BASE}/leaderboard/bat-tracking/swing-timing-distribution/player/{pid}"
            f"?season%5B%5D={season}&type={type_}")


def tails_url(pid: int, season: int, type_: str) -> str:
    return f"{BASE}/leaderboard/services/swing-timing?id={pid}&season%5B%5D={season}&type={type_}"


# ----------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------
def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def _path(url: str, kind: str) -> Path:
    return CACHE / f"{_key(url)}.{kind}"


def _valid(body: bytes, kind: str) -> str | None:
    """Returns None when the body is usable, else a reason."""
    if kind == "json":
        try:
            j = json.loads(body)
        except ValueError:
            return "not json"
        if isinstance(j, dict) and "error" in j:
            return "error body: " + str(j["error"])[:120]
        return None
    if kind == "csv":
        head = body[:16].lstrip(b"\xef\xbb\xbf")
        return None if head.startswith(b'"id"') else "not a leaderboard csv"
    if kind == "html":
        return None if b"TIMING_BINNED_DATA_LEAGUE" in body else "no TIMING block"
    if kind == "swingpath_html":
        # a different leaderboard, with its own embedded var and no TIMING block
        return None if re.search(rb"\bdata\s*=\s*\[\{", body) else "no swing path data var"
    return None


def cached_get(url: str, kind: str, attempts: int = 3) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = _path(url, kind)
    if p.exists() and p.stat().st_size > 0:
        return p
    last = None
    for i in range(attempts):
        t0 = time.time()
        try:
            r = _session.get(url, timeout=60)
            status, body = r.status_code, r.content
        except requests.RequestException as e:  # noqa: BLE001
            status, body, last = 0, b"", str(e)
        reason = _valid(body, kind) if status == 200 else f"http {status}"
        if reason is None:
            p.write_bytes(body)
            with open(MANIFEST, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "url": url, "file": p.name, "kind": kind, "status": status,
                    "bytes": len(body), "seconds": round(time.time() - t0, 2),
                    "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                }) + "\n")
            time.sleep(THROTTLE_S)
            return p
        last = reason
        time.sleep(THROTTLE_S * (2 ** (i + 1)))
    raise RuntimeError(f"gave up on {url}: {last}")


# ----------------------------------------------------------------------------
# fetch everything
# ----------------------------------------------------------------------------
def fetch_season(season: int, log=print) -> dict:
    counts = {"season": season, "csv": 0, "html": 0, "hist": 0, "tails": 0, "failed": []}
    ids = {}
    for t in ("pitcher", "batter", "league"):
        for split in (None, "bat_contact_code"):
            p = cached_get(leaderboard_url(season, t, split), "csv")
            counts["csv"] += 1
            if t in TYPES and split is None:
                df = pd.read_csv(p, encoding="utf-8-sig")
                ids[t] = sorted(int(x) for x in df["id"].dropna().unique())
                log(f"season {season} {t}: {len(ids[t])} ids")
    for split in (None, "bat_contact_code"):
        cached_get(leaderboard_url(season, "league", split, csv_=False), "html")
        counts["html"] += 1
    for t in TYPES:
        for n, pid in enumerate(ids[t], 1):
            for kind, fn in (("hist", histogram_url), ("tails", tails_url)):
                try:
                    cached_get(fn(pid, season, t), "json")
                    counts[kind] += 1
                except RuntimeError as e:
                    counts["failed"].append({"id": pid, "type": t, "kind": kind, "err": str(e)[:200]})
                    log(f"  FAIL {t} {pid} {kind}: {e}")
            if n % 50 == 0:
                log(f"season {season} {t}: {n}/{len(ids[t])} players")
    counts["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (CACHE / f"done_{season}.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    log(f"DONE season {season}: {counts}")
    return counts


# ----------------------------------------------------------------------------
# loaders (read only from cache; never fetch)
# ----------------------------------------------------------------------------
def _read_cached(url: str, kind: str) -> Path:
    p = _path(url, kind)
    if not p.exists():
        raise FileNotFoundError(f"not cached: {url}")
    return p


def load_leaderboard(season: int, type_: str, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(_read_cached(leaderboard_url(season, type_, split), "csv"),
                     encoding="utf-8-sig")
    df["season"] = season
    df["type"] = type_
    if split == "bat_contact_code":
        df["contact_type"] = df["bat_contact_code"].map(CONTACT_CODES)
    return df


def _page_var(html: str, name: str):
    m = re.search(r"%s\s*=\s*(\[.*?\]|\{.*?\});" % re.escape(name), html, re.S)
    return json.loads(m.group(1)) if m else None


BIN_WIDTH = {"x": 2.0, "y": 2.0, "z": 1.0}


def _lower_edge(axis, label):
    """Savant labels a histogram bin by its CENTRE. Everything downstream of these
    loaders works in lower edges, so the conversion happens once, here.

    This is Savant agreeing with Savant, not a fit. Read as centres, the pooled
    2025 pitcher histogram's three band masses ARE that leaderboard's three
    category rates: the gap is 0.0000 on y, 0.0032 on z and 0.0165 on x, the last
    carrying the known 0.15 percent count gap between that histogram and n_swings.
    Read as lower edges the same three gaps are 0.0317, 0.0864 and 0.0734. One
    offset, three axes, no free parameter.

    Read the wrong way every binned artifact is half a bin out: the per-type
    target's shape, the per-player Wasserstein distance, the league bin figure and
    the heatmap grid, and through the quantile map's grid every calibrated value.
    """
    w = np.array([BIN_WIDTH[a] for a in np.asarray(axis)], float)
    return np.asarray(label, float) - w / 2.0


def bin_of(axis: str, values) -> np.ndarray:
    """The lower edge of the Savant bin a value falls in. NaN survives.

    This has to agree with _lower_edge or every merge on bin quietly finds nothing
    in common. Savant's labels are multiples of the width, so its lower edges are
    offset half a width from them: x and y sit on the odd numbers, z on the halves.
    Flooring to a plain multiple of the width, which is what this used to do, puts
    our x grid on the even numbers and Savant's on the odd ones, with no overlap.
    """
    w = BIN_WIDTH[axis]
    return np.floor((np.asarray(values, float) + w / 2.0) / w) * w - w / 2.0


def load_league_bins(season: int, split: str | None = None) -> pd.DataFrame:
    html = _read_cached(leaderboard_url(season, "league", split, csv_=False), "html").read_text(encoding="utf-8")
    d = _page_var(html, "TIMING_BINNED_DATA_LEAGUE")
    rows = []
    for axis, lst in d.items():
        for r in lst:
            rows.append({"axis": axis, **r})
    df = pd.DataFrame(rows)
    df["season"] = season
    df["bin"] = _lower_edge(df["axis"], df["bin"])
    return df


def load_league_grouped(season: int, split: str | None = None) -> pd.DataFrame:
    html = _read_cached(leaderboard_url(season, "league", split, csv_=False), "html").read_text(encoding="utf-8")
    df = pd.DataFrame(_page_var(html, "TIMING_leagueAverageDataFilteredGrouped"))
    df["season"] = season
    return df


def _ids(season: int, type_: str) -> list[int]:
    return sorted(int(x) for x in load_leaderboard(season, type_)["id"].dropna().unique())


def load_histograms(season: int, type_: str) -> pd.DataFrame:
    rows = []
    for pid in _ids(season, type_):
        p = _path(histogram_url(pid, season, type_), "json")
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for axis in ("x", "y", "z"):
            for r in d.get(axis, []):
                rows.append({"id": pid, "type": type_, "axis": axis, "bin": r["bin"],
                             "n": r["n"], "avg_run_exp": r["avg_run_exp"], "season": season})
    df = pd.DataFrame(rows)
    if len(df):
        df["bin"] = _lower_edge(df["axis"], df["bin"])
    return df


TAIL_COLS = ["play_id", "game_pk", "game_date", "pitcher", "savant_batter_id", "pitch_type",
             "is_swing_sword", "miss_distance_inches", "delta_batball_tiedup_neg_x2",
             "delta_batball_late_neg_y2_msec", "ball_pos_above_plane",
             "timing_x_tiedupflail", "timing_y_earlylate", "timing_z_overunder"]


def load_tails(season: int) -> pd.DataFrame:
    """One row per unique play_id across every list, type and player.

    `lists` records which (type, list) selections carried the row, so the
    selection bias is visible downstream.
    """
    seen: dict[str, dict] = {}
    for t in TYPES:
        for pid in _ids(season, t):
            p = _path(tails_url(pid, season, t), "json")
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            for lst, rows in d.items():
                for r in rows:
                    rec = seen.setdefault(r["play_id"], {k: r.get(k) for k in TAIL_COLS})
                    rec.setdefault("lists", set()).add(f"{t}:{lst}")
    df = pd.DataFrame(list(seen.values()))
    if df.empty:
        return df
    df["lists"] = df["lists"].map(lambda s: ",".join(sorted(s)))
    df["miss_distance_inches"] = pd.to_numeric(df["miss_distance_inches"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["season"] = season
    return df.rename(columns={"savant_batter_id": "batter",
                              "delta_batball_tiedup_neg_x2": "sav_x",
                              "delta_batball_late_neg_y2_msec": "sav_y",
                              "ball_pos_above_plane": "sav_z"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--seasons", nargs="+", type=int, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        for s in args.seasons:
            if (CACHE / f"done_{s}.json").exists():
                print(f"skip season {s}: done marker exists", flush=True)
                continue
            fetch_season(s, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
