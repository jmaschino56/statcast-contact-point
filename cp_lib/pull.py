"""One-shot pull of every regular-season swing for one season into parquet.

The notebook never opens a database connection; this is the only module that
does, and it runs once per season, detached, outside the 7am CT run window.

    .venv/bin/python -m cp_lib.pull --seasons 2025 2024 2026

Why the shape it has:
  * Named cursor, 50,000 rows per fetch, one parquet row group per fetch: the
    whole season never sits in memory on either side.
  * No ORDER BY. savant_data is RANGE partitioned by game_year, so the WHERE
    prunes to one partition and reads it as a plain sequential scan; a sort
    would spill about 850 MB to the production volume first (measured for the
    season export on 2026-09-03).
  * Arrow schema comes from information_schema, never per-chunk inference, so
    an all-null chunk cannot flip a column's type.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
REPO_ROOT = HERE.parents[2]
load_dotenv(REPO_ROOT / ".env")

# Everything Savant counts as a swing. Bunt attempts are kept and flagged
# downstream (join.py decides the swing filter against Savant's n_swings).
SWING_DESCRIPTIONS = [
    "foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
    "foul_pitchout", "swinging_pitchout",
]

COLUMNS = [
    # identifiers
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "batter", "pitcher", "player_name", "stand", "p_throws", "pitch_type",
    # pitch
    "release_speed", "effective_speed", "release_extension",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z", "sz_top", "sz_bot", "vaa", "haa", "pfx_x", "pfx_z",
    # bat tracking
    "bat_speed", "swing_length", "attack_angle", "attack_direction",
    "swing_path_tilt", "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches", "miss_distance",
    # outcome
    "description", "events", "bb_type", "launch_speed", "launch_angle",
    "hc_x", "hc_y", "hit_distance_sc", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "delta_run_exp", "delta_pitcher_run_exp",
    "batted_ball_xrv",
    # context
    "balls", "strikes", "outs_when_up", "inning", "home_team", "away_team",
    "age_bat",
    # Task 14: more per-swing signal for x and z. The pitch's shape and where it
    # crossed, the swing's own classification, and the outcome's finer grain.
    # Nothing here is a Savant aggregate; every one is a per-pitch column.
    "non_magnus_vertical_acceleration", "non_magnus_horizontal_acceleration",
    "magnus_vertical_break", "non_magnus_vertical_break",
    "seam_shifted_wake_x", "seam_shifted_wake_y",
    "vaa_vs_expected", "haa_vs_expected", "expected_vaa",
    "release_spin_rate", "spin_axis", "induced_vb", "hb",
    "zone", "plate_z_normalized", "hit_location", "launch_speed_angle",
    "hyper_speed", "swing_take", "arm_angle",
]

_PG_TO_ARROW = {
    "smallint": pa.int64(), "integer": pa.int64(), "bigint": pa.int64(),
    "double precision": pa.float64(), "real": pa.float64(), "numeric": pa.float64(),
    "text": pa.string(), "character varying": pa.string(), "character": pa.string(),
    "date": pa.date32(), "boolean": pa.bool_(),
    "timestamp without time zone": pa.timestamp("us"),
    "timestamp with time zone": pa.timestamp("us", tz="UTC"),
}


def connect():
    return psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        sslmode="require",
        options="-c statement_timeout=1800000",
        connect_timeout=10,
        keepalives=1, keepalives_idle=60, keepalives_interval=15, keepalives_count=4,
        application_name="contact_point_pull",
    )


def arrow_schema(conn) -> pa.Schema:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'savant_data'"
        )
        types = dict(cur.fetchall())
    missing = [c for c in COLUMNS if c not in types]
    if missing:
        raise SystemExit(f"savant_data lacks columns: {missing}")
    fields = []
    for c in COLUMNS:
        t = _PG_TO_ARROW.get(types[c])
        if t is None:
            raise SystemExit(f"no arrow mapping for {c}: {types[c]}")
        fields.append(pa.field(c, t))
    return pa.schema(fields)


def _coerce(col, typ):
    if pa.types.is_integer(typ):
        return [None if v is None else int(v) for v in col]
    if pa.types.is_floating(typ):
        return [None if v is None else float(v) for v in col]
    return list(col)


def pull_season(season: int, out_dir: Path = DATA, batch: int = 50_000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"swings_{season}.parquet"
    tmp = out.with_suffix(".parquet.partial")
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    conn = connect()
    try:
        schema = arrow_schema(conn)
        sql = (
            f"SELECT {', '.join(COLUMNS)} FROM savant_data "
            "WHERE game_year = %s AND game_type = 'R' "
            "AND (description = ANY(%s) OR bat_speed IS NOT NULL)"
        )
        # named cursors need a transaction, so autocommit stays off here
        cur = conn.cursor(name=f"cp_pull_{season}")
        cur.itersize = batch
        cur.execute(sql, (season, SWING_DESCRIPTIONS))
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        total = 0
        try:
            while True:
                rows = cur.fetchmany(batch)
                if not rows:
                    break
                cols = list(zip(*rows))
                arrays = [pa.array(_coerce(cols[i], f.type), type=f.type)
                          for i, f in enumerate(schema)]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                total += len(rows)
                print(f"  season {season}: {total:,} rows, {time.time() - t0:.0f}s", flush=True)
        finally:
            writer.close()
            cur.close()
        conn.rollback()
    finally:
        conn.close()
    tmp.replace(out)
    rec = {
        "season": season, "rows": total, "columns": COLUMNS, "file": str(out),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "pull_manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"DONE season {season}: {total:,} rows in {rec['seconds']}s -> {out}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--force", action="store_true", help="re-pull even if the parquet exists")
    args = ap.parse_args(argv)
    for season in args.seasons:
        out = args.out_dir / f"swings_{season}.parquet"
        if out.exists() and not args.force:
            print(f"skip season {season}: {out} exists", flush=True)
            continue
        pull_season(season, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
