"""
Validate a rebuilt database against a previous build (e.g. a snapshot), session by session.

Use case: after a ``full_rebuild`` (now with build history/logging), confirm the rebuild
reproduces the old data for the settled session set before promoting it. Compares two builds
— each just a prefix with ``session_table.parquet`` / ``trial_table/`` / ``event_table/`` (local
dir or ``s3://``) — restricted to sessions dated **before** a cutoff (default ``2026-06-01``, so
the last few days before a build, where late-arriving data can legitimately differ, are excluded).

Strategy (counts + checksums, drill down):
  1. Session set: compare the ``_session_id`` sets (only-in-old / only-in-new / shared).
  2. Trial & event tables: for the shared sessions, compute per-session ``COUNT(*)`` and an
     order-independent value checksum (``bit_xor(hash(...))`` over key columns); flag any session
     where the count or checksum differs, and dump a row-level diff for the first few mismatches.
  3. Schema: report any column added/removed between the two builds.

Run::

    python -m aind_dynamic_foraging_database.build.validate.validate_rebuild \\
        --old s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260604 \\
        --new s3://aind-scratch-data/aind-dynamic-foraging-cache \\
        --cutoff-date 2026-06-01 --out-dir /root/capsule/scratch/tmp/validate_rebuild

Or programmatically::

    from aind_dynamic_foraging_database.build.validate.validate_rebuild import validate_builds
    report = validate_builds(old_prefix, new_prefix, cutoff_date="2026-06-01")
"""

import argparse
import os

import duckdb

# Key columns whose values are checksummed per session (must exist in both builds; the available
# subset is used, and anything missing is reported as a schema diff).
TRIAL_KEY_COLS = ["trial", "animal_response", "earned_reward",
                  "reward_probabilityL", "reward_probabilityR"]
EVENT_KEY_COLS = ["trial", "event", "timestamps", "data"]
DEFAULT_CUTOFF = "2026-06-01"
_DRILL_DOWN_SESSIONS = 10  # full row-level diff for at most this many mismatched sessions


def _table_src(prefix, table):
    """A ``read_parquet(...)`` clause for ``table`` (trial_table|event_table) under ``prefix``."""
    return (f"read_parquet('{prefix.rstrip('/')}/{table}/**/*.parquet', "
            f"hive_partitioning=true, union_by_name=true)")


def _columns(con, src):
    """Return the set of column names exposed by a read_parquet source."""
    return set(con.sql(f"DESCRIBE SELECT * FROM {src}").df()["column_name"])


def _digest_expr(key_cols):
    """An order-independent per-group value checksum over ``key_cols``.

    Numeric-castable values are normalized to a canonical DOUBLE string so a benign dtype
    change (e.g. ``trial`` DOUBLE vs BIGINT: ``'0.0'`` vs ``'0'``) doesn't spuriously differ;
    genuine strings (e.g. ``'none'``) fall through to their text form unchanged.
    """
    casts = ", ".join(
        f"COALESCE(CAST(TRY_CAST({c} AS DOUBLE) AS VARCHAR), CAST({c} AS VARCHAR))"
        for c in key_cols
    )
    return f"bit_xor(hash({casts}))"


def _session_digest(con, src, key_cols):
    """Per-session (``session_id``) row count + value checksum over ``key_cols`` for one build."""
    return con.sql(
        f"SELECT session_id, COUNT(*) AS n, {_digest_expr(key_cols)} AS checksum "
        f"FROM {src} WHERE session_id IN (SELECT session_id FROM _shared) "
        f"GROUP BY session_id"
    ).df()


def _compare_digests(old_df, new_df):
    """Merge two per-session digests; return matched/mismatched counts + the mismatch rows.

    A session matches when it is present in both builds with the same row count and checksum.
    Returned ``mismatches`` is a DataFrame with old/new counts + checksums for every differing
    (or one-sided) session, with a ``reason`` column.
    """
    merged = old_df.merge(new_df, on="session_id", how="outer",
                          suffixes=("_old", "_new"), indicator=True)
    same = (merged["_merge"] == "both") & \
           (merged["n_old"] == merged["n_new"]) & \
           (merged["checksum_old"] == merged["checksum_new"])
    mism = merged[~same].copy()

    def _reason(row):
        """Label why a session differs (missing on a side, row count, or values)."""
        if row["_merge"] == "left_only":
            return "missing_in_new"
        if row["_merge"] == "right_only":
            return "missing_in_old"
        if row["n_old"] != row["n_new"]:
            return "row_count"
        return "checksum"

    if len(mism):
        mism["reason"] = mism.apply(_reason, axis=1)
    return {"n_shared": int((merged["_merge"] == "both").sum()),
            "n_match": int(same.sum()), "n_mismatch": int(len(mism)),
            "mismatches": mism.drop(columns="_merge")}


def _shared_sessions(con, old_prefix, new_prefix, cutoff_date):
    """Sessions present in BOTH builds' session tables with ``session_date < cutoff_date``.

    Returns ``(shared_df, only_old, only_new)`` where the latter two are the one-sided id lists.
    """
    def _ids(prefix):
        """Pre-cutoff ``_session_id`` set from a build's session table."""
        path = f"{prefix.rstrip('/')}/session_table.parquet"
        return set(con.sql(
            f"SELECT _session_id FROM read_parquet('{path}') "
            f"WHERE session_date < '{cutoff_date}'"
        ).df()["_session_id"])

    old_ids, new_ids = _ids(old_prefix), _ids(new_prefix)
    shared = old_ids & new_ids
    import pandas as pd
    return (pd.DataFrame({"session_id": sorted(shared)}),
            sorted(old_ids - new_ids), sorted(new_ids - old_ids))


def validate_builds(old_prefix, new_prefix, cutoff_date=DEFAULT_CUTOFF,
                    out_dir=None, con=None, verbose=True):
    """Compare two builds for sessions dated before ``cutoff_date``; return a report dict.

    Parameters
    ----------
    old_prefix, new_prefix : str
        Build prefixes (local dir or ``s3://``) each holding ``session_table.parquet`` /
        ``trial_table/`` / ``event_table/``. ``old`` is the reference (e.g. a snapshot).
    cutoff_date : str
        ``YYYY-MM-DD``; only sessions with ``session_date < cutoff_date`` are compared.
    out_dir : str, optional
        If given, write the mismatch CSVs + row-level diffs here.
    con : duckdb connection, optional
        Connection to run on (default: the module connection).
    verbose : bool

    Returns
    -------
    dict
        ``{cutoff_date, only_in_old, only_in_new, trial, event, schema}`` where ``trial``/``event``
        each hold the matched/mismatched counts (see :func:`_compare_digests`).
    """
    con = con or duckdb
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    shared, only_old, only_new = _shared_sessions(con, old_prefix, new_prefix, cutoff_date)
    con.register("_shared", shared)
    if verbose:
        print(f"Sessions before {cutoff_date}: shared={len(shared)}, "
              f"only_in_old={len(only_old)}, only_in_new={len(only_new)}")

    report = {"cutoff_date": cutoff_date, "n_shared": len(shared),
              "only_in_old": only_old, "only_in_new": only_new, "schema": {}}
    try:
        for table, keys in (("trial_table", TRIAL_KEY_COLS), ("event_table", EVENT_KEY_COLS)):
            name = table.split("_")[0]  # 'trial' | 'event'
            old_src, new_src = _table_src(old_prefix, table), _table_src(new_prefix, table)
            old_cols, new_cols = _columns(con, old_src), _columns(con, new_src)
            report["schema"][name] = {"only_in_old": sorted(old_cols - new_cols),
                                      "only_in_new": sorted(new_cols - old_cols)}
            use_keys = [c for c in keys if c in old_cols and c in new_cols]
            res = _compare_digests(_session_digest(con, old_src, use_keys),
                                   _session_digest(con, new_src, use_keys))
            report[name] = {k: res[k] for k in ("n_shared", "n_match", "n_mismatch")}
            report[name]["key_cols"] = use_keys
            if verbose:
                print(f"  {name}: matched={res['n_match']}, mismatched={res['n_mismatch']} "
                      f"(keys={use_keys})")
            if out_dir and res["n_mismatch"]:
                res["mismatches"].to_csv(f"{out_dir}/{name}_mismatches.csv", index=False)
                _dump_row_diffs(con, old_src, new_src, res["mismatches"], use_keys,
                                f"{out_dir}/{name}_row_diffs.csv")
    finally:
        con.unregister("_shared")
    return report


def _dump_row_diffs(con, old_src, new_src, mismatches, key_cols, out_csv):
    """Write a row-level diff (full outer join on session_id+trial) for the first few mismatches."""
    sids = [s for s in mismatches["session_id"].tolist()[:_DRILL_DOWN_SESSIONS] if s is not None]
    if not sids:
        return
    in_list = ", ".join("'" + str(s).replace("'", "''") + "'" for s in sids)
    cols = ", ".join(key_cols)
    con.sql(f"""
        SELECT COALESCE(o.session_id, n.session_id) AS session_id,
               COALESCE(o.trial, n.trial) AS trial,
               {_side_cols('o', key_cols)}, {_side_cols('n', key_cols)},
               CASE WHEN o.session_id IS NULL THEN 'only_new'
                    WHEN n.session_id IS NULL THEN 'only_old' ELSE 'differs' END AS side
        FROM (SELECT session_id, {cols} FROM {old_src} WHERE session_id IN ({in_list})) o
        FULL OUTER JOIN
             (SELECT session_id, {cols} FROM {new_src} WHERE session_id IN ({in_list})) n
        ON o.session_id = n.session_id AND o.trial = n.trial
        WHERE o.trial IS NULL OR n.trial IS NULL
           OR {_any_diff(key_cols)}
        ORDER BY session_id, trial
    """).df().to_csv(out_csv, index=False)


def _side_cols(alias, key_cols):
    """Render ``alias.col AS col_<alias>`` for each key column (one build's side of a diff)."""
    return ", ".join(f"{alias}.{c} AS {c}_{alias}" for c in key_cols if c != "trial")


def _any_diff(key_cols):
    """A predicate true when any non-key column differs between the two sides (NULL-safe)."""
    cols = [c for c in key_cols if c != "trial"]
    return " OR ".join(f"o.{c} IS DISTINCT FROM n.{c}" for c in cols) or "FALSE"


def _cli(argv=None):
    """Parse argv and run :func:`validate_builds`."""
    p = argparse.ArgumentParser(description="Validate a rebuilt database against a prior build.")
    p.add_argument("--old", required=True, help="reference build prefix (local dir or s3://)")
    p.add_argument("--new", required=True, help="rebuilt build prefix (local dir or s3://)")
    p.add_argument("--cutoff-date", default=DEFAULT_CUTOFF,
                   help=f"compare sessions with session_date < this (default: {DEFAULT_CUTOFF})")
    p.add_argument("--out-dir", default=None, help="write mismatch CSVs + row diffs here")
    args = p.parse_args(argv)
    report = validate_builds(args.old, args.new, cutoff_date=args.cutoff_date, out_dir=args.out_dir)
    ok = report.get("trial", {}).get("n_mismatch", 1) == 0 and \
        report.get("event", {}).get("n_mismatch", 1) == 0 and not report["only_in_old"]
    print("\nRESULT:", "MATCH" if ok else "DIFFERENCES FOUND")
    return report


if __name__ == "__main__":
    _cli()
