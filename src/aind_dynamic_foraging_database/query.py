"""
DuckDB query helpers for the foraging parquet database.

Two layers — reach for the simple helpers first, drop to native SQL when you need more:

  Layer 1 (convenience — the common "return loop"):
      select_sessions -> fetch_trials / fetch_events
    Filter the (small) session table on any metric / metadata, then pull those sessions'
    trials or events with the session metadata already joined on — in one call.

  Layer 0 (escape hatch — covers ANY query):
      read_trials / read_events
    Return a fast, partition-scoped ``read_parquet(...)`` clause for a set of subjects.
    Drop it into whatever SQL you write — aggregations, window functions, trial<->event
    joins, custom GROUP BY. You keep the full power of SQL; the helper only does the part
    that is easy to get wrong or slow (reading the right partition files, fast + correct).

Why scoped reads are fast: a full ``trial_table/**/*.parquet`` glob with ``union_by_name``
must read *every* subject file's footer to build the column union before it can prune
(~25 s cold). Scoping the read to just the subjects you asked for reads only their footers
(~1 s), while still unioning their columns correctly.

Everything reads the public S3 database (no AWS credentials needed). To query a local build,
pass ``base=`` (or reassign ``SESSION_DB`` / ``TRIAL_DB`` / ``EVENT_DB``).
"""

import duckdb

PROD_S3_PREFIX = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
SESSION_DB = f"{PROD_S3_PREFIX}/session_table.parquet"  # flat session table
TRIAL_DB = f"{PROD_S3_PREFIX}/trial_table"  # Hive-partitioned by subject_id
EVENT_DB = f"{PROD_S3_PREFIX}/event_table"  # Hive-partitioned by subject_id

# SELECT * over the trial table is ~21 GB — always project. These small defaults cover
# the usual choice/reward analysis; pass columns=[...] for others, or columns="*" for all.
DEFAULT_TRIAL_COLUMNS = [
    "trial", "animal_response", "earned_reward",
    "reward_probabilityL", "reward_probabilityR",
]
DEFAULT_EVENT_COLUMNS = ["trial", "timestamps", "event", "data"]

# Leading identity columns we always emit and never duplicate from the trial/event side.
_KEYS = ("subject_id", "session_date", "session_id")

# Per-table suffixes under a prefix (live prefix or a snapshot dir share this layout).
_TABLE_SUFFIX = {
    "session": "session_table.parquet",
    "trial": "trial_table",
    "event": "event_table",
}
# Module default per table, used when reading the latest (live) database.
_DEFAULT_BASE = {"session": SESSION_DB, "trial": TRIAL_DB, "event": EVENT_DB}


# ---------------------------------------------------------------------------
# Snapshots — pin reads to a frozen copy of the database (see create_snapshot)
# ---------------------------------------------------------------------------

# Globally selected snapshot ("20260604") or None for the latest (live) database.
_SNAPSHOT = None
# Sentinel for the per-call ``snapshot=`` kwarg, so an explicit ``snapshot=None``
# (force latest) is distinguishable from "not passed" (fall back to the global).
_UNSET = object()


def use_snapshot(date):
    """Globally pin all subsequent reads to a snapshot, or reset to the latest database.

    Snapshots are immutable, dated copies of the whole database (see
    :func:`aind_dynamic_foraging_database.build.snapshot.create_snapshot`). Pinning a snapshot
    fixes the data source for an analysis or training run even as the live database keeps growing::

        import aind_dynamic_foraging_database as db
        db.use_snapshot("20260604")     # every read now hits snapshots/20260604/
        ...
        db.use_snapshot(None)           # back to the latest (live) database

    Per-call ``snapshot=`` on the query helpers overrides this for a single call. Setting the
    global drops the partition-listing cache so a stale listing from a different prefix can't leak.

    Parameters
    ----------
    date : str or None
        Snapshot id, ``YYYYMMDD`` (e.g. ``"20260604"``), or ``None`` to read the latest database.
    """
    global _SNAPSHOT
    if date != _SNAPSHOT:
        clear_caches()
    _SNAPSHOT = date


def current_snapshot():
    """Return the globally pinned snapshot id, or ``None`` if reading the latest database."""
    return _SNAPSHOT


def _resolve_base(base, snapshot, kind):
    """Resolve the read path for table ``kind`` ('session' | 'trial' | 'event').

    Precedence (explicit always wins): an explicit ``base=`` > the effective snapshot > the latest
    default. The effective snapshot is the per-call ``snapshot`` if passed (``is not _UNSET``) —
    including an explicit ``None`` that forces latest — otherwise the global :func:`use_snapshot`.
    """
    if base is not None:
        return base
    snap = _SNAPSHOT if snapshot is _UNSET else snapshot
    if snap is None:
        return _DEFAULT_BASE[kind]
    return f"{PROD_S3_PREFIX}/snapshots/{snap}/{_TABLE_SUFFIX[kind]}"


# Snapshot-aware path accessors — the snapshot-respecting counterparts of the static
# SESSION_DB / TRIAL_DB / EVENT_DB constants, for writing raw DuckDB SQL by hand. The
# constants always point at the latest database; these honour use_snapshot / snapshot=.


def session_db(snapshot=_UNSET):
    """Session-table path (``session_table.parquet``), honouring the selected snapshot.

    The snapshot-aware counterpart of :data:`SESSION_DB`, for raw SQL::

        duckdb.sql(f"SELECT * FROM read_parquet('{session_db()}') WHERE foraging_eff > 0.8")

    Returns the latest path by default, or ``snapshots/<date>/session_table.parquet`` when a
    snapshot is pinned via :func:`use_snapshot`. Pass ``snapshot="20260604"`` to target one
    directly, or ``snapshot=None`` to force latest regardless of the global.
    """
    return _resolve_base(None, snapshot, "session")


def trial_db(snapshot=_UNSET):
    """Trial-table directory prefix, honouring the selected snapshot.

    The snapshot-aware counterpart of :data:`TRIAL_DB` (see :func:`session_db`). Use it as the
    base of a partitioned read::

        duckdb.sql(f"SELECT * FROM read_parquet('{trial_db()}/**/*.parquet', "
                   f"hive_partitioning=true, union_by_name=true)")
    """
    return _resolve_base(None, snapshot, "trial")


def event_db(snapshot=_UNSET):
    """Event-table directory prefix, honouring the selected snapshot.

    The snapshot-aware counterpart of :data:`EVENT_DB` (see :func:`session_db` / :func:`trial_db`).
    """
    return _resolve_base(None, snapshot, "event")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _conn(con):
    """Return the DuckDB connection to use (the given one, or the default module conn)."""
    return con if con is not None else duckdb


def _quote_in(values):
    """Render an iterable as a SQL IN-list of quoted, escaped string literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


# Per-base cache of the partition subject set. The listing is one S3 LIST (~0.4 s) and the
# cache is otherwise re-run on *every* fetch; the set of partitions is static within a session.
_PARTITION_CACHE = {}


def _partition_subjects(base, con=None):
    """Subject ids that actually have a partition file under ``base`` (memoized per base).

    One cheap S3 LIST via ``glob()`` (not a footer scan) — used to drop requested
    subjects with no files, since a scoped ``read_parquet`` list errors on a path that
    matches nothing. Cached per ``base`` (call :func:`clear_caches` after rebuilding a
    local cache in the same session).
    """
    cached = _PARTITION_CACHE.get(base)
    if cached is not None:
        return cached
    rows = _conn(con).sql(f"SELECT file FROM glob('{base}/subject_id=*/*.parquet')").df()
    found = rows["file"].str.extract(r"subject_id=([^/]+)/")[0].dropna()
    _PARTITION_CACHE[base] = result = set(found)
    return result


def clear_caches():
    """Drop the memoized partition listings (call after rebuilding a cache in-session)."""
    _PARTITION_CACHE.clear()


def _full_glob(base):
    """The correct-but-slow read over every subject (reads all footers for the union)."""
    return f"read_parquet('{base}/**/*.parquet', hive_partitioning=true, union_by_name=true)"


# Above this many subjects, an explicit per-subject file list (one S3 listing per path + a slow
# read_parquet([...]) union) costs more than the full glob with a partition-pruning WHERE.
_SCOPED_MAX = 100


def _scoped_read(base, subjects, con):
    """Build a ``read_parquet(...)`` clause scoped to ``subjects`` (or the full glob).

    Few subjects -> an explicit per-subject file list (skips the full-footer scan, ~1 s). Many
    subjects (> ``_SCOPED_MAX``) -> the full glob with ``WHERE subject_id IN (...)``, which DuckDB
    prunes to just those partitions and is far faster than a long ``read_parquet([...])`` list.
    """
    if subjects is None:
        return _full_glob(base)
    want = sorted({str(s) for s in subjects} & _partition_subjects(base, con))
    if not want:
        # No requested subject has data: yield zero rows but the correct full schema.
        return f"(SELECT * FROM {_full_glob(base)} WHERE false)"
    if len(want) > _SCOPED_MAX:
        return (f"(SELECT * FROM {_full_glob(base)} "
                f"WHERE CAST(subject_id AS VARCHAR) IN ({_quote_in(want)}))")
    files = [f"'{base}/subject_id={s}/*.parquet'" for s in want]
    return f"read_parquet([{', '.join(files)}], hive_partitioning=true, union_by_name=true)"


# ---------------------------------------------------------------------------
# Layer 0 — escape hatch: a fast, partition-scoped read_parquet(...) source
# ---------------------------------------------------------------------------


def read_trials(subjects=None, base=None, con=None, snapshot=_UNSET):
    """Return a ``read_parquet(...)`` clause for the trial table, scoped to ``subjects``.

    Drop the returned string into any SQL you write::

        src = read_trials(['754372', '758435'])
        duckdb.sql(f"SELECT subject_id, AVG(earned_reward::DOUBLE) FROM {src} GROUP BY subject_id")

    Scoping to the subjects you need reads only their partition files (~1 s) instead of
    every subject's footer. ``subjects=None`` falls back to the full (slow) glob over all
    subjects. Note a scoped read exposes only the columns present in *those* subjects'
    files; selecting a column none of them has will raise.

    Parameters
    ----------
    subjects : iterable, optional
        Subject ids to scope the read to. ``None`` reads the full table (slow glob).
    base : str, optional
        Trial-table location — the partitioned-table **directory** prefix (default: the
        production S3 ``trial_table``). Pass a local dir / other S3 prefix for another build.
    con : duckdb connection, optional
        DuckDB connection to run the partition listing on (default: the module connection).
        Pass your own for warm reuse, or custom settings (S3 region/creds, threads, memory).
    snapshot : str, optional
        Read from a snapshot (``"20260604"``) instead of the latest database, overriding any
        global :func:`use_snapshot` for this call. Pass ``None`` to force latest. Ignored if
        ``base`` is given. See :func:`use_snapshot`.
    """
    return _scoped_read(_resolve_base(base, snapshot, "trial"), subjects, con)


def read_events(subjects=None, base=None, con=None, snapshot=_UNSET):
    """Return a ``read_parquet(...)`` clause for the event table, scoped to ``subjects``.

    The event-table counterpart of :func:`read_trials` — same ``subjects`` / ``base`` / ``con`` /
    ``snapshot`` behaviour, except ``base`` defaults to the production S3 ``event_table`` directory
    prefix.
    """
    return _scoped_read(_resolve_base(base, snapshot, "event"), subjects, con)


# ---------------------------------------------------------------------------
# Layer 1 — convenience: filter sessions, then fetch their trials / events
# ---------------------------------------------------------------------------


def select_sessions(where=None, subjects=None, columns=None, base=None, con=None,
                    order_by="subject_id, session_date", snapshot=_UNSET):
    """Filter the (small) session table; return the selected sessions as a DataFrame.

    The first step of both common workflows — filter on session metrics/metadata, or on
    subject first, or both — then hand the result to :func:`fetch_trials` /
    :func:`fetch_events`.

    Parameters
    ----------
    where : str, optional
        Raw SQL predicate on the session table, e.g.
        ``"task LIKE '%Uncoupled%' AND foraging_eff > 0.8"``.
    subjects : iterable, optional
        Restrict to these subject ids (adds ``subject_id IN (...)``).
    columns : list[str], optional
        Extra session-metadata columns to carry along (and onto trials/events later).
        ``_session_id, subject_id, session_date`` are always included as leading columns.
    base : str, optional
        Session table to read — the ``session_table.parquet`` **file** (default: the
        production S3 database). Pass a local file / other S3 path to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).
    order_by : str, optional
        SQL ORDER BY clause (default: ``"subject_id, session_date"``); pass ``None`` for none.
    snapshot : str, optional
        Read from a snapshot (``"20260604"``) instead of the latest database, overriding any
        global :func:`use_snapshot` for this call. Pass ``None`` to force latest. Ignored if
        ``base`` is given. See :func:`use_snapshot`.

    Returns
    -------
    pandas.DataFrame
        One row per selected session, with ``_session_id`` as the join key.
    """
    base = _resolve_base(base, snapshot, "session")
    extra = [c for c in (columns or []) if c not in ("_session_id", *_KEYS)]
    sel_cols = ", ".join(["_session_id", "subject_id", "session_date", *extra])
    clauses = []
    if subjects is not None:
        clauses.append(f"subject_id IN ({_quote_in(subjects)})")
    if where:
        clauses.append(f"({where})")
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_sql = f"ORDER BY {order_by}" if order_by else ""
    return _conn(con).sql(
        f"SELECT {sel_cols} FROM read_parquet('{base}') {where_sql} {order_sql}"
    ).df()


def fetch_trials(sessions, columns=None, base=None, con=None, snapshot=_UNSET):
    """Pull trial rows for a set of selected sessions, with session metadata joined on.

    Reads only the selected subjects' partitions (fast) and inner-joins to ``sessions`` on
    the session key, so exactly the selected sessions' trials are returned — each row
    carrying its session metadata.

    Parameters
    ----------
    sessions : pandas.DataFrame
        Selected sessions (e.g. from :func:`select_sessions`). Must contain ``_session_id``
        and ``subject_id``; every other column is carried onto each trial row.
    columns : list[str] or "*", optional
        Trial columns to project (default: a small choice/reward set). ``"*"`` returns all
        103 columns (large). Columns absent for the selected subjects come back all-NULL.
        The within-session index ``trial`` is **always** included (you can't identify a trial
        without it), so you never need to list it — it leads the projected columns.
    base : str, optional
        Trial-table **directory** prefix (default: the production S3 ``trial_table``). Pass a
        local dir / other S3 prefix to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).
    snapshot : str, optional
        Read from a snapshot (``"20260604"``) instead of the latest database, overriding any
        global :func:`use_snapshot` for this call. Pass ``None`` to force latest. Ignored if
        ``base`` is given. See :func:`use_snapshot`.

    Returns
    -------
    pandas.DataFrame
        One row per trial, leading ``subject_id, session_date, session_id, trial``, ordered by
        ``subject_id, session_date, trial``.
    """
    return _fetch(sessions, _resolve_base(base, snapshot, "trial"),
                  columns or DEFAULT_TRIAL_COLUMNS, con, order_tail="trial", lead="trial")


def fetch_events(sessions, events=None, columns=None, base=None, con=None, snapshot=_UNSET):
    """Pull event rows for a set of selected sessions, with session metadata joined on.

    Like :func:`fetch_trials`, for the event table.

    Parameters
    ----------
    sessions : pandas.DataFrame
        Selected sessions (needs ``_session_id`` and ``subject_id``).
    events : iterable, optional
        Restrict to these event types, e.g. ``['left_lick_time', 'right_lick_time']``.
    columns : list[str] or "*", optional
        Event columns to project (default: ``trial, timestamps, event, data``). Columns absent
        for the selected subjects come back all-NULL.
    base : str, optional
        Event-table **directory** prefix (default: the production S3 ``event_table``). Pass a
        local dir / other S3 prefix to query another build.
    con : duckdb connection, optional
        DuckDB connection to run on (default: the module connection). Pass your own for warm
        reuse across calls, or custom settings (S3 region/creds, threads, memory).
    snapshot : str, optional
        Read from a snapshot (``"20260604"``) instead of the latest database, overriding any
        global :func:`use_snapshot` for this call. Pass ``None`` to force latest. Ignored if
        ``base`` is given. See :func:`use_snapshot`.

    Returns
    -------
    pandas.DataFrame
        One row per event, leading ``subject_id, session_date, session_id``, ordered by
        ``subject_id, session_date, timestamps``.
    """
    extra_where = f"t.event IN ({_quote_in(events)})" if events else None
    return _fetch(sessions, _resolve_base(base, snapshot, "event"),
                  columns or DEFAULT_EVENT_COLUMNS, con, order_tail="timestamps",
                  extra_where=extra_where)


def _fetch(sessions, base, columns, con, order_tail, extra_where=None, lead=None):
    """Shared core for fetch_trials / fetch_events: scoped read + join to selected sessions.

    Runs optimistically — ``union_by_name`` already null-fills columns missing from *some* selected
    files. Only if a requested column (or the sort key) is absent from *every* selected file (a
    BinderError) do we DESCRIBE the source, null-pad those columns / drop the sort key, and retry.
    This keeps the common path off the expensive full-footer DESCRIBE.
    """
    import pandas as pd

    if len(sessions) == 0:
        return pd.DataFrame()
    conn = _conn(con)
    src = _scoped_read(base, sessions["subject_id"].unique().tolist(), con)
    conn.register("_sel_sessions", sessions)
    try:
        try:
            return _run_fetch(conn, src, sessions, columns, order_tail, extra_where, None, lead)
        except duckdb.BinderException:
            avail = set(conn.sql(f"DESCRIBE SELECT * FROM {src}").df()["column_name"])
            return _run_fetch(conn, src, sessions, columns, order_tail, extra_where, avail, lead)
    finally:
        conn.unregister("_sel_sessions")


def _col_expr(col, avail):
    """Project trial/event column ``col``, null-padding (as DOUBLE) if absent from every file."""
    return f"t.{col}" if (avail is None or col in avail) else f"CAST(NULL AS DOUBLE) AS {col}"


def _run_fetch(conn, src, sessions, columns, order_tail, extra_where, avail, lead=None):
    """Build + run the scoped fetch query and join in the selected sessions' metadata.

    ``avail=None`` projects every requested column as-is (optimistic). A set of available columns
    null-pads any that are missing and drops the sort key if it's absent. ``lead`` (if given) is
    always projected, right after the session keys and before the carried metadata — so e.g.
    ``trial`` stays adjacent to the identity columns regardless of the requested ``columns``.
    """
    meta = [f"s.{c}" for c in sessions.columns if c not in ("_session_id", *_KEYS)]
    lead_proj = [_col_expr(lead, avail)] if lead else []
    if columns in ("*", ["*"]):
        excl = [k for k in _KEYS if avail is None or k in avail]
        if lead and (avail is None or lead in avail):
            excl.append(lead)  # pulled to the front via lead_proj instead of left in t.*
        proj = [f"t.* EXCLUDE ({', '.join(excl)})"]
    else:
        proj = [_col_expr(c, avail) for c in columns if c not in _KEYS and c != lead]
    select = ", ".join(["s.subject_id", "s.session_date", "t.session_id", *lead_proj, *meta, *proj])
    where_sql = f"WHERE {extra_where}" if extra_where else ""
    order = ["s.subject_id", "s.session_date"]
    if avail is None or order_tail in avail:
        order.append(f"t.{order_tail}")
    return conn.sql(f"""
        SELECT {select}
        FROM {src} t
        JOIN _sel_sessions s ON t.session_id = s._session_id
        {where_sql}
        ORDER BY {', '.join(order)}
    """).df()
