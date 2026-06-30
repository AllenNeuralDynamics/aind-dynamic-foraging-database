"""
Create immutable, dated snapshots of the whole foraging database.

A snapshot is a frozen copy of the live database prefix — all tables *and* logs
(``session_table.parquet``, ``trial_table/``, ``event_table/``, ``build_metadata.json``,
``processing_log.csv``, …) — written under ``<prefix>/snapshots/<date>/``. The live database
is updated in place (incrementally), so a snapshot lets an analysis or training run pin a
reproducible data source even as the live prefix keeps growing.

Read a snapshot back with the query-side selector::

    import aind_dynamic_foraging_database as db
    db.use_snapshot("20260604")           # all reads now hit snapshots/20260604/
    db.select_sessions(subjects=["754372"])

Creating a snapshot writes to S3, so it needs the ``[build]`` extra (``s3fs``) and write
credentials for the bucket. The copy is **server-side** (S3 CopyObject — no download).
"""

import os
import shutil
from datetime import datetime, timezone

from aind_dynamic_foraging_database.query import PROD_S3_PREFIX

# Top-level name that holds snapshots; never copied into a snapshot itself.
_SNAPSHOTS_DIRNAME = "snapshots"


def _today_utc() -> str:
    """Return today's date (UTC) as a ``YYYYMMDD`` string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _validate_date(date: str) -> str:
    """Return ``date`` if it is a valid ``YYYYMMDD`` string, else raise ``ValueError``."""
    try:
        datetime.strptime(date, "%Y%m%d")
    except (ValueError, TypeError):
        raise ValueError(f"snapshot date must be 'YYYYMMDD', got {date!r}")
    return date


def create_snapshot(date=None, prefix=PROD_S3_PREFIX, *, overwrite=False) -> str:
    """Copy the whole database to an immutable snapshot at ``<prefix>/snapshots/<date>/``.

    Copies every top-level entry under ``prefix`` (all tables and logs) **except** the
    ``snapshots/`` directory, so snapshots never nest. On S3 the copy is server-side
    (no download); a local ``prefix`` is supported too (for tests / dev).

    Parameters
    ----------
    date : str, optional
        Snapshot id, ``YYYYMMDD`` (e.g. ``"20260604"``). Defaults to today (UTC).
    prefix : str, optional
        Database prefix to snapshot — an ``s3://`` URI (default: the production prefix) or a
        local directory.
    overwrite : bool, optional
        If the snapshot already exists, overwrite it instead of raising (default: ``False``).

    Returns
    -------
    str
        The snapshot prefix that was written (e.g. ``"s3://.../snapshots/20260604"``).
    """
    date = _validate_date(date or _today_utc())
    prefix = prefix.rstrip("/")
    dest = f"{prefix}/{_SNAPSHOTS_DIRNAME}/{date}"
    if prefix.startswith("s3://"):
        _create_snapshot_s3(prefix, dest, overwrite)
    else:
        _create_snapshot_local(prefix, dest, overwrite)
    return dest


def _create_snapshot_s3(prefix: str, dest: str, overwrite: bool) -> None:  # pragma: no cover
    """Server-side copy each top-level entry (except ``snapshots/``) from ``prefix`` to ``dest``."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=False)
    dest_key = dest[len("s3://"):]
    if fs.exists(dest_key):
        if not overwrite:
            raise FileExistsError(
                f"snapshot already exists: {dest} (pass overwrite=True to replace it)"
            )
        fs.rm(dest_key, recursive=True)
    for entry in fs.ls(prefix[len("s3://"):], detail=False):
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        if name == _SNAPSHOTS_DIRNAME:
            continue
        fs.copy(entry, f"{dest_key}/{name}", recursive=True)


def _create_snapshot_local(prefix: str, dest: str, overwrite: bool) -> None:
    """Copy every top-level entry (except ``snapshots/``) from local dir ``prefix`` to ``dest``."""
    if os.path.exists(dest):
        if not overwrite:
            raise FileExistsError(
                f"snapshot already exists: {dest} (pass overwrite=True to replace it)"
            )
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(prefix):
        if name == _SNAPSHOTS_DIRNAME:
            continue
        src = os.path.join(prefix, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
