"""
Main entry point to build (or incrementally extend) the foraging parquet cache.

Runs the full pipeline over ALL sessions in the Han session table, exercising
all three NWB routes (see references/data-sources.md):

  - CO asset  -> AIND reader (nwb_utils) on the docDB S3 URI
  - bonsai S3 -> legacy reader (Han bonsai NWB)
  - bpod S3   -> legacy reader (Han bpod NWB)

Incremental by default: only sessions not already recorded in
``build_metadata.json`` are processed, so re-running cheaply adds new sessions.
Pass ``--full-rebuild`` to reprocess everything.

To query the built cache (the read-back / "return loop"), use the companion
``query_examples`` module or ``query_examples.ipynb`` — querying is intentionally
kept out of this build script.

Output target:
  - Default is a local scratch directory (safe for dev iteration).
  - Point ``--out-dir`` at the canonical S3 prefix to write the production
    database:  ``--out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache``

Run:
    # incremental local build (default scratch dir)
    python -m aind_dynamic_foraging_database.build_cache

    # production build/update on S3 (--n-workers 64 ~= 4x faster; see --help)
    python -m aind_dynamic_foraging_database.build_cache \\
        --out-dir s3://aind-scratch-data/aind-dynamic-foraging-cache --n-workers 64

    # quick smoke test on a random 300-session subset (spans all three routes)
    python -m aind_dynamic_foraging_database.build_cache --limit 300

Or drive it programmatically (the module is import-safe — nothing runs on import):
    from aind_dynamic_foraging_database.build import build_cache as b
    b.main(b.Config(out_dir="/root/capsule/scratch/tmp/foraging_cache", limit=300))
"""

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from aind_dynamic_foraging_database import __version__
from aind_dynamic_foraging_database.build import parquet_builder

logger = logging.getLogger(__name__)

# Default output: local scratch (never /tmp). Use --out-dir for S3 prod.
DEFAULT_OUT_DIR = "/root/capsule/scratch/tmp/foraging_cache"
PROD_S3_OUT_DIR = "s3://aind-scratch-data/aind-dynamic-foraging-cache"


@dataclass
class Config:
    """Inputs and derived output paths for one build."""

    out_dir: str = DEFAULT_OUT_DIR
    limit: Optional[int] = None  # cap sessions for a quick test (random subset)
    full_rebuild: bool = False  # ignore build metadata; reprocess everything
    random_seed: int = 42
    n_workers: Optional[int] = None  # worker processes; None -> CO_CPUS-1
    coalesce: bool = True  # merge each subject's sessions into one parquet file
    co_cache: Optional[str] = None  # dev: cache the docDB discovery (pickle) here
    cutoff_date: Optional[str] = None  # YYYY-MM-DD: only sessions on/before this date

    @property
    def is_s3(self) -> bool:
        """Whether the output target is an S3 prefix (vs a local dir)."""
        return self.out_dir.startswith("s3://")

    @property
    def session_out(self) -> str:
        """Path to the flat session-table parquet."""
        return f"{self.out_dir}/session_table.parquet"

    @property
    def trial_out(self) -> str:
        """Prefix of the Hive-partitioned trial table."""
        return f"{self.out_dir}/trial_table"

    @property
    def event_out(self) -> str:
        """Prefix of the Hive-partitioned event table."""
        return f"{self.out_dir}/event_table"

    @property
    def meta_out(self) -> str:
        """Path to the incremental-build metadata JSON."""
        return f"{self.out_dir}/build_metadata.json"

    @property
    def log_csv(self) -> str:
        """Path to the human-readable per-session triage log CSV."""
        return f"{self.out_dir}/processing_log.csv"

    @property
    def history_out(self) -> str:
        """Path to the append-only build-history JSON (one entry per build run)."""
        return f"{self.out_dir}/build_history.json"

    @property
    def logs_dir(self) -> str:
        """Directory holding the raw per-build log files (``logs/<build_id>.log``)."""
        return f"{self.out_dir}/logs"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def build_sessions(cfg: Config):
    """
    Build the complete session table: Han ∪ docDB/CO universe, CO assets attached.

    See parquet_builder.build_session_table / _merge_han_and_co for the match
    rule. The ~137 s docDB discovery can be cached locally for dev iteration via
    --co-cache (loaded if present, else fetched once and saved).
    """
    _banner("Building session table (Han ∪ docDB CO universe)")
    return parquet_builder.build_session_table(
        output_path=cfg.session_out,
        include_co_assets=True,
        co_discovery=_load_or_fetch_co_discovery(cfg),
        cutoff_date=cfg.cutoff_date,
        verbose=True,
    )


def _load_or_fetch_co_discovery(cfg: Config):
    """
    Dev helper: if --co-cache is set, load the cached docDB discovery (or fetch once and
    save it). Returns None when no cache is configured, so build_session_table fetches fresh.

    Cached as a **pickle** (not parquet): the docDB result has list-valued columns such as
    ``co_task`` that pyarrow can't serialize, and pickle round-trips the full DataFrame
    unchanged so a cached run matches a fresh fetch.
    """
    if not cfg.co_cache:
        return None
    import pandas as pd

    if os.path.exists(cfg.co_cache):
        print(f"  using cached docDB discovery: {cfg.co_cache}")
        return pd.read_pickle(cfg.co_cache)

    from aind_dynamic_foraging_data_utils.code_ocean_utils import get_dynamic_foraging_assets

    print(f"  fetching docDB discovery, caching -> {cfg.co_cache}")
    co = get_dynamic_foraging_assets()
    co.to_pickle(cfg.co_cache)
    return co


def select_sessions(cfg: Config, session_df):
    """
    Optionally subsample the complete session table for a quick test (--limit N,
    seeded). The full table is still written to session_out by build_sessions;
    only the trial/event build is limited.
    """
    if cfg.limit is None or len(session_df) <= cfg.limit:
        return session_df
    sampled = session_df.sample(n=cfg.limit, random_state=cfg.random_seed)
    print(f"  --limit: randomly sampled {len(sampled)} of {len(session_df)} sessions")
    return sampled.reset_index(drop=True)


def build_trial_event_tables(cfg: Config, session_df, nwb_index: dict) -> dict:
    """Build the Hive-partitioned trial + event tables for the selected sessions."""
    _banner("Building trial and event tables")
    return parquet_builder.build_trial_and_event_tables(
        session_df=session_df,
        trial_output_prefix=cfg.trial_out,
        event_output_prefix=cfg.event_out,
        nwb_file_index=nwb_index,
        build_metadata_path=cfg.meta_out,
        incremental=not cfg.full_rebuild,
        n_workers=cfg.n_workers,
        coalesce=cfg.coalesce,
        log_csv_path=cfg.log_csv,
        verbose=True,
    )


def print_summary(cfg: Config, summary: dict) -> None:
    """Print the build-result breakdown."""
    _banner("BUILD SUMMARY")
    print(f"  Output dir                   : {cfg.out_dir}")
    print(f"  Processed (ok)               : {summary['n_processed']}")
    print(f"  Skipped (no NWB found)       : {summary['n_skipped']}")
    print(f"  Failed                       : {summary['n_failed']}")
    print("\n  Data source breakdown:")
    print(f"    CO asset                   : {summary['n_co_asset']}")
    print(f"    Bonsai S3                  : {summary['n_bonsai_s3']}")
    print(f"    bpod S3                    : {summary['n_bpod_s3']}")
    print("\n  NWB reader breakdown:")
    print(f"    AIND reader (CO asset)     : {summary['n_aind_reader']}")
    print(f"    AIND->legacy fallback      : {summary['n_aind_fallback_legacy']}")
    print(f"    Legacy bonsai              : {summary['n_legacy_bonsai']}")
    print(f"    Legacy bpod                : {summary['n_legacy_bpod']}")
    if summary["failed_sessions"]:
        print(f"\n  Failed sessions ({summary['n_failed']}), first 20:")
        for fs in summary["failed_sessions"][:20]:
            print(f"    [{fs.get('data_source', '?')}] {fs['session_id']}  --  {fs['error']}")
        if summary["n_failed"] > 20:
            print(f"    ... and {summary['n_failed'] - 20} more")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def update_database(n_workers: Optional[int] = 64, **kwargs) -> dict:
    """Incrementally update the production S3 database from all sources, in one call.

    Thin wrapper over :func:`main` pinned to the production prefix. Incremental by default
    (only sessions not already in ``build_metadata.json`` are processed) and covers all three
    sources (CO asset, bonsai S3, bpod S3) in one pass. Pair with
    :func:`aind_dynamic_foraging_database.build.snapshot.create_snapshot` to freeze the result.

    Parameters
    ----------
    n_workers : int, optional
        Worker processes (default: 64; see :class:`Config`).
    **kwargs
        Extra :class:`Config` fields (e.g. ``full_rebuild=True``, ``limit=300``).

    Returns
    -------
    dict
        The build summary (see :func:`print_summary`).
    """
    return main(Config(out_dir=PROD_S3_OUT_DIR, n_workers=n_workers, **kwargs))


def main(cfg: Config) -> dict:
    """Run the full build pipeline end to end, recording build history + a raw log.

    Each run appends one entry to ``build_history.json`` (an append-only ledger of every
    build/append event) and writes the run's raw console+warning log to ``logs/<build_id>.log``,
    both under ``out_dir`` — so they travel with the data into any snapshot. Returns the summary.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # s3fs/aiobotocore emit benign "Task ... attached to a different loop"
    # tracebacks when worker S3 clients are torn down (the read/write already
    # succeeded). Quiet the asyncio logger so they don't clutter the build log.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    if not cfg.is_s3:
        os.makedirs(cfg.out_dir, exist_ok=True)

    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = datetime.now(timezone.utc)
    summary = None
    status, error = "ok", None
    buf = io.StringIO()
    try:
        with _capture_run_log(buf):
            print(f"build_id: {build_id}   code version: {__version__}   git: {_git_commit()}")
            summary = _run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 - record the failure in history/log, then re-raise
        import traceback

        status, error = "error", str(exc)
        traceback.print_exc(file=buf)
        raise
    finally:
        finished = datetime.now(timezone.utc)
        _record_build(cfg, build_id, started, finished, summary, status, error, buf.getvalue())
    return summary


def _run_pipeline(cfg: Config) -> dict:
    """Index NWBs, build the session table, then the trial/event tables. Returns the summary."""
    _banner("Indexing local Han NWB files")
    nwb_index = parquet_builder.build_nwb_file_index()
    print(f"  Total NWB files indexed: {len(nwb_index)}")

    # Build the complete session table (Han ∪ CO universe, CO assets attached),
    # then optionally subsample for a quick --limit test, then build the tables.
    session_df = build_sessions(cfg)
    session_df = select_sessions(cfg, session_df)

    summary = build_trial_event_tables(cfg, session_df, nwb_index)
    print_summary(cfg, summary)
    return summary


class _Tee:
    """A minimal write-through stream that forwards writes/flushes to several streams."""

    def __init__(self, *streams):
        """Store the target streams to forward to."""
        self._streams = streams

    def write(self, s):
        """Write ``s`` to every target stream."""
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        """Flush every target stream."""
        for st in self._streams:
            st.flush()


@contextlib.contextmanager
def _capture_run_log(buf):
    """Tee stdout and WARNING+ logging into ``buf`` while keeping live console output."""
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            yield
    finally:
        root.removeHandler(handler)


def _git_commit() -> Optional[str]:
    """Return the build code's git commit (``<sha>`` or ``<sha>-dirty``), or ``None``.

    Best-effort, in order: (1) a live git checkout (dev / mounted source with ``.git``), then
    (2) the commit pip recorded when the package was installed from git (PEP 610
    ``direct_url.json`` -> ``vcs_info.commit_id``). (2) is what populates the commit inside a
    Code Ocean Reproducible Run, where the package is pip-installed from ``git+...@<ref>`` into
    site-packages (no ``.git``, so the git command finds nothing). Returns ``None`` for a plain
    wheel/PyPI install. Never fails a build.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0:
            commit = sha.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", pkg_dir, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                commit += "-dirty"
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    # Fall back to the commit pip recorded for a git install (no .git in site-packages).
    try:
        from importlib.metadata import distribution

        raw = distribution("aind-dynamic-foraging-database").read_text("direct_url.json")
        if raw:
            return (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    except Exception:  # noqa: BLE001 - best-effort provenance; never fail a build
        pass
    return None


def _record_build(cfg, build_id, started, finished, summary, status, error, log_text) -> None:
    """Append a build-history entry and write the raw run log (best-effort; never fails a build)."""
    event = {
        "build_id": build_id,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 1),
        "status": status,
        "code_version": __version__,
        "git_commit": _git_commit(),
        "out_dir": cfg.out_dir,
        "incremental": not cfg.full_rebuild,
        "limit": cfg.limit,
        "cutoff_date": cfg.cutoff_date,
        "n_workers": cfg.n_workers,
    }
    if summary is not None:
        event.update({
            "n_processed": summary["n_processed"],
            "n_skipped": summary["n_skipped"],
            "n_failed": summary["n_failed"],
            "source_breakdown": {
                "co_asset": summary["n_co_asset"],
                "bonsai_s3": summary["n_bonsai_s3"],
                "bpod_s3": summary["n_bpod_s3"],
            },
            "reader_breakdown": {
                "aind": summary["n_aind_reader"],
                "aind_fallback_legacy": summary["n_aind_fallback_legacy"],
                "legacy_bonsai": summary["n_legacy_bonsai"],
                "legacy_bpod": summary["n_legacy_bpod"],
            },
        })
    if error is not None:
        event["error"] = error
    try:
        _append_build_history(cfg.history_out, event)
        parquet_builder._write_text(log_text, f"{cfg.logs_dir}/{build_id}.log")
    except Exception as exc:  # noqa: BLE001 - logging must never sink the build itself
        logger.warning("failed to write build history/log: %s", exc)


def _append_build_history(path, event) -> None:
    """Append ``event`` to the JSON-array build-history file (read-modify-write; absent is ok)."""
    try:
        history = parquet_builder._read_json(path)
        if not isinstance(history, list):
            history = []
    except FileNotFoundError:
        history = []
    history.append(event)
    parquet_builder._write_json(history, path)


def parse_args(argv=None) -> Config:
    """Parse CLI arguments into a Config."""
    p = argparse.ArgumentParser(description="Build/extend the foraging parquet cache.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"output dir or S3 prefix (default: %(default)s; "
                        f"prod: {PROD_S3_OUT_DIR})")
    p.add_argument("--limit", type=int, default=None,
                   help="cap to a random N-session subset for a quick test (default: all)")
    p.add_argument("--full-rebuild", action="store_true",
                   help="ignore build metadata and reprocess every session")
    p.add_argument("--n-workers", type=int, default=None,
                   help="worker processes (default: CO_CPUS-1). CO-asset reads are "
                        "I/O-bound, so oversubscribing past CPU count overlaps S3 "
                        "latency. Recommended ~64 on a 16-core box: ~4x faster than "
                        "the default, and beyond ~64 there's no gain (the "
                        "create_df_trials parse saturates the cores). RAM is not "
                        "the limit (~21 GB at 128 workers).")
    p.add_argument("--no-coalesce", action="store_true",
                   help="keep one parquet file per session instead of merging each "
                        "subject's sessions into a single sorted file (the default)")
    p.add_argument("--co-cache", default=None,
                   help="dev: path to cache the ~137s docDB discovery "
                        "(loaded if present, else fetched once and saved)")
    p.add_argument("--cutoff-date", default=None,
                   help="YYYY-MM-DD: build only sessions on/before this date, for a "
                        "reproducible 'as of <date>' build (default: all sessions)")
    args = p.parse_args(argv)
    return Config(
        out_dir=args.out_dir,
        limit=args.limit,
        full_rebuild=args.full_rebuild,
        n_workers=args.n_workers,
        coalesce=not args.no_coalesce,
        co_cache=args.co_cache,
        cutoff_date=args.cutoff_date,
    )


def _banner(title: str) -> None:
    """Print a titled banner to delimit pipeline stages in the log."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _cli():
    """Console-script entry point (``aind-df-build``): parse argv and run the build."""
    main(parse_args())


if __name__ == "__main__":
    _cli()
