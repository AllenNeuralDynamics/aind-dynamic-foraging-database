"""
aind-dynamic-foraging-database — query (and build) the AIND dynamic-foraging parquet database.

Querying is lightweight (just ``duckdb`` + ``pandas``). Reach for the helpers first; drop to
native DuckDB SQL when you need more::

    from aind_dynamic_foraging_database import select_sessions, fetch_trials
    sel = select_sessions("task LIKE '%Uncoupled%' AND finished_rate > 0.9")
    trials = fetch_trials(sel, columns=["animal_response", "earned_reward"])

The default read targets (``SESSION_DB`` / ``TRIAL_DB`` / ``EVENT_DB``) live on a public S3 bucket,
so reading needs no AWS credentials.

Building/extending the database lives in ``build_cache`` / ``util.parquet_builder`` and needs the
optional ``[build]`` extra (NWB readers + ``aind-dynamic-foraging-data-utils``); it is intentionally
**not** imported here, so importing this package to *query* stays lightweight.
"""

__version__ = "0.0.1"

from aind_dynamic_foraging_database.query import (  # noqa: F401
    EVENT_DB,
    PROD_S3_PREFIX,
    SESSION_DB,
    TRIAL_DB,
    fetch_events,
    fetch_trials,
    read_events,
    read_trials,
    select_sessions,
)
