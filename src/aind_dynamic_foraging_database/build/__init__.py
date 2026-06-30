"""
Build / maintain the foraging database (needs the optional ``[build]`` extra).

Two convenience entry points are re-exported here::

    from aind_dynamic_foraging_database.build import update_database, create_snapshot
    update_database(n_workers=64)   # incremental update from all sources
    create_snapshot("20260604")     # freeze the current state to a dated snapshot

Lower-level building lives in :mod:`.build_cache` / :mod:`.parquet_builder`.
"""

from aind_dynamic_foraging_database.build.build_cache import update_database  # noqa: F401
from aind_dynamic_foraging_database.build.snapshot import create_snapshot  # noqa: F401
