#!/usr/bin/env python
"""
Code Ocean reproducible-run: incrementally update the dynamic-foraging database, then archive
this run's build log + provenance JSONs into /results (Code Ocean versions /results per run).

Incremental by default (only sessions not already in build_metadata.json are processed) and
covers all sources in one pass. The heavy lifting is the package's update_database(); this script
just parses args, runs it, and copies the provenance artifacts into /results.
"""
import argparse
import logging
import os

import s3fs

from aind_dynamic_foraging_database.build import update_database
from aind_dynamic_foraging_database.build.build_cache import PROD_S3_OUT_DIR

RESULTS_DIR = "/results"
ARTIFACTS = (
    "build_history.json",
    "build_metadata.json",
    "processing_log.csv",
    "co_skipped_sessions.csv",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_update_dfdb")


def archive_provenance():
    """Copy build history/metadata, triage CSVs, and this run's raw log from S3 into /results."""
    fs = s3fs.S3FileSystem(anon=False)
    base = PROD_S3_OUT_DIR[len("s3://"):]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in ARTIFACTS:
        src = f"{base}/{name}"
        if fs.exists(src):
            fs.get(src, os.path.join(RESULTS_DIR, name))
            logger.info("archived %s", name)
    if fs.exists(f"{base}/logs"):
        raw = sorted(p for p in fs.ls(f"{base}/logs") if p.endswith(".log"))
        if raw:  # newest per-build log = this run's
            fs.get(raw[-1], os.path.join(RESULTS_DIR, os.path.basename(raw[-1])))
            logger.info("archived %s", os.path.basename(raw[-1]))


def main():
    """Parse args, run the incremental update, then archive provenance into /results."""
    p = argparse.ArgumentParser(description="Incrementally update the dynamic-foraging database.")
    p.add_argument("--n-workers", type=int, default=96, help="worker processes (default: 96)")
    p.add_argument("--cutoff-date", default=None,
                   help="only process sessions dated <= this YYYY-MM-DD (default: all)")
    args = p.parse_args()

    logger.info("updating %s (n_workers=%s, cutoff_date=%s)",
                PROD_S3_OUT_DIR, args.n_workers, args.cutoff_date)
    update_database(n_workers=args.n_workers, cutoff_date=args.cutoff_date)
    archive_provenance()
    logger.info("done; provenance copied to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
