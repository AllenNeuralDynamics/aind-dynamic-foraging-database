"""Tests for aind-dynamic-foraging-database."""
import os

# Tests build tiny local caches; never kick off the prod-S3 background warmup on import.
os.environ.setdefault("AIND_DF_NO_WARMUP", "1")
