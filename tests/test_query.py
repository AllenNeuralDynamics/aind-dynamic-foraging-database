"""
Tests for the DuckDB query helpers (foraging_cache.query).

Builds a tiny local cache from the real NWB files in tests/nwb/, writes a minimal
session_table.parquet whose keys match the built trials, then exercises the helpers
against that local build via ``base=`` — no S3 / network needed.
"""

import glob
import os
import tempfile
import unittest

import duckdb
import pandas as pd

from aind_dynamic_foraging_database import query
from aind_dynamic_foraging_database.build import parquet_builder
from aind_dynamic_foraging_database.build.snapshot import create_snapshot

TEST_NWB_DIR = "./tests/nwb"


def _build_local_cache(tmpdir):
    """Build trial+event tables from tests/nwb/ and a matching session_table.parquet.

    Returns (session_path, trial_prefix, event_prefix). The session table's ``_session_id``
    is taken from the built trials so the helper joins line up exactly, plus two fake
    metadata columns (``task``, ``foraging_eff``) to exercise filtering + metadata join.
    """
    nwb_files = glob.glob(os.path.join(TEST_NWB_DIR, "*.nwb"))
    rows = []
    for fpath in nwb_files:
        parsed = parquet_builder._parse_nwb_filename(fpath)
        if parsed is None:
            continue
        subject_id, session_date, nwb_suffix = parsed
        rows.append({"subject_id": subject_id, "session_date": session_date,
                     "nwb_suffix": nwb_suffix, "data_source": "bonsai",
                     "co_asset_id": None, "co_s3_nwb_uri": None,
                     "nwb_data_source": "bonsai_s3"})
    session_df = pd.DataFrame(rows)

    trial_prefix = os.path.join(tmpdir, "trial_table")
    event_prefix = os.path.join(tmpdir, "event_table")
    nwb_index = parquet_builder.build_nwb_file_index(bonsai_dir=TEST_NWB_DIR, bpod_dir=TEST_NWB_DIR)
    parquet_builder.build_trial_and_event_tables(
        session_df=session_df, trial_output_prefix=trial_prefix,
        event_output_prefix=event_prefix, nwb_file_index=nwb_index,
        incremental=False, verbose=False)

    # Derive a session table from the built trials so _session_id matches session_id.
    sess = duckdb.sql(
        f"SELECT DISTINCT session_id AS _session_id, CAST(subject_id AS VARCHAR) AS subject_id, "
        f"session_date FROM read_parquet('{trial_prefix}/**/*.parquet', "
        f"hive_partitioning=true, union_by_name=true)").df()
    sess["task"] = "TestTask"
    sess["foraging_eff"] = 0.5
    session_path = os.path.join(tmpdir, "session_table.parquet")
    sess.to_parquet(session_path, index=False)
    return session_path, trial_prefix, event_prefix


class TestQueryHelpers(unittest.TestCase):
    """End-to-end helper tests against a local build."""

    def test_select_fetch_and_escape_hatch(self):
        """select_sessions -> fetch_trials/fetch_events + the read_trials escape hatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path, trial_prefix, event_prefix = _build_local_cache(tmpdir)

            # --- select_sessions: metric filter, carrying metadata ---
            sel = query.select_sessions("foraging_eff > 0", base=session_path,
                                        columns=["task", "foraging_eff"])
            self.assertGreater(len(sel), 0)
            self.assertIn("_session_id", sel.columns)
            self.assertEqual(list(sel.columns[:3]), ["_session_id", "subject_id", "session_date"])

            # --- fetch_trials: scoped read + metadata join ---
            tr = query.fetch_trials(sel, base=trial_prefix,
                                    columns=["animal_response", "earned_reward"])
            self.assertGreater(len(tr), 0)
            # keys + the always-returned within-session 'trial' index lead the columns,
            # even though 'trial' was not among the requested columns
            self.assertEqual(list(tr.columns[:4]),
                             ["subject_id", "session_date", "session_id", "trial"])
            self.assertEqual(tr["trial"].isna().sum(), 0)
            for col in ["task", "foraging_eff", "animal_response", "earned_reward"]:
                self.assertIn(col, tr.columns)
            # exactly the selected sessions, nothing extra
            self.assertTrue(set(tr["session_id"]).issubset(set(sel["_session_id"])))
            # ordered by subject_id, session_date, trial
            self.assertTrue(tr["subject_id"].is_monotonic_increasing)

            # --- fetch_events: with event-type filter ---
            ev = query.fetch_events(sel, base=event_prefix,
                                    events=["left_lick_time", "right_lick_time"],
                                    columns=["trial", "timestamps", "event"])
            self.assertGreater(len(ev), 0)
            self.assertEqual(set(ev["event"].unique()), {"left_lick_time", "right_lick_time"})

            # --- read_trials escape hatch: scoped source drives arbitrary SQL ---
            subs = sel["subject_id"].unique().tolist()
            src = query.read_trials(subs, base=trial_prefix)
            n_scoped = duckdb.sql(f"SELECT COUNT(*) c FROM {src}").df().c[0]
            n_all = duckdb.sql(
                f"SELECT COUNT(*) c FROM read_parquet('{trial_prefix}/**/*.parquet', "
                f"hive_partitioning=true, union_by_name=true)").df().c[0]
            self.assertEqual(n_scoped, n_all)  # only subject(s) in this tiny cache

    def test_read_trials_edge_cases(self):
        """read_trials: full-glob fallback and empty-source path for a missing subject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, trial_prefix, _ = _build_local_cache(tmpdir)
            # subjects=None -> full glob
            self.assertIn("**", query.read_trials(base=trial_prefix))
            # a subject with no partition -> safe empty source (no error)
            empty_src = query.read_trials(["000000"], base=trial_prefix)
            self.assertEqual(int(duckdb.sql(f"SELECT COUNT(*) c FROM {empty_src}").df().c[0]), 0)

    def test_fetch_empty_selection(self):
        """fetch_trials returns an empty frame when the session selection is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path, trial_prefix, _ = _build_local_cache(tmpdir)
            empty = query.select_sessions("foraging_eff > 999", base=session_path)
            self.assertEqual(len(empty), 0)
            self.assertEqual(len(query.fetch_trials(empty, base=trial_prefix)), 0)


class TestSnapshots(unittest.TestCase):
    """Snapshot creation (local) and the use_snapshot / snapshot= read selector."""

    def tearDown(self):
        """Always reset the global snapshot so tests don't leak state into each other."""
        query.use_snapshot(None)

    def test_resolve_base_precedence(self):
        """_resolve_base: explicit base > per-call snapshot > global > latest default."""
        P = query.PROD_S3_PREFIX
        # default (no snapshot) -> the live module defaults
        self.assertEqual(query._resolve_base(None, query._UNSET, "session"), query.SESSION_DB)
        self.assertEqual(query._resolve_base(None, query._UNSET, "trial"), query.TRIAL_DB)
        self.assertEqual(query._resolve_base(None, query._UNSET, "event"), query.EVENT_DB)
        # explicit base always wins, even with a snapshot passed
        self.assertEqual(query._resolve_base("/x", query._UNSET, "trial"), "/x")
        self.assertEqual(query._resolve_base("/x", "20260604", "trial"), "/x")
        # per-call snapshot -> snapshots/<date>/<suffix>
        self.assertEqual(query._resolve_base(None, "20260604", "session"),
                         f"{P}/snapshots/20260604/session_table.parquet")
        self.assertEqual(query._resolve_base(None, "20260604", "trial"),
                         f"{P}/snapshots/20260604/trial_table")
        self.assertEqual(query._resolve_base(None, "20260604", "event"),
                         f"{P}/snapshots/20260604/event_table")
        # global setter + round-trip; explicit None overrides the global back to latest
        query.use_snapshot("20260604")
        self.assertEqual(query.current_snapshot(), "20260604")
        self.assertEqual(query._resolve_base(None, query._UNSET, "trial"),
                         f"{P}/snapshots/20260604/trial_table")
        self.assertEqual(query._resolve_base(None, None, "trial"), query.TRIAL_DB)
        query.use_snapshot(None)
        self.assertIsNone(query.current_snapshot())
        self.assertEqual(query._resolve_base(None, query._UNSET, "trial"), query.TRIAL_DB)

    def test_path_accessors_honour_snapshot(self):
        """session_db/trial_db/event_db track the global + per-call snapshot (unlike constants)."""
        P = query.PROD_S3_PREFIX
        # latest by default
        self.assertEqual(query.session_db(), query.SESSION_DB)
        self.assertEqual(query.trial_db(), query.TRIAL_DB)
        self.assertEqual(query.event_db(), query.EVENT_DB)
        # per-call snapshot
        self.assertEqual(query.trial_db(snapshot="20260604"),
                         f"{P}/snapshots/20260604/trial_table")
        # global snapshot is honoured; explicit None overrides back to latest
        query.use_snapshot("20260604")
        self.assertEqual(query.session_db(), f"{P}/snapshots/20260604/session_table.parquet")
        self.assertEqual(query.event_db(), f"{P}/snapshots/20260604/event_table")
        self.assertEqual(query.trial_db(snapshot=None), query.TRIAL_DB)
        query.use_snapshot(None)

    def test_use_snapshot_clears_partition_cache(self):
        """Switching the global snapshot drops the per-base partition listing cache."""
        query._PARTITION_CACHE["some/base"] = {"754372"}
        query.use_snapshot("20260604")
        self.assertEqual(query._PARTITION_CACHE, {})

    def test_create_snapshot_local_and_query_through_selector(self):
        """create_snapshot copies all tables (excl. snapshots/); snapshot= reads match latest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)
            session_path, trial_prefix, event_prefix = _build_local_cache(cache_dir)
            # a pre-existing snapshots/ dir must NOT be copied into the new snapshot
            os.makedirs(os.path.join(cache_dir, "snapshots", "old"))

            # build provenance files that the snapshot should freeze + summarize in its manifest
            import json
            with open(os.path.join(cache_dir, "build_history.json"), "w") as f:
                json.dump([{"build_id": "20260604T101530Z", "n_processed": 3}], f)
            with open(os.path.join(cache_dir, "build_metadata.json"), "w") as f:
                json.dump({"n_processed": 3}, f)

            snap = create_snapshot("20260604", prefix=cache_dir)
            self.assertEqual(snap, os.path.join(cache_dir, "snapshots", "20260604"))
            self.assertEqual(
                sorted(os.listdir(snap)),
                ["build_history.json", "build_metadata.json", "event_table",
                 "session_table.parquet", "snapshot_manifest.json", "trial_table"])
            # manifest captures provenance from the copied build files
            manifest = json.load(open(os.path.join(snap, "snapshot_manifest.json")))
            self.assertEqual(manifest["snapshot_date"], "20260604")
            self.assertEqual(manifest["latest_build_id"], "20260604T101530Z")
            self.assertEqual(manifest["n_sessions"], 3)
            self.assertEqual(manifest["source_prefix"], cache_dir)
            self.assertIn("created_at_utc", manifest)

            # latest (live) read
            latest = query.select_sessions("foraging_eff > 0", base=session_path)
            # snapshot read via the global selector: point PROD_S3_PREFIX at our local cache
            orig_prefix = query.PROD_S3_PREFIX
            try:
                query.PROD_S3_PREFIX = cache_dir
                query.use_snapshot("20260604")
                snap_sel = query.select_sessions("foraging_eff > 0")
                snap_tr = query.fetch_trials(snap_sel, columns=["animal_response"])
            finally:
                query.PROD_S3_PREFIX = orig_prefix
                query.use_snapshot(None)
            # the snapshot is a faithful copy -> same sessions and trial counts as latest
            self.assertEqual(sorted(snap_sel["_session_id"]), sorted(latest["_session_id"]))
            latest_tr = query.fetch_trials(latest, base=trial_prefix, columns=["animal_response"])
            self.assertEqual(len(snap_tr), len(latest_tr))

    def test_create_snapshot_overwrite_and_bad_date(self):
        """create_snapshot guards an existing snapshot and rejects a non-YYYYMMDD date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)
            _build_local_cache(cache_dir)
            create_snapshot("20260604", prefix=cache_dir)
            with self.assertRaises(FileExistsError):
                create_snapshot("20260604", prefix=cache_dir)
            create_snapshot("20260604", prefix=cache_dir, overwrite=True)  # ok
            with self.assertRaises(ValueError):
                create_snapshot("2026-06-04", prefix=cache_dir)


class TestCutoffDate(unittest.TestCase):
    """The session-date cutoff filter and its CLI wiring."""

    def test_apply_session_cutoff(self):
        """_apply_session_cutoff keeps only sessions on/before the cutoff; None is a no-op."""
        df = pd.DataFrame(
            {"session_date": ["2026-05-30", "2026-06-01", "2026-06-04", "2026-06-10"]})
        self.assertEqual(len(parquet_builder._apply_session_cutoff(df, None)), 4)
        kept = parquet_builder._apply_session_cutoff(df, "2026-06-04", verbose=False)
        self.assertEqual(sorted(kept["session_date"]),
                         ["2026-05-30", "2026-06-01", "2026-06-04"])
        # accepts other date spellings (normalised before compare)
        kept2 = parquet_builder._apply_session_cutoff(df, "2026/06/01", verbose=False)
        self.assertEqual(sorted(kept2["session_date"]), ["2026-05-30", "2026-06-01"])

    def test_cli_cutoff_date_wiring(self):
        """--cutoff-date flows into Config.cutoff_date."""
        from aind_dynamic_foraging_database.build import build_cache
        cfg = build_cache.parse_args(["--cutoff-date", "2026-06-04"])
        self.assertEqual(cfg.cutoff_date, "2026-06-04")
        self.assertIsNone(build_cache.parse_args([]).cutoff_date)


if __name__ == "__main__":
    unittest.main()
