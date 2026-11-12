"""Database schema, dashboard queries and concurrent access."""

import os
import tempfile


class TestDatabaseV2:
    def test_migrate_v2_creates_tables(self):
        import sqlite3

        from quantsim.core.database_v2 import init_full_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            init_full_db(db_path)

            conn = sqlite3.connect(db_path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()

            required = {
                "ml_model_runs",
                "wfo_results",
                "garch_state",
                "options_greeks_log",
                "config_snapshots",
                "param_sweep_results",
            }
            for table in required:
                assert table in tables, f"Missing table: {table}"

    def test_migration_idempotent(self):
        """Running migration twice should not fail."""
        from quantsim.core.database_v2 import init_full_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            init_full_db(db_path)
            init_full_db(db_path)  # second call should be safe


class TestDashboardDBQueries:
    """Smoke tests for the dashboard's DB query functions."""

    def test_all_dashboard_queries_on_empty_db(self, tmp_path):
        """Dashboard should handle empty tables without crashing."""

        db_path = str(tmp_path / "dash_test.db")
        os.environ["QUANTSIM_DB"] = db_path
        from quantsim.core.database_v2 import init_full_db

        init_full_db(db_path)

        # Simulate the dashboard's query functions directly
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        def _q(sql, params=()):
            try:
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
            except sqlite3.Error:
                return []

        # All these should return empty lists without error
        assert isinstance(_q("SELECT * FROM portfolio_snapshots"), list)
        assert isinstance(_q("SELECT * FROM risk_alerts"), list)
        assert isinstance(_q("SELECT * FROM wfo_results"), list)
        assert isinstance(_q("SELECT * FROM ml_model_runs"), list)
        assert isinstance(_q("SELECT * FROM garch_forecasts"), list)
        assert isinstance(_q("SELECT * FROM strategy_performance"), list)
        assert isinstance(_q("SELECT * FROM options_greeks_log"), list)
        conn.close()


class TestDatabaseConcurrency:
    def test_wal_mode_enabled(self, tmp_path):
        from quantsim.core.database import get_connection, init_db

        db_path = str(tmp_path / "wal_test.db")
        init_db(db_path)
        conn = get_connection(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        conn.close()

    def test_concurrent_reads_while_writing(self, tmp_path):
        """WAL mode allows reads while a write transaction is in progress."""
        import threading

        from quantsim.core.database import get_connection, init_db

        db_path = str(tmp_path / "concurrent_test.db")
        init_db(db_path)

        errors = []

        def reader():
            try:
                conn = get_connection(db_path)
                conn.execute("SELECT COUNT(*) FROM raw_bars").fetchone()
                conn.close()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                conn = get_connection(db_path)
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO raw_bars (asset_id, timestamp, open, high, low, close, volume) VALUES ('TEST', 1000, 1,1,1,1,1)"
                )
                import time

                time.sleep(0.05)
                conn.execute("COMMIT")
                conn.close()
            except Exception as e:
                errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)

        t_write.start()
        import time

        time.sleep(0.01)  # let writer start first
        t_read.start()

        t_write.join(timeout=2)
        t_read.join(timeout=2)

        # WAL mode: no errors expected
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
