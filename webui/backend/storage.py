import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4


class SessionExpiredError(ValueError):
    pass


class Storage:
    def __init__(self, root: Path):
        self.root = root
        self.database_path = root / "state.db"
        self.uploads_dir = root / "uploads"
        self.outputs_dir = root / "outputs"
        self.attempts_dir = root / "attempts"
        self.bundles_dir = root / "bundles"

        for path in (
            self.uploads_dir,
            self.outputs_dir,
            self.attempts_dir,
            self.bundles_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")

        upload_columns = {row[1] for row in connection.execute("PRAGMA table_info(uploads)").fetchall()}
        legacy_uploads = "path" in upload_columns
        if legacy_uploads:
            connection.execute("ALTER TABLE uploads RENAME TO uploads_legacy")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_blobs (
                content_hash TEXT PRIMARY KEY,
                path TEXT NOT NULL
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_key TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_touched_at REAL NOT NULL,
                UNIQUE (content_hash, filename, mime_type)
            )
            """,
        )
        if legacy_uploads:
            migrated_at = time.time()
            connection.execute(
                """
                INSERT OR IGNORE INTO upload_blobs (content_hash, path)
                SELECT content_hash, path
                FROM uploads_legacy
                """,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO uploads (
                    upload_key,
                    content_hash,
                    filename,
                    mime_type,
                    created_at,
                    last_touched_at
                )
                SELECT upload_key, content_hash, filename, mime_type, ?, ?
                FROM uploads_legacy
                """,
                (migrated_at, migrated_at),
            )
            connection.execute("DROP TABLE uploads_legacy")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model_id TEXT,
                revision INTEGER NOT NULL,
                active_run_keys_json TEXT NOT NULL,
                actions_by_source_json TEXT NOT NULL,
                status TEXT NOT NULL,
                last_activity_at REAL NOT NULL,
                expired_at REAL
            )
            """,
        )
        session_columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
        if "status" not in session_columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            )
        if "last_activity_at" not in session_columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN last_activity_at REAL NOT NULL DEFAULT 0",
            )
            connection.execute(
                "UPDATE sessions SET last_activity_at = ?",
                (time.time(),),
            )
        if "expired_at" not in session_columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN expired_at REAL")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS node_runs (
                node_run_key TEXT PRIMARY KEY,
                record_json TEXT NOT NULL
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                action_key TEXT PRIMARY KEY,
                record_json TEXT NOT NULL
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN (
                        'prepared',
                        'queued',
                        'running',
                        'committing',
                        'completed',
                        'failed'
                    )
                ),
                request_json TEXT NOT NULL,
                runtime_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                reservation_expires_at REAL,
                queue_event_id TEXT,
                queue_lease_expires_at REAL,
                attempt_id TEXT,
                work_dir TEXT
            )
            """,
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS session_tasks_active_session
            ON session_tasks (session_id)
            WHERE state IN ('prepared', 'queued', 'running', 'committing')
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_node_runs (
                session_id TEXT NOT NULL,
                node_run_key TEXT NOT NULL,
                input_upload_keys_json TEXT NOT NULL,
                PRIMARY KEY (session_id, node_run_key)
            )
            """,
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS session_node_runs_node_run
            ON session_node_runs (node_run_key)
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_actions (
                session_id TEXT NOT NULL,
                action_key TEXT NOT NULL,
                PRIMARY KEY (session_id, action_key)
            )
            """,
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS session_actions_action
            ON session_actions (action_key)
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_uploads (
                session_id TEXT NOT NULL,
                upload_key TEXT NOT NULL,
                PRIMARY KEY (session_id, upload_key)
            )
            """,
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS session_uploads_upload
            ON session_uploads (upload_key)
            """,
        )

        session_rows = connection.execute(
            """
            SELECT session_id, active_run_keys_json, actions_by_source_json
            FROM sessions
            WHERE status = 'active'
            """,
        ).fetchall()
        for session_id, active_run_keys_json, actions_by_source_json in session_rows:
            for node_run_key in json.loads(active_run_keys_json):
                record_row = connection.execute(
                    """
                    SELECT record_json
                    FROM node_runs
                    WHERE node_run_key = ?
                    """,
                    (node_run_key,),
                ).fetchone()
                if record_row is None:
                    continue

                input_upload_keys = json.loads(record_row[0])["input_upload_keys"]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_node_runs (
                        session_id,
                        node_run_key,
                        input_upload_keys_json
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        session_id,
                        node_run_key,
                        json.dumps(input_upload_keys),
                    ),
                )
                for upload_key in input_upload_keys:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO session_uploads (session_id, upload_key)
                        VALUES (?, ?)
                        """,
                        (session_id, upload_key),
                    )

            actions_by_source = json.loads(actions_by_source_json)
            for action_keys in actions_by_source.values():
                for action_key in action_keys:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO session_actions (session_id, action_key)
                        VALUES (?, ?)
                        """,
                        (session_id, action_key),
                    )

        connection.commit()
        connection.close()

    def fail_stale_runtime_tasks(self, runtime_id: str, now: float) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE session_tasks
            SET state = 'failed', updated_at = ?
            WHERE runtime_id != ?
              AND state IN ('prepared', 'queued', 'running', 'committing')
            """,
            (now, runtime_id),
        )
        connection.commit()
        connection.close()

    def cleanup_expired_sessions(
        self,
        now: float,
        session_timeout_seconds: int,
    ) -> None:
        stale_before = now - session_timeout_seconds
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE session_tasks
            SET state = 'failed', updated_at = ?
            WHERE (state = 'prepared' AND reservation_expires_at <= ?)
               OR (state = 'queued' AND queue_lease_expires_at <= ?)
            """,
            (now, now, now),
        )
        session_ids = [
            row[0]
            for row in connection.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE status = 'active'
                  AND last_activity_at <= ?
                """,
                (stale_before,),
            ).fetchall()
        ]
        connection.commit()
        connection.close()

        for session_id in session_ids:
            output_directories: set[Path] = set()
            bundle_paths: set[Path] = set()
            upload_paths: set[Path] = set()
            attempt_directories: set[Path] = set()
            connection = sqlite3.connect(self.database_path)
            connection.execute("BEGIN IMMEDIATE")
            session_row = connection.execute(
                """
                SELECT status, last_activity_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if session_row is None or session_row[0] != "active" or session_row[1] > stale_before:
                connection.rollback()
                connection.close()
                continue

            connection.execute(
                """
                UPDATE session_tasks
                SET state = 'failed', updated_at = ?
                WHERE session_id = ?
                  AND (
                        (state = 'prepared' AND reservation_expires_at <= ?)
                     OR (state = 'queued' AND queue_lease_expires_at <= ?)
                  )
                """,
                (now, session_id, now, now),
            )
            active_task = connection.execute(
                """
                SELECT 1
                FROM session_tasks
                WHERE session_id = ?
                  AND state IN ('prepared', 'queued', 'running', 'committing')
                """,
                (session_id,),
            ).fetchone()
            if active_task is not None:
                connection.rollback()
                connection.close()
                continue

            node_run_keys = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT node_run_key
                    FROM session_node_runs
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            ]
            action_keys = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT action_key
                    FROM session_actions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            ]
            upload_keys = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT upload_key
                    FROM session_uploads
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            ]
            for (attempt_id,) in connection.execute(
                """
                SELECT attempt_id
                FROM session_tasks
                WHERE session_id = ?
                  AND attempt_id IS NOT NULL
                """,
                (session_id,),
            ).fetchall():
                attempt_directories.add(self.attempts_dir / attempt_id)

            connection.execute(
                """
                UPDATE sessions
                SET
                    model_id = NULL,
                    revision = 0,
                    active_run_keys_json = '[]',
                    actions_by_source_json = '{}',
                    status = 'expired',
                    expired_at = ?
                WHERE session_id = ?
                """,
                (now, session_id),
            )
            connection.execute(
                "DELETE FROM session_node_runs WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                "DELETE FROM session_actions WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                "DELETE FROM session_uploads WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                "DELETE FROM session_tasks WHERE session_id = ?",
                (session_id,),
            )

            for node_run_key in node_run_keys:
                referenced = connection.execute(
                    """
                    SELECT 1
                    FROM session_node_runs
                    WHERE node_run_key = ?
                    """,
                    (node_run_key,),
                ).fetchone()
                if referenced is None:
                    connection.execute(
                        "DELETE FROM node_runs WHERE node_run_key = ?",
                        (node_run_key,),
                    )
                    output_directories.add(self.outputs_dir / "node_runs" / node_run_key)

            for action_key in action_keys:
                referenced = connection.execute(
                    """
                    SELECT 1
                    FROM session_actions
                    WHERE action_key = ?
                    """,
                    (action_key,),
                ).fetchone()
                if referenced is None:
                    connection.execute(
                        "DELETE FROM actions WHERE action_key = ?",
                        (action_key,),
                    )
                    output_directories.add(self.outputs_dir / "actions" / action_key)
                    bundle_paths.add(self.bundles_dir / f"{action_key}.zip")

            content_hashes: set[str] = set()
            for upload_key in upload_keys:
                referenced = connection.execute(
                    """
                    SELECT 1
                    FROM session_uploads
                    WHERE upload_key = ?
                    """,
                    (upload_key,),
                ).fetchone()
                if referenced is None:
                    upload_row = connection.execute(
                        """
                        SELECT content_hash
                        FROM uploads
                        WHERE upload_key = ?
                        """,
                        (upload_key,),
                    ).fetchone()
                    if upload_row is not None:
                        content_hashes.add(upload_row[0])
                        connection.execute(
                            "DELETE FROM uploads WHERE upload_key = ?",
                            (upload_key,),
                        )

            for content_hash in content_hashes:
                referenced = connection.execute(
                    """
                    SELECT 1
                    FROM uploads
                    WHERE content_hash = ?
                    """,
                    (content_hash,),
                ).fetchone()
                if referenced is None:
                    blob_row = connection.execute(
                        """
                        SELECT path
                        FROM upload_blobs
                        WHERE content_hash = ?
                        """,
                        (content_hash,),
                    ).fetchone()
                    if blob_row is not None:
                        upload_paths.add(Path(blob_row[0]))
                        connection.execute(
                            "DELETE FROM upload_blobs WHERE content_hash = ?",
                            (content_hash,),
                        )

            connection.commit()
            connection.close()

            for path in output_directories | attempt_directories:
                if path.exists():
                    shutil.rmtree(path)
            for path in bundle_paths | upload_paths:
                path.unlink(missing_ok=True)

        attempt_directories: set[Path] = set()
        output_directories: set[Path] = set()
        bundle_paths: set[Path] = set()
        upload_paths: set[Path] = set()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        for (attempt_id,) in connection.execute(
            """
            SELECT attempt_id
            FROM session_tasks
            WHERE state = 'failed'
              AND attempt_id IS NOT NULL
            """,
        ).fetchall():
            attempt_directories.add(self.attempts_dir / attempt_id)
        connection.execute("DELETE FROM session_tasks WHERE state = 'failed'")

        unreferenced_node_runs = [
            row[0]
            for row in connection.execute(
                """
                SELECT node_runs.node_run_key
                FROM node_runs
                LEFT JOIN session_node_runs USING (node_run_key)
                WHERE session_node_runs.node_run_key IS NULL
                """,
            ).fetchall()
        ]
        for node_run_key in unreferenced_node_runs:
            connection.execute(
                "DELETE FROM node_runs WHERE node_run_key = ?",
                (node_run_key,),
            )
            output_directories.add(self.outputs_dir / "node_runs" / node_run_key)

        unreferenced_actions = [
            row[0]
            for row in connection.execute(
                """
                SELECT actions.action_key
                FROM actions
                LEFT JOIN session_actions USING (action_key)
                WHERE session_actions.action_key IS NULL
                """,
            ).fetchall()
        ]
        for action_key in unreferenced_actions:
            connection.execute(
                "DELETE FROM actions WHERE action_key = ?",
                (action_key,),
            )
            output_directories.add(self.outputs_dir / "actions" / action_key)
            bundle_paths.add(self.bundles_dir / f"{action_key}.zip")

        unclaimed_uploads = connection.execute(
            """
            SELECT uploads.upload_key, uploads.content_hash
            FROM uploads
            LEFT JOIN session_uploads USING (upload_key)
            WHERE session_uploads.upload_key IS NULL
              AND uploads.last_touched_at <= ?
            """,
            (stale_before,),
        ).fetchall()
        content_hashes = {row[1] for row in unclaimed_uploads}
        for upload_key, _ in unclaimed_uploads:
            connection.execute(
                "DELETE FROM uploads WHERE upload_key = ?",
                (upload_key,),
            )
        for content_hash in content_hashes:
            referenced = connection.execute(
                "SELECT 1 FROM uploads WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if referenced is None:
                blob_row = connection.execute(
                    "SELECT path FROM upload_blobs WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
                if blob_row is not None:
                    upload_paths.add(Path(blob_row[0]))
                    connection.execute(
                        "DELETE FROM upload_blobs WHERE content_hash = ?",
                        (content_hash,),
                    )

        unreferenced_blobs = connection.execute(
            """
            SELECT upload_blobs.content_hash, upload_blobs.path
            FROM upload_blobs
            LEFT JOIN uploads USING (content_hash)
            WHERE uploads.content_hash IS NULL
            """,
        ).fetchall()
        for content_hash, path in unreferenced_blobs:
            upload_paths.add(Path(path))
            connection.execute(
                "DELETE FROM upload_blobs WHERE content_hash = ?",
                (content_hash,),
            )

        active_attempt_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT attempt_id
                FROM session_tasks
                WHERE state IN ('prepared', 'queued', 'running', 'committing')
                  AND attempt_id IS NOT NULL
                """,
            ).fetchall()
        }
        node_run_keys = {row[0] for row in connection.execute("SELECT node_run_key FROM node_runs").fetchall()}
        action_keys = {row[0] for row in connection.execute("SELECT action_key FROM actions").fetchall()}
        blob_paths = {Path(row[0]) for row in connection.execute("SELECT path FROM upload_blobs").fetchall()}
        connection.commit()
        connection.close()

        for path in attempt_directories | output_directories:
            if path.exists():
                shutil.rmtree(path)
        for path in bundle_paths | upload_paths:
            path.unlink(missing_ok=True)

        for path in self.attempts_dir.iterdir():
            if path.is_dir() and path.name not in active_attempt_ids and path.stat().st_mtime <= stale_before:
                shutil.rmtree(path)

        for execution_kind, record_keys in (
            ("node_runs", node_run_keys),
            ("actions", action_keys),
        ):
            parent = self.outputs_dir / execution_kind
            if parent.exists():
                for path in parent.iterdir():
                    if path.is_dir() and path.name not in record_keys and path.stat().st_mtime <= stale_before:
                        shutil.rmtree(path)

        for path in self.bundles_dir.iterdir():
            if path.stat().st_mtime > stale_before:
                continue
            if path.name.endswith(".zip.tmp"):
                path.unlink()
            elif path.suffix == ".zip" and path.stem not in action_keys:
                path.unlink()

        for path in self.uploads_dir.iterdir():
            if path.stat().st_mtime > stale_before:
                continue
            if path.suffix == ".tmp" or (path.name.startswith("blob_") and path not in blob_paths):
                path.unlink()

    def save_upload(
        self,
        fileobj: BinaryIO,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        digest = hashlib.sha256()
        temp_path = self.uploads_dir / f"upload_{uuid4().hex}.tmp"

        with temp_path.open("wb") as output:
            chunk = fileobj.read(1024 * 1024)
            while chunk:
                digest.update(chunk)
                output.write(chunk)
                chunk = fileobj.read(1024 * 1024)

        content_hash = digest.hexdigest()
        blob_path = self.uploads_dir / f"blob_{content_hash}"
        touched_at = time.time()

        connection = sqlite3.connect(self.database_path)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            blob_row = connection.execute(
                """
                SELECT path
                FROM upload_blobs
                WHERE content_hash = ?
                """,
                (content_hash,),
            ).fetchone()
            if blob_row is None:
                if blob_path.exists():
                    temp_path.unlink()
                else:
                    os.replace(temp_path, blob_path)
                connection.execute(
                    """
                    INSERT INTO upload_blobs (content_hash, path)
                    VALUES (?, ?)
                    """,
                    (content_hash, str(blob_path)),
                )
            else:
                temp_path.unlink()

            upload_row = connection.execute(
                """
                SELECT upload_key
                FROM uploads
                WHERE content_hash = ?
                  AND filename = ?
                  AND mime_type = ?
                """,
                (content_hash, filename, mime_type),
            ).fetchone()
            if upload_row is None:
                upload_identity = json.dumps(
                    {
                        "content_hash": content_hash,
                        "filename": filename,
                        "mime_type": mime_type,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                upload_key = f"upload_{hashlib.sha256(upload_identity).hexdigest()}"
                connection.execute(
                    """
                    INSERT INTO uploads (
                        upload_key,
                        content_hash,
                        filename,
                        mime_type,
                        created_at,
                        last_touched_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_key,
                        content_hash,
                        filename,
                        mime_type,
                        touched_at,
                        touched_at,
                    ),
                )
            else:
                upload_key = upload_row[0]
                connection.execute(
                    """
                    UPDATE uploads
                    SET last_touched_at = ?
                    WHERE upload_key = ?
                    """,
                    (touched_at, upload_key),
                )

            row = connection.execute(
                """
                SELECT
                    uploads.upload_key,
                    uploads.content_hash,
                    uploads.filename,
                    uploads.mime_type,
                    upload_blobs.path
                FROM uploads
                JOIN upload_blobs USING (content_hash)
                WHERE uploads.upload_key = ?
                """,
                (upload_key,),
            ).fetchone()
        connection.close()

        return {
            "upload_key": row[0],
            "content_hash": row[1],
            "filename": row[2],
            "mime_type": row[3],
            "path": row[4],
        }

    def read_upload(self, upload_key: str) -> dict[str, Any] | None:
        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            """
            SELECT
                uploads.upload_key,
                uploads.content_hash,
                uploads.filename,
                uploads.mime_type,
                upload_blobs.path
            FROM uploads
            JOIN upload_blobs USING (content_hash)
            WHERE uploads.upload_key = ?
            """,
            (upload_key,),
        ).fetchone()
        connection.close()

        if row is None:
            return None

        return {
            "upload_key": row[0],
            "content_hash": row[1],
            "filename": row[2],
            "mime_type": row[3],
            "path": row[4],
        }

    def upload_path(self, upload_key: str) -> Path:
        return Path(self.read_upload(upload_key)["path"])

    def create_session(self, model_id: str | None) -> dict[str, Any]:
        session = {
            "session_id": f"session_{uuid4().hex}",
            "model_id": model_id,
            "revision": 0,
            "active_run_keys": [],
            "actions_by_source": {},
            "status": "active",
            "last_activity_at": time.time(),
            "expired_at": None,
        }

        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            INSERT INTO sessions (
                session_id,
                model_id,
                revision,
                active_run_keys_json,
                actions_by_source_json,
                status,
                last_activity_at,
                expired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["session_id"],
                session["model_id"],
                session["revision"],
                json.dumps(session["active_run_keys"]),
                json.dumps(session["actions_by_source"]),
                session["status"],
                session["last_activity_at"],
                session["expired_at"],
            ),
        )
        connection.commit()
        connection.close()

        return session

    def read_session(self, session_id: str) -> dict[str, Any] | None:
        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            """
            SELECT
                session_id,
                model_id,
                revision,
                active_run_keys_json,
                actions_by_source_json,
                status,
                last_activity_at,
                expired_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        connection.close()

        if row is None:
            return None

        return {
            "session_id": row[0],
            "model_id": row[1],
            "revision": row[2],
            "active_run_keys": json.loads(row[3]),
            "actions_by_source": json.loads(row[4]),
            "status": row[5],
            "last_activity_at": row[6],
            "expired_at": row[7],
        }

    def read_session_restore_snapshot(self, session_id: str) -> dict[str, Any]:
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT
                    session_id,
                    model_id,
                    revision,
                    active_run_keys_json,
                    actions_by_source_json,
                    status,
                    last_activity_at,
                    expired_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                snapshot = {"status": "not_found"}
            elif row[5] == "expired":
                snapshot = {"status": "session_expired"}
            else:
                session = {
                    "session_id": row[0],
                    "model_id": row[1],
                    "revision": row[2],
                    "active_run_keys": json.loads(row[3]),
                    "actions_by_source": json.loads(row[4]),
                    "status": row[5],
                    "last_activity_at": row[6],
                    "expired_at": row[7],
                }
                node_runs_by_key = {}
                for node_run_key in session["active_run_keys"]:
                    record_json = connection.execute(
                        """
                        SELECT record_json
                        FROM node_runs
                        WHERE node_run_key = ?
                        """,
                        (node_run_key,),
                    ).fetchone()[0]
                    node_runs_by_key[node_run_key] = json.loads(record_json)

                actions_by_key = {}
                for action_keys in session["actions_by_source"].values():
                    for action_key in action_keys:
                        record_json = connection.execute(
                            """
                            SELECT record_json
                            FROM actions
                            WHERE action_key = ?
                            """,
                            (action_key,),
                        ).fetchone()[0]
                        actions_by_key[action_key] = json.loads(record_json)

                snapshot = {
                    "status": "active",
                    "session": session,
                    "node_runs_by_key": node_runs_by_key,
                    "actions_by_key": actions_by_key,
                }

            connection.commit()
            return snapshot
        finally:
            connection.close()

    def begin_session_task(
        self,
        request: dict[str, Any],
        creates_session: bool,
        runtime_id: str,
        prepared_reservation_seconds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        created_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")

        session_id = request["session_id"]
        created_session = creates_session and session_id is None
        if creates_session and session_id is None:
            if request["session_revision"] != 0:
                connection.rollback()
                connection.close()
                raise ValueError("A new session must start at revision 0")

            session_id = f"session_{uuid4().hex}"
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    model_id,
                    revision,
                    active_run_keys_json,
                    actions_by_source_json,
                    status,
                    last_activity_at,
                    expired_at
                ) VALUES (?, ?, 0, '[]', '{}', 'active', ?, NULL)
                """,
                (session_id, request["model_id"], created_at),
            )
        elif session_id is None:
            connection.rollback()
            connection.close()
            raise ValueError("The operation requires session_id")

        session_row = connection.execute(
            """
            SELECT
                session_id,
                model_id,
                revision,
                active_run_keys_json,
                actions_by_source_json,
                status,
                last_activity_at,
                expired_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if session_row is None:
            connection.rollback()
            connection.close()
            raise ValueError(f"Session not found: {session_id}")
        if session_row[5] != "active":
            connection.rollback()
            connection.close()
            raise SessionExpiredError(f"Session expired: {session_id}")
        if request["session_revision"] != session_row[2]:
            connection.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Session revision conflict: {session_id}")
        if request["model_id"] != session_row[1]:
            connection.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(
                f"Session {session_id} belongs to model {session_row[1]}",
            )

        connection.execute(
            """
            UPDATE session_tasks
            SET state = 'failed', updated_at = ?
            WHERE session_id = ?
              AND (
                    (state = 'prepared' AND reservation_expires_at <= ?)
                 OR (state = 'queued' AND queue_lease_expires_at <= ?)
              )
            """,
            (created_at, session_id, created_at, created_at),
        )
        active_task = connection.execute(
            """
            SELECT task_id
            FROM session_tasks
            WHERE session_id = ?
              AND state IN ('prepared', 'queued', 'running', 'committing')
            """,
            (session_id,),
        ).fetchone()
        if active_task is not None:
            connection.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Session already has an active task: {session_id}")

        for upload_key in request["input_upload_keys"]:
            upload_row = connection.execute(
                "SELECT upload_key FROM uploads WHERE upload_key = ?",
                (upload_key,),
            ).fetchone()
            if upload_row is None:
                if created_session:
                    connection.rollback()
                else:
                    connection.execute(
                        "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                        (created_at, session_id),
                    )
                    connection.commit()
                connection.close()
                raise ValueError(f"Input upload record not found: {upload_key}")

        for upload_key in request["input_upload_keys"]:
            connection.execute(
                """
                INSERT OR IGNORE INTO session_uploads (session_id, upload_key)
                VALUES (?, ?)
                """,
                (session_id, upload_key),
            )

        stored_request = {
            **request,
            "session_id": session_id,
        }
        task_id = f"task_{uuid4().hex}"
        reservation_expires_at = created_at + prepared_reservation_seconds
        connection.execute(
            """
            INSERT INTO session_tasks (
                task_id,
                session_id,
                state,
                request_json,
                runtime_id,
                created_at,
                updated_at,
                reservation_expires_at,
                queue_event_id,
                queue_lease_expires_at,
                attempt_id,
                work_dir
            ) VALUES (?, ?, 'prepared', ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                task_id,
                session_id,
                json.dumps(stored_request, sort_keys=True),
                runtime_id,
                created_at,
                created_at,
                reservation_expires_at,
            ),
        )
        connection.execute(
            """
            UPDATE sessions
            SET last_activity_at = ?
            WHERE session_id = ?
            """,
            (created_at, session_id),
        )
        connection.commit()
        connection.close()

        task = {
            "task_id": task_id,
            "session_id": session_id,
            "state": "prepared",
            "request": stored_request,
            "runtime_id": runtime_id,
            "created_at": created_at,
            "updated_at": created_at,
            "reservation_expires_at": reservation_expires_at,
            "queue_event_id": None,
            "queue_lease_expires_at": None,
            "attempt_id": None,
            "work_dir": None,
        }
        session = {
            "session_id": session_row[0],
            "model_id": session_row[1],
            "revision": session_row[2],
            "active_run_keys": json.loads(session_row[3]),
            "actions_by_source": json.loads(session_row[4]),
            "status": session_row[5],
            "last_activity_at": created_at,
            "expired_at": session_row[7],
        }
        return task, session

    def mark_execution_queued(
        self,
        task_id: str,
        event_id: str,
        queue_lease_seconds: int,
    ) -> None:
        queued_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT session_tasks.state, session_tasks.reservation_expires_at, sessions.status
            FROM session_tasks
            JOIN sessions USING (session_id)
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            connection.close()
            raise ValueError(f"Task not found: {task_id}")
        if row[0] != "prepared":
            connection.rollback()
            connection.close()
            raise ValueError(f"Task is not prepared: {task_id}")
        if row[1] <= queued_at:
            connection.execute(
                "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                (queued_at, task_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Task reservation expired: {task_id}")
        if row[2] != "active":
            connection.rollback()
            connection.close()
            raise SessionExpiredError(f"Task session expired: {task_id}")

        connection.execute(
            """
            UPDATE session_tasks
            SET
                state = 'queued',
                updated_at = ?,
                queue_event_id = ?,
                queue_lease_expires_at = ?
            WHERE task_id = ?
            """,
            (
                queued_at,
                event_id,
                queued_at + queue_lease_seconds,
                task_id,
            ),
        )
        connection.commit()
        connection.close()

    def renew_execution_queue(
        self,
        task_id: str,
        event_id: str,
        queue_lease_seconds: int,
    ) -> bool:
        renewed_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT
                session_tasks.state,
                session_tasks.queue_event_id,
                session_tasks.queue_lease_expires_at,
                sessions.status
            FROM session_tasks
            JOIN sessions USING (session_id)
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            connection.close()
            raise ValueError(f"Task not found: {task_id}")
        if row[0] in ("running", "committing") and row[1] == event_id:
            connection.rollback()
            connection.close()
            return False
        if row[0] != "queued" or row[1] != event_id:
            connection.rollback()
            connection.close()
            raise ValueError(f"Queued task does not match event: {task_id}")
        if row[2] <= renewed_at:
            connection.execute(
                "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                (renewed_at, task_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Task queue lease expired: {task_id}")
        if row[3] != "active":
            connection.rollback()
            connection.close()
            raise SessionExpiredError(f"Task session expired: {task_id}")

        connection.execute(
            """
            UPDATE session_tasks
            SET updated_at = ?, queue_lease_expires_at = ?
            WHERE task_id = ?
            """,
            (renewed_at, renewed_at + queue_lease_seconds, task_id),
        )
        connection.commit()
        connection.close()
        return True

    def start_queued_execution(self, task_id: str) -> dict[str, Any]:
        while True:
            started_at = time.time()
            connection = sqlite3.connect(self.database_path)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT
                    session_tasks.session_id,
                    session_tasks.state,
                    session_tasks.request_json,
                    session_tasks.reservation_expires_at,
                    session_tasks.queue_lease_expires_at,
                    sessions.status,
                    sessions.revision
                FROM session_tasks
                JOIN sessions USING (session_id)
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                connection.close()
                raise ValueError(f"Task not found: {task_id}")
            if row[1] == "prepared" and row[3] > started_at:
                connection.rollback()
                connection.close()
                time.sleep(0.01)
                continue
            if row[1] == "prepared":
                connection.execute(
                    "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                    (started_at, task_id),
                )
                connection.commit()
                connection.close()
                raise ValueError(f"Task reservation expired: {task_id}")
            if row[1] != "queued":
                connection.rollback()
                connection.close()
                raise ValueError(f"Task is not queued: {task_id}")
            if row[4] <= started_at:
                connection.execute(
                    "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                    (started_at, task_id),
                )
                connection.commit()
                connection.close()
                raise ValueError(f"Task queue lease expired: {task_id}")
            if row[5] != "active":
                connection.rollback()
                connection.close()
                raise SessionExpiredError(f"Task session expired: {task_id}")

            request = json.loads(row[2])
            if request["session_revision"] != row[6]:
                connection.rollback()
                connection.close()
                raise ValueError(f"Session revision conflict: {row[0]}")

            connection.execute(
                """
                UPDATE session_tasks
                SET state = 'running', updated_at = ?
                WHERE task_id = ?
                """,
                (started_at, task_id),
            )
            connection.commit()
            connection.close()
            return {
                "task_id": task_id,
                "session_id": row[0],
                "state": "running",
                "request": request,
            }

    def start_inline_execution(self, task_id: str) -> None:
        started_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT session_tasks.state, session_tasks.reservation_expires_at, sessions.status
            FROM session_tasks
            JOIN sessions USING (session_id)
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None or row[0] != "prepared":
            connection.rollback()
            connection.close()
            raise ValueError(f"Task is not prepared: {task_id}")
        if row[1] <= started_at:
            connection.execute(
                "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                (started_at, task_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Task reservation expired: {task_id}")
        if row[2] != "active":
            connection.rollback()
            connection.close()
            raise SessionExpiredError(f"Task session expired: {task_id}")

        connection.execute(
            "UPDATE session_tasks SET state = 'running', updated_at = ? WHERE task_id = ?",
            (started_at, task_id),
        )
        connection.commit()
        connection.close()

    def create_execution_attempt(self, task_id: str) -> tuple[str, Path]:
        created_at = time.time()
        attempt_id = f"attempt_{uuid4().hex}"
        work_dir = self.attempts_dir / attempt_id
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT state, attempt_id, work_dir
            FROM session_tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None or row[0] != "running":
            connection.rollback()
            connection.close()
            raise ValueError(f"Task is not running: {task_id}")
        if row[1] is not None or row[2] is not None:
            connection.rollback()
            connection.close()
            raise ValueError(f"Task already has an attempt: {task_id}")

        work_dir.mkdir()
        connection.execute(
            """
            UPDATE session_tasks
            SET attempt_id = ?, work_dir = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (attempt_id, str(work_dir), created_at, task_id),
        )
        connection.commit()
        connection.close()
        return attempt_id, work_dir

    def mark_task_committing(self, task_id: str) -> None:
        committing_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT session_tasks.state, session_tasks.reservation_expires_at, sessions.status
            FROM session_tasks
            JOIN sessions USING (session_id)
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None or row[0] not in ("prepared", "running"):
            connection.rollback()
            connection.close()
            raise ValueError(f"Task cannot enter committing: {task_id}")
        if row[0] == "prepared" and row[1] <= committing_at:
            connection.execute(
                "UPDATE session_tasks SET state = 'failed', updated_at = ? WHERE task_id = ?",
                (committing_at, task_id),
            )
            connection.commit()
            connection.close()
            raise ValueError(f"Task reservation expired: {task_id}")
        if row[2] != "active":
            connection.rollback()
            connection.close()
            raise SessionExpiredError(f"Task session expired: {task_id}")

        connection.execute(
            "UPDATE session_tasks SET state = 'committing', updated_at = ? WHERE task_id = ?",
            (committing_at, task_id),
        )
        connection.commit()
        connection.close()

    def fail_execution(self, task_id: str) -> None:
        failed_at = time.time()
        connection = sqlite3.connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        task_row = connection.execute(
            "SELECT session_id FROM session_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        connection.execute(
            """
            UPDATE session_tasks
            SET state = 'failed', updated_at = ?
            WHERE task_id = ?
              AND state IN ('prepared', 'queued', 'running', 'committing')
            """,
            (failed_at, task_id),
        )
        if task_row is not None:
            connection.execute(
                """
                UPDATE sessions
                SET last_activity_at = ?
                WHERE session_id = ?
                  AND status = 'active'
                """,
                (failed_at, task_row[0]),
            )
        connection.commit()
        connection.close()

    def read_node_run(self, node_run_key: str) -> dict[str, Any] | None:
        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            """
            SELECT record_json
            FROM node_runs
            WHERE node_run_key = ?
            """,
            (node_run_key,),
        ).fetchone()
        connection.close()

        if row is None:
            return None

        return json.loads(row[0])

    def read_action(self, action_key: str) -> dict[str, Any] | None:
        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            """
            SELECT record_json
            FROM actions
            WHERE action_key = ?
            """,
            (action_key,),
        ).fetchone()
        connection.close()

        if row is None:
            return None

        return json.loads(row[0])

    def commit_execution(
        self,
        task_id: str,
        execution_kind: str,
        key: str,
        record: dict[str, Any],
        session: dict[str, Any],
        expected_revision: int,
        actual_work_dir: Path | None,
    ) -> tuple[dict[str, Any], int, bool]:
        final_parent = self.outputs_dir / ("node_runs" if execution_kind == "node_run" else "actions")
        final_parent.mkdir(parents=True, exist_ok=True)
        final_dir = final_parent / key
        published = False
        cache_hit = False
        connection = sqlite3.connect(self.database_path)

        try:
            connection.execute("BEGIN IMMEDIATE")
            task_row = connection.execute(
                """
                SELECT session_id, state, work_dir
                FROM session_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if task_row is None or task_row[1] != "committing":
                raise ValueError(f"Task is not committing: {task_id}")
            if task_row[0] != session["session_id"]:
                raise ValueError(f"Task session mismatch: {task_id}")
            if actual_work_dir is None:
                if task_row[2] is not None:
                    raise ValueError(f"Task work_dir mismatch: {task_id}")
            elif task_row[2] is None or Path(task_row[2]).resolve() != actual_work_dir.resolve():
                raise ValueError(f"Task work_dir mismatch: {task_id}")

            session_row = connection.execute(
                """
                SELECT status, revision
                FROM sessions
                WHERE session_id = ?
                """,
                (session["session_id"],),
            ).fetchone()
            if session_row is None:
                raise ValueError(f"Session not found: {session['session_id']}")
            if session_row[0] != "active":
                raise SessionExpiredError(f"Session expired: {session['session_id']}")
            if session_row[1] != expected_revision:
                raise ValueError(
                    f"Session revision conflict: {session['session_id']}",
                )

            if execution_kind == "node_run":
                record_row = connection.execute(
                    """
                    SELECT record_json
                    FROM node_runs
                    WHERE node_run_key = ?
                    """,
                    (key,),
                ).fetchone()
            else:
                record_row = connection.execute(
                    """
                    SELECT record_json
                    FROM actions
                    WHERE action_key = ?
                    """,
                    (key,),
                ).fetchone()

            if record_row is None:
                if actual_work_dir is None:
                    raise ValueError(f"Execution has no attempt directory: {task_id}")
                if final_dir.exists():
                    raise ValueError(f"Final output directory already exists: {key}")

                os.replace(actual_work_dir, final_dir)
                published = True
                if execution_kind == "node_run":
                    connection.execute(
                        """
                        INSERT INTO node_runs (node_run_key, record_json)
                        VALUES (?, ?)
                        """,
                        (key, json.dumps(record, sort_keys=True)),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO actions (action_key, record_json)
                        VALUES (?, ?)
                        """,
                        (key, json.dumps(record, sort_keys=True)),
                    )
                actual_record = record
            else:
                cache_hit = True
                actual_record = json.loads(record_row[0])

            if execution_kind == "node_run":
                input_upload_keys = actual_record["input_upload_keys"]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_node_runs (
                        session_id,
                        node_run_key,
                        input_upload_keys_json
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        session["session_id"],
                        key,
                        json.dumps(input_upload_keys),
                    ),
                )
                for upload_key in input_upload_keys:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO session_uploads (session_id, upload_key)
                        VALUES (?, ?)
                        """,
                        (session["session_id"], upload_key),
                    )
            else:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO session_actions (session_id, action_key)
                    VALUES (?, ?)
                    """,
                    (session["session_id"], key),
                )

            update = connection.execute(
                """
                UPDATE sessions
                SET
                    active_run_keys_json = ?,
                    actions_by_source_json = ?,
                    revision = revision + 1,
                    last_activity_at = ?
                WHERE session_id = ?
                  AND revision = ?
                  AND status = 'active'
                """,
                (
                    json.dumps(session["active_run_keys"]),
                    json.dumps(session["actions_by_source"]),
                    time.time(),
                    session["session_id"],
                    expected_revision,
                ),
            )

            if update.rowcount != 1:
                raise ValueError(
                    f"Session revision conflict: {session['session_id']}",
                )

            task_delete = connection.execute(
                """
                DELETE FROM session_tasks
                WHERE task_id = ?
                  AND state = 'committing'
                """,
                (task_id,),
            )
            if task_delete.rowcount != 1:
                raise ValueError(f"Task commit conflict: {task_id}")

            revision_row = connection.execute(
                """
                SELECT revision
                FROM sessions
                WHERE session_id = ?
                """,
                (session["session_id"],),
            ).fetchone()
            connection.commit()
            connection.close()
        except Exception:
            connection.rollback()
            connection.close()
            if published:
                verification = sqlite3.connect(self.database_path)
                if execution_kind == "node_run":
                    persisted = verification.execute(
                        "SELECT 1 FROM node_runs WHERE node_run_key = ?",
                        (key,),
                    ).fetchone()
                else:
                    persisted = verification.execute(
                        "SELECT 1 FROM actions WHERE action_key = ?",
                        (key,),
                    ).fetchone()
                verification.close()

                if persisted is None and final_dir.exists():
                    if actual_work_dir is not None and not actual_work_dir.exists():
                        os.replace(final_dir, actual_work_dir)
                    else:
                        shutil.rmtree(final_dir)
            raise

        if cache_hit and actual_work_dir is not None and actual_work_dir.exists():
            shutil.rmtree(actual_work_dir)

        return actual_record, revision_row[0], cache_hit
