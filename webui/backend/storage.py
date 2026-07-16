import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4


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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_key TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                path TEXT NOT NULL
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model_id TEXT,
                revision INTEGER NOT NULL,
                active_run_keys_json TEXT NOT NULL,
                actions_by_source_json TEXT NOT NULL
            )
            """,
        )
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
        connection.commit()
        connection.close()

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
        upload_key = f"upload_{content_hash}"
        final_path = self.upload_path(upload_key)

        if final_path.exists():
            temp_path.unlink()
        else:
            os.replace(temp_path, final_path)

        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            INSERT OR IGNORE INTO uploads (
                upload_key,
                content_hash,
                filename,
                mime_type,
                path
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                upload_key,
                content_hash,
                filename,
                mime_type,
                str(final_path),
            ),
        )
        connection.commit()
        row = connection.execute(
            """
            SELECT upload_key, content_hash, filename, mime_type, path
            FROM uploads
            WHERE upload_key = ?
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
            SELECT upload_key, content_hash, filename, mime_type, path
            FROM uploads
            WHERE upload_key = ?
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
        return self.uploads_dir / upload_key

    def create_session(self, model_id: str | None) -> dict[str, Any]:
        session = {
            "session_id": f"session_{uuid4().hex}",
            "model_id": model_id,
            "revision": 0,
            "active_run_keys": [],
            "actions_by_source": {},
        }

        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            INSERT INTO sessions (
                session_id,
                model_id,
                revision,
                active_run_keys_json,
                actions_by_source_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                session["session_id"],
                session["model_id"],
                session["revision"],
                json.dumps(session["active_run_keys"]),
                json.dumps(session["actions_by_source"]),
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
                actions_by_source_json
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
        }

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

    def execution_work_dir(self, request_id: str) -> Path:
        work_dir = self.attempts_dir / request_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def output_path(self, kind: str, key: str, role: str) -> Path:
        if kind == "node_run":
            return self.outputs_dir / "node_runs" / key / role
        return self.outputs_dir / "actions" / key / role

    def commit_execution(
        self,
        execution_kind: str,
        key: str,
        record: dict[str, Any],
        session: dict[str, Any],
        expected_revision: int,
    ) -> tuple[dict[str, Any], int]:
        connection = sqlite3.connect(self.database_path)

        with connection:
            connection.execute("BEGIN IMMEDIATE")

            if execution_kind == "node_run":
                connection.execute(
                    """
                    INSERT OR IGNORE INTO node_runs (node_run_key, record_json)
                    VALUES (?, ?)
                    """,
                    (key, json.dumps(record, sort_keys=True)),
                )
            else:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO actions (action_key, record_json)
                    VALUES (?, ?)
                    """,
                    (key, json.dumps(record, sort_keys=True)),
                )

            update = connection.execute(
                """
                UPDATE sessions
                SET
                    active_run_keys_json = ?,
                    actions_by_source_json = ?,
                    revision = revision + 1
                WHERE session_id = ?
                  AND revision = ?
                """,
                (
                    json.dumps(session["active_run_keys"]),
                    json.dumps(session["actions_by_source"]),
                    session["session_id"],
                    expected_revision,
                ),
            )

            if update.rowcount != 1:
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

            revision_row = connection.execute(
                """
                SELECT revision
                FROM sessions
                WHERE session_id = ?
                """,
                (session["session_id"],),
            ).fetchone()

        connection.close()
        return json.loads(record_row[0]), revision_row[0]
