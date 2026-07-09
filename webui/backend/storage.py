import hashlib
import json
import shutil
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4


class Storage:
    def __init__(self, root: Path):
        self.root = root
        self.uploads_dir = root / "uploads"
        self.node_runs_dir = root / "node_runs"
        self.actions_dir = root / "actions"
        self.sessions_dir = root / "sessions"
        self.tmp_dir = root / "tmp"

        for path in (
            self.uploads_dir,
            self.node_runs_dir,
            self.actions_dir,
            self.sessions_dir,
            self.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        fileobj: BinaryIO,
        filename: str,
        mime_type: str,
    ) -> dict:
        digest = hashlib.sha256()
        temp_path = self.tmp_dir / f"upload_{uuid4().hex}"

        with temp_path.open("wb") as output:
            chunk = fileobj.read(1024 * 1024)
            while chunk:
                digest.update(chunk)
                output.write(chunk)
                chunk = fileobj.read(1024 * 1024)

        content_hash = digest.hexdigest()
        upload_key = f"upload_{content_hash}"
        upload_dir = self.uploads_dir / upload_key
        upload_dir.mkdir(parents=True, exist_ok=True)

        content_path = upload_dir / "content"
        if content_path.exists():
            temp_path.unlink()
        else:
            shutil.move(str(temp_path), str(content_path))

        record = {
            "upload_key": upload_key,
            "content_hash": content_hash,
            "filename": filename,
            "mime_type": mime_type,
            "path": str(content_path),
        }
        with (upload_dir / "record.json").open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)

        return record

    def upload_path(self, upload_key: str) -> Path:
        return self.uploads_dir / upload_key / "content"

    def read_upload(self, upload_key: str) -> dict | None:
        record_path = self.uploads_dir / upload_key / "record.json"
        if not record_path.exists():
            return None
        with record_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    def execution_work_dir(self, execution_key: str) -> Path:
        work_dir = self.tmp_dir / execution_key
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        return work_dir

    def node_run_dir(self, node_run_key: str) -> Path:
        return self.node_runs_dir / node_run_key

    def node_run_output_path(self, node_run_key: str, role: str) -> Path:
        return self.node_run_dir(node_run_key) / "outputs" / role

    def write_node_run(self, node_run_key: str, record: dict) -> None:
        run_dir = self.node_run_dir(node_run_key)
        (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        with (run_dir / "record.json").open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)

    def read_node_run(self, node_run_key: str) -> dict | None:
        record_path = self.node_run_dir(node_run_key) / "record.json"
        if not record_path.exists():
            return None
        with record_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    def action_dir(self, action_key: str) -> Path:
        return self.actions_dir / action_key

    def action_output_path(self, action_key: str, role: str) -> Path:
        return self.action_dir(action_key) / "outputs" / role

    def write_action(self, action_key: str, record: dict) -> None:
        action_dir = self.action_dir(action_key)
        (action_dir / "outputs").mkdir(parents=True, exist_ok=True)
        with (action_dir / "record.json").open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)

    def read_action(self, action_key: str) -> dict | None:
        record_path = self.action_dir(action_key) / "record.json"
        if not record_path.exists():
            return None
        with record_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    def write_session(self, session_id: str, record: dict) -> None:
        session_path = self.sessions_dir / f"{session_id}.json"
        with session_path.open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)

    def read_session(self, session_id: str) -> dict | None:
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return None
        with session_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
