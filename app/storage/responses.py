import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResponseStore:
    def __init__(self, root: Path):
        self.root = root

    def append_session_json(self, relative_dir: str, session_id: str, data: dict[str, Any]) -> Path:
        path = self._session_file(relative_dir, session_id)
        existing = self._read_json_list(path)
        existing.append(data)
        self._write_json(path, existing)
        return path

    def write_session_json(self, relative_dir: str, session_id: str, data: dict[str, Any]) -> Path:
        path = self._session_file(relative_dir, session_id)
        self._write_json(path, data)
        return path

    def append_jsonl(self, relative_path: str, data: dict[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(data)
        data["server_timestamp"] = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
        return path

    def read_all_json_entries(self, relative_dir: str) -> list[Any]:
        directory = self.root / relative_dir
        if not directory.exists():
            return []

        entries: list[Any] = []
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)
        return entries

    def _session_file(self, relative_dir: str, session_id: str) -> Path:
        safe_session = self._safe_name(session_id or "anonymous")
        return self.root / relative_dir / f"{safe_session}.json"

    def _read_json_list(self, path: Path) -> list[Any]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else [data]

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        allowed = []
        for char in value:
            if char.isalnum() or char in ("-", "_"):
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed) or "anonymous"

