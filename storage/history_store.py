from datetime import datetime, timezone
from pathlib import Path
import json


class HistoryStore:
    def __init__(self, output_dir=Path("outputs")):
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / "chat_history.jsonl"
        self.screenshot_dir = self.output_dir / "screenshots"

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, role, content, meta=None):
        if not content:
            return
        self._ensure_output_dir()
        record = {
            "role": role,
            "content": content,
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if meta:
            record["meta"] = meta

        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self):
        if not self.history_path.exists():
            return []

        history = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = item.get("role")
            content = item.get("content")
            if role and content:
                history.append({"role": role, "content": content})
        return history

    def clear(self, delete_screenshots=True):
        if self.history_path.exists():
            self.history_path.write_text("", encoding="utf-8")

        if delete_screenshots and self.screenshot_dir.exists():
            for item in self.screenshot_dir.glob("*.*"):
                try:
                    item.unlink()
                except OSError:
                    print(f"[warning] 无法删除截图: {item}")
