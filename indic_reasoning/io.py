"""JSON and JSONL artifact I/O."""
import json
from pathlib import Path

def read_jsonl(path):
    """Read physical JSONL records; Unicode separators inside strings are text."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
