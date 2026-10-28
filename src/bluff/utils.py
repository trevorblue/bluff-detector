"""Small shared helpers: seeding, hashing, JSON/JSONL I/O.

CRISP-DM: shared by every phase.
Used by: almost every module.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy. Torch is seeded by the HF backend when it loads."""
    random.seed(seed)
    np.random.seed(seed)


def stable_hash(text: str, mod: int) -> int:
    """Deterministic hash (Python's hash() is salted per process)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % mod


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash: turns 'the October data' into an identity a run can be rebuilt from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
