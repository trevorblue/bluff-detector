"""A model registry with movable aliases (@champion, @challenger).

Works offline as a JSON file plus versioned artefact folders, so CI and the API need no
server. When tracking.mlflow is true, every registration is mirrored to the MLflow Model
Registry with the same aliases. Promotion and rollback are one pointer move.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from bluff.utils import ensure_dir, read_json, write_json


class LocalRegistry:
    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)
        self.index = self.root / "registry.json"

    def _load(self) -> dict[str, Any]:
        return read_json(self.index) if self.index.exists() else {"models": {}}

    def _save(self, data: dict[str, Any]) -> None:
        write_json(self.index, data)

    def register(self, name: str, artefact_dir: Path, info: dict[str, Any]) -> int:
        data = self._load()
        model = data["models"].setdefault(name, {"versions": [], "aliases": {}})
        version = len(model["versions"]) + 1
        dest = self.root / name / f"v{version}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(artefact_dir, dest)
        model["versions"].append(
            {"version": version, "path": str(dest), "registered_at": time.time(), **info}
        )
        self._save(data)
        return version

    def set_alias(self, name: str, alias: str, version: int) -> None:
        data = self._load()
        model = data["models"][name]
        if not any(v["version"] == version for v in model["versions"]):
            raise ValueError(f"{name} has no version {version}")
        model["aliases"][alias] = version
        self._save(data)

    def get(self, name: str, alias: str) -> dict[str, Any] | None:
        data = self._load()
        model = data["models"].get(name)
        if not model or alias not in model["aliases"]:
            return None
        v = model["aliases"][alias]
        return next(x for x in model["versions"] if x["version"] == v)

    def versions(self, name: str) -> list[dict[str, Any]]:
        return self._load()["models"].get(name, {}).get("versions", [])
