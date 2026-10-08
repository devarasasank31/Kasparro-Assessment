"""Lightweight JSON file cache used to avoid repeat network calls.

Two namespaces are used in practice:

- ``github`` - profile + repository payloads (TTL 6h)
- ``llm``    - structured model responses (TTL 24h)

The cache is best-effort: a corrupt or unwritable cache file is ignored rather
than failing the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .utils import get_logger

log = get_logger("cache")

DEFAULT_TTL: dict[str, float] = {
    "github": 6 * 3600.0,
    "llm": 24 * 3600.0,
}


class RunCache:
    """Namespace-aware, file-backed cache with per-entry timestamps."""

    def __init__(self, directory: Path | None = None, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = enabled
        self._memory: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()
        self.stats: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    def _bucket(self, namespace: str) -> dict[str, Any]:
        if namespace not in self._memory:
            self._memory[namespace] = self._load(namespace)
        return self._memory[namespace]

    def _path(self, namespace: str) -> Path | None:
        if not self.directory:
            return None
        return self.directory / f"{namespace}.json"

    def _load(self, namespace: str) -> dict[str, Any]:
        path = self._path(namespace)
        if not self.enabled or path is None or not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable cache file %s (%s)", path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def _counter(self, namespace: str) -> dict[str, int]:
        return self.stats.setdefault(namespace, {"hits": 0, "misses": 0, "writes": 0})

    # ------------------------------------------------------------------
    def get(self, namespace: str, key: str) -> Any | None:
        """Return a cached value or ``None`` on miss / expiry."""
        if not self.enabled:
            return None
        counter = self._counter(namespace)
        entry = self._bucket(namespace).get(key)
        if not isinstance(entry, dict) or "value" not in entry:
            counter["misses"] += 1
            return None
        ttl = DEFAULT_TTL.get(namespace)
        stored_at = entry.get("stored_at", 0)
        if ttl is not None and (time.time() - float(stored_at)) > ttl:
            counter["misses"] += 1
            return None
        counter["hits"] += 1
        return entry["value"]

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._counter(namespace)["writes"] += 1
        self._bucket(namespace)[key] = {"stored_at": time.time(), "value": value}
        self._dirty.add(namespace)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Persist dirty namespaces. Never raises."""
        if not self.enabled:
            return
        for namespace in list(self._dirty):
            path = self._path(namespace)
            if path is None:
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(self._memory.get(namespace, {}), ensure_ascii=False, indent=0),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("Could not write cache file %s (%s)", path, exc)
            finally:
                self._dirty.discard(namespace)

    def summary(self) -> dict[str, dict[str, int]]:
        return dict(self.stats)
