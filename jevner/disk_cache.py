"""Optional persistent answer cache (sqlite) for JevNER instances.

Opt in with ``JevNER(disk_cache_path=...)``: gateway answers survive
process restarts, so repeated runs, benchmark iterations, and RAG
re-queries cost zero requests. The in-process LRU caches always sit
in front (a disk hit promotes the value into the LRU); the disk layer
only matters across processes.

Keys are the same tuples the LRU caches use, prefixed with the model
name — answers from one model are never reused for another, and the
typing keys already embed the full type fingerprint (labels,
descriptions, examples, counter-examples), so a schema change
invalidates itself automatically.

Values are JSON documents; callers encode/decode their own payloads
(tuples, dataclasses, and enums are encoded at the call sites), so
this module stays a dumb string-keyed store. All access is serialized
behind one lock; jevner only ever touches it from the private event
loop thread, but the lock also covers the constructing thread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _key_str(key: tuple[Any, ...]) -> str:
    """Serialize a heterogeneous key tuple to a stable string."""
    return json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)


class DiskCache:
    """A tiny sqlite key-value store for cache tuples."""

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the cache database.

        Args:
            path: Database file path; parent directories are created.

        Raises:
            sqlite3.Error: The database cannot be opened or initialized.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(target, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()

    def read(self, key: tuple[Any, ...]) -> Any:
        """Return the cached JSON value for a key tuple, or None.

        Args:
            key: The cache key tuple (any JSON-encodable elements).

        Returns:
            The decoded value, or None on a miss.
        """
        with self._lock:
            row = self._db.execute("SELECT v FROM answers WHERE k = ?", (_key_str(key),)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def write(self, key: tuple[Any, ...], value: Any) -> None:
        """Store a JSON value under a key tuple (insert or replace).

        Args:
            key: The cache key tuple.
            value: A JSON-serializable value.
        """
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO answers (k, v) VALUES (?, ?)",
                (_key_str(key), json.dumps(value, ensure_ascii=False)),
            )
            self._db.commit()

    def close(self) -> None:
        """Close the database connection. Idempotent."""
        with self._lock:
            self._db.close()
