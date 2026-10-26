"""Sharded persistent state store for the full POI-road environment.

SQLite is used here because it is part of Python, transactional, and can hold
millions of keyed states without requiring one giant in-memory dictionary.
Each shard is an independent database; a timestep can process shards in batches
and commit every batch, so a new observation changes the persistent state even
when the whole city does not fit in RAM.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator


class StateStore:
    def __init__(self, root: Path, shards: int = 64) -> None:
        if shards < 1:
            raise ValueError("shards must be positive")
        self.root = Path(root)
        self.shards = shards
        self.root.mkdir(parents=True, exist_ok=True)
        self._connections: dict[int, sqlite3.Connection] = {}
        for shard in range(shards):
            connection = sqlite3.connect(self._path(shard), timeout=60)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS poi_state (
                    object_id INTEGER PRIMARY KEY, state_json TEXT NOT NULL, updated_slot INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS road_state (
                    object_id INTEGER PRIMARY KEY, state_json TEXT NOT NULL, updated_slot INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transit (
                    arrival_slot INTEGER NOT NULL, request_id TEXT PRIMARY KEY, state_json TEXT NOT NULL
                );
                """
            )
            connection.commit()
            self._connections[shard] = connection

    def _path(self, shard: int) -> Path:
        return self.root / f"state_{shard:03d}.sqlite3"

    def _shard(self, object_id: int) -> int:
        return int(object_id) % self.shards

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _upsert(self, table: str, rows: Iterable[tuple[int, dict, int]]) -> int:
        if table not in ("poi_state", "road_state"):
            raise ValueError("invalid state table")
        grouped: dict[int, list[tuple[int, str, int]]] = {}
        for object_id, state, slot in rows:
            grouped.setdefault(self._shard(object_id), []).append(
                (int(object_id), json.dumps(state, separators=(",", ":")), int(slot))
            )
        count = 0
        for shard, batch in grouped.items():
            self._connections[shard].executemany(
                f"INSERT INTO {table}(object_id,state_json,updated_slot) VALUES(?,?,?) "
                "ON CONFLICT(object_id) DO UPDATE SET state_json=excluded.state_json, updated_slot=excluded.updated_slot",
                batch,
            )
            self._connections[shard].commit()
            count += len(batch)
        return count

    def upsert_pois(self, rows: Iterable[tuple[int, dict, int]]) -> int:
        return self._upsert("poi_state", rows)

    def upsert_roads(self, rows: Iterable[tuple[int, dict, int]]) -> int:
        return self._upsert("road_state", rows)

    def get(self, kind: str, object_id: int) -> tuple[dict, int] | None:
        table = "poi_state" if kind == "poi" else "road_state" if kind == "road" else None
        if table is None:
            raise ValueError("kind must be poi or road")
        row = self._connections[self._shard(object_id)].execute(
            f"SELECT state_json,updated_slot FROM {table} WHERE object_id=?", (object_id,)
        ).fetchone()
        return None if row is None else (json.loads(row[0]), int(row[1]))

    def snapshot(self, kind: str, object_ids: Iterable[int]) -> dict[int, tuple[dict, int]]:
        """Load a bounded object snapshot for local route/MPC evaluation."""
        result = {}
        for object_id in object_ids:
            value = self.get(kind, int(object_id))
            if value is not None:
                result[int(object_id)] = value
        return result

    def update_observation(self, kind: str, object_id: int, state: dict, slot: int) -> None:
        """Atomically overwrite one object with a newly observed state."""
        if kind == "poi":
            self.upsert_pois(((object_id, state, slot),))
        elif kind == "road":
            self.upsert_roads(((object_id, state, slot),))
        else:
            raise ValueError("kind must be poi or road")

    def iter_batches(self, kind: str, batch_size: int = 4096) -> Iterator[list[tuple[int, dict, int]]]:
        table = "poi_state" if kind == "poi" else "road_state" if kind == "road" else None
        if table is None or batch_size < 1:
            raise ValueError("kind must be poi or road and batch_size positive")
        for shard in range(self.shards):
            cursor = self._connections[shard].execute(
                f"SELECT object_id,state_json,updated_slot FROM {table} ORDER BY object_id"
            )
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [(int(object_id), json.loads(state), int(slot)) for object_id, state, slot in rows]

    def add_in_transit(self, request_id: str, arrival_slot: int, state: dict) -> None:
        shard = self._shard(abs(hash(request_id)))
        self._connections[shard].execute(
            "INSERT INTO transit(arrival_slot,request_id,state_json) VALUES(?,?,?) "
            "ON CONFLICT(request_id) DO UPDATE SET arrival_slot=excluded.arrival_slot,state_json=excluded.state_json",
            (int(arrival_slot), request_id, json.dumps(state, separators=(",", ":"))),
        )
        self._connections[shard].commit()

    def pop_arrivals(self, arrival_slot: int) -> list[dict]:
        result = []
        for shard, connection in self._connections.items():
            rows = connection.execute(
                "SELECT request_id,state_json FROM transit WHERE arrival_slot=?", (int(arrival_slot),)
            ).fetchall()
            for request_id, state_json in rows:
                result.append({"request_id": request_id, **json.loads(state_json)})
            connection.executemany("DELETE FROM transit WHERE arrival_slot=?", [(int(arrival_slot),)])
            connection.commit()
        return result

    def counts(self) -> dict[str, int]:
        result = {}
        for table, key in (("poi_state", "pois"), ("road_state", "roads"), ("transit", "in_transit")):
            result[key] = sum(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                               for connection in self._connections.values())
        return result
