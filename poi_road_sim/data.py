"""Read compact object histories without loading the citywide state files."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Iterator


SLOTS_PER_DAY = 96


def slot_of(time: str) -> int:
    hour, minute = map(int, time.split(":"))
    if not (0 <= hour < 24 and minute in (0, 15, 30, 45)):
        raise ValueError(f"Invalid 15-minute time: {time}")
    return hour * 4 + minute // 15


def time_of(slot: int) -> str:
    slot %= SLOTS_PER_DAY
    return f"{slot // 4:02d}:{slot % 4 * 15:02d}"


def load_pois(path: Path) -> dict[int, dict[str, float | int | None]]:
    pois = {}
    with path.open(encoding="utf-8") as source:
        source.readline()
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 8:
                continue
            match = re.match(r"^\s*(\d+(?:\.\d+)?)", row[4])
            capacity = float(match.group(1)) if match else None
            pois[int(row[0])] = {
                "latitude": float(row[1]), "longitude": float(row[2]),
                "category": int(row[3]), "capacity": capacity,
            }
    return pois


def load_edges(path: Path, wanted: set[int] | None = None) -> dict[int, dict[str, float | int]]:
    edges = {}
    with path.open(encoding="utf-8") as source:
        source.readline()
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 5:
                continue
            edge_id = int(row[0])
            if wanted is not None and edge_id not in wanted:
                continue
            edges[edge_id] = {
                "from_node": int(row[1]), "to_node": int(row[2]),
                "length_m": float(row[3]), "speed_limit_kmh": float(row[4]),
            }
    return edges


def load_mta_signal(period_path: Path, link_path: Path) -> dict[int, tuple[float, ...]]:
    station_period: dict[tuple[str, str], float] = {}
    period_names = ("00:00", "06:00", "12:00", "18:00")
    with period_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            station_period[(row["station_complex_id"], row["time"])] = float(
                row["ridership_mean_per_observation"]
            )
    linked = {}
    with link_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            station_id = row["station_complex_id"]
            if station_id:
                linked[int(row["poi_id"])] = tuple(
                    station_period.get((station_id, time), 0.0) for time in period_names
                )
    return linked


def _parse_state(line: bytes, kind: str) -> tuple[int, int, tuple[float, ...]] | None:
    fields = line.decode("utf-8").rstrip("\r\n").split("\t")
    if len(fields) < 6 or len(fields[0]) != 5 or fields[0][2] != ":":
        return None
    try:
        slot = slot_of(fields[0])
        identifier = int(fields[1])
        if kind == "poi":
            values = tuple(float(value) for value in fields[2:8])
        else:
            values = tuple(float(value) for value in fields[2:6])
    except (ValueError, IndexError):
        return None
    return slot, identifier, values


def sample_histories(path: Path, kind: str, count: int, seed: int) -> Iterator[tuple[int, list[tuple[float, ...]]]]:
    """Seek to random complete object blocks; citywide road data stays on disk."""
    if kind not in ("poi", "road") or count < 1:
        raise ValueError("kind must be poi or road, and count must be positive")
    rng = random.Random(seed)
    found: set[int] = set()
    with path.open("rb") as source:
        source.readline()
        source.readline()
        header_end = source.tell()
        file_size = path.stat().st_size
        # A forward seek cannot find the first block, so include it explicitly.
        first = source.readline()
        parsed = _parse_state(first, kind)
        if parsed is not None and parsed[0] == 0:
            identifier = parsed[1]
            history = [parsed[2]]
            for expected_slot in range(1, SLOTS_PER_DAY):
                next_row = _parse_state(source.readline(), kind)
                if next_row is None or next_row[0] != expected_slot or next_row[1] != identifier:
                    break
                history.append(next_row[2])
            if len(history) == SLOTS_PER_DAY:
                found.add(identifier)
                yield identifier, history
        attempts = 0
        while len(found) < count and attempts < count * 20:
            attempts += 1
            source.seek(rng.randrange(header_end, file_size))
            source.readline()  # Discard a partial row.
            while True:
                first = source.readline()
                if not first:
                    break
                parsed = _parse_state(first, kind)
                if parsed is not None and parsed[0] == 0:
                    break
            if not first or parsed is None:
                continue
            _, identifier, first_values = parsed
            if identifier in found:
                continue
            history = [first_values]
            for expected_slot in range(1, SLOTS_PER_DAY):
                next_row = _parse_state(source.readline(), kind)
                if next_row is None or next_row[0] != expected_slot or next_row[1] != identifier:
                    break
                history.append(next_row[2])
            if len(history) == SLOTS_PER_DAY:
                found.add(identifier)
                yield identifier, history
    if not found:
        raise ValueError(f"No complete {kind} histories found in {path}")


def iter_histories(path: Path, kind: str, limit: int | None = None) -> Iterator[tuple[int, list[tuple[float, ...]]]]:
    """Sequentially yield complete object blocks with constant memory usage."""
    if kind not in ("poi", "road"):
        raise ValueError("kind must be poi or road")
    yielded = 0
    with path.open("rb") as source:
        source.readline()
        source.readline()
        while limit is None or yielded < limit:
            first = _parse_state(source.readline(), kind)
            if first is None:
                break
            slot, identifier, values = first
            if slot != 0:
                raise ValueError(f"{path}: expected slot 00:00, got {slot} for {identifier}")
            history = [values]
            for expected_slot in range(1, SLOTS_PER_DAY):
                next_row = _parse_state(source.readline(), kind)
                if next_row is None or next_row[0] != expected_slot or next_row[1] != identifier:
                    raise ValueError(f"{path}: incomplete history for object {identifier}")
                history.append(next_row[2])
            yielded += 1
            yield identifier, history


def iter_road_histories_with_edges(
    state_path: Path, edge_path: Path, limit: int | None = None
) -> Iterator[tuple[int, dict[str, float | int], list[tuple[float, ...]]]]:
    """Stream a road history alongside its static edge row, without an edge dict."""
    with state_path.open("rb") as states, edge_path.open(encoding="utf-8") as edges:
        state_count = int(states.readline())
        states.readline()
        edge_header = edges.readline().rstrip("\n").split("\t")
        edge_count = int(edge_header[1])
        if state_count != edge_count:
            raise ValueError("Road state and edge counts differ")
        yielded = 0
        while limit is None or yielded < limit:
            edge_row = edges.readline()
            first = _parse_state(states.readline(), "road")
            if not edge_row or first is None:
                break
            edge = edge_row.rstrip("\n").split("\t")
            if len(edge) < 5 or int(edge[0]) != first[1] or first[0] != 0:
                raise ValueError(f"Road static/dynamic ID mismatch at {first[1]}")
            history = [first[2]]
            for expected_slot in range(1, SLOTS_PER_DAY):
                next_row = _parse_state(states.readline(), "road")
                if next_row is None or next_row[0] != expected_slot or next_row[1] != first[1]:
                    raise ValueError(f"Incomplete road history for edge {first[1]}")
                history.append(next_row[2])
            yield first[1], {
                "from_node": int(edge[1]), "to_node": int(edge[2]),
                "length_m": float(edge[3]), "speed_limit_kmh": float(edge[4]),
            }, history
            yielded += 1


def load_initial_rows(path: Path, kind: str, wanted: set[int]) -> dict[int, tuple[float, ...]]:
    """Get slot 00:00 for selected IDs from a small four-period baseline file."""
    result = {}
    with path.open(encoding="utf-8") as source:
        source.readline()
        source.readline()
        for line in source:
            parsed = _parse_state(line.encode("utf-8"), kind)
            if parsed and parsed[0] == 0 and parsed[1] in wanted:
                result[parsed[1]] = parsed[2]
                if len(result) == len(wanted):
                    break
    return result
