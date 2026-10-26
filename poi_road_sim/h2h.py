"""Build and query directed road routes using an undirected H2H lower bound.

The H2H labels are built on the largest weakly connected component. They are
admissible A* heuristics; returned paths always follow the original directed
edges. For other components, A* uses a zero heuristic.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import sqlite3
import subprocess
import shutil
from array import array
from pathlib import Path
from typing import Mapping

from .grid_index import GridQuery, POIGridIndex, generate_grid_candidates


SOURCE = Path(__file__).resolve().parent / "native" / "h2h_router.cpp"
BUILDER = Path(__file__).resolve().parent / "native" / "h2h_build.cpp"


def compile_native(output: Path) -> tuple[Path, Path]:
    """Compile the original builder and the persistent directed query library."""
    output.mkdir(parents=True, exist_ok=True)
    builder = output / "h2h_build"
    library = output / "libh2h_router.so"
    local_compiler = Path(__file__).resolve().parents[2] / ".toolchain" / "bin" / "g++"
    compiler = str(local_compiler) if local_compiler.is_file() else shutil.which("g++")
    if not compiler:
        raise RuntimeError("H2H requires g++; install it or provide .toolchain/bin/g++")
    subprocess.run([compiler, "-O2", "-std=c++14", str(BUILDER), "-o", str(builder)], check=True)
    subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC", str(SOURCE),
                    "-o", str(library)], check=True)
    return builder, library


def _effective_speeds(store: Path, edge_count: int) -> array:
    """Read the fixed simulated road limits without loading state JSON in Python."""
    speeds = array("d", [0.0]) * (edge_count + 1)
    count = 0
    shards = sorted(store.glob("state_*.sqlite3"))
    if not shards:
        raise ValueError("state store has no SQLite road shards")
    for shard in shards:
        connection = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        try:
            for edge_id, speed in connection.execute(
                "SELECT object_id, json_extract(state_json, '$.speed_limit_kmh') FROM road_state"
            ):
                if not (1 <= edge_id <= edge_count) or speeds[edge_id] or speed is None or speed <= 0:
                    raise ValueError(f"invalid effective speed at edge {edge_id}")
                speeds[edge_id] = float(speed)
                count += 1
        finally:
            connection.close()
    if count != edge_count:
        raise ValueError(f"state store has {count} roads; expected {edge_count}")
    return speeds


def prepare_graph(edge_file: Path, output: Path, state_store: Path | None = None) -> dict:
    """Export the full directed graph and a connected undirected H2H graph.

    H2H uses floor(lower-bound seconds * 100) for every edge. With a complete
    state store, effective simulated speed limits supply the bound; otherwise
    unknown static limits use the simulator's one-second edge minimum.
    """
    output.mkdir(parents=True, exist_ok=True)
    with edge_file.open(encoding="utf-8") as source:
        header = source.readline().split()
    if len(header) != 2:
        raise ValueError("edge.txt must start with node and edge counts")
    node_count, edge_count = map(int, header)
    speeds = _effective_speeds(state_store, edge_count) if state_store else None
    parent = list(range(node_count + 1))
    sizes = [1] * (node_count + 1)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            if sizes[left] < sizes[right]:
                left, right = right, left
            parent[right] = left
            sizes[left] += sizes[right]

    with edge_file.open(encoding="utf-8") as source:
        source.readline()
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 5:
                raise ValueError("incomplete edge row")
            left, right = int(row[1]), int(row[2])
            if not (1 <= left <= node_count and 1 <= right <= node_count):
                raise ValueError("edge endpoint outside node range")
            union(left, right)
    largest = max(range(1, node_count + 1), key=lambda node: sizes[find(node)])
    root = find(largest)
    local = [0] * (node_count + 1)
    selected = [node for node in range(1, node_count + 1) if find(node) == root]
    for dense, original in enumerate(selected, 1):
        local[original] = dense
    with (output / "nodes.txt").open("w", encoding="utf-8") as mapping:
        mapping.write(f"{node_count} {len(selected)}\n")
        for original in selected:
            mapping.write(f"{original} {local[original]}\n")

    directed_count = 0
    indexed_count = 0
    with (output / "directed.txt").open("w", encoding="utf-8") as directed, \
            (output / "undirected.gr").open("w", encoding="utf-8") as undirected, \
            edge_file.open(encoding="utf-8") as source:
        directed.write(f"{edge_count}\n")
        undirected.write(f"{len(selected):020d} {0:020d}\n")
        source.readline()
        for row in csv.reader(source, delimiter="\t"):
            edge_id, left, right = map(int, row[:3])
            length, speed = float(row[3]), float(row[4])
            if not math.isfinite(length) or not math.isfinite(speed) or length < 0:
                raise ValueError(f"invalid road weight at edge {edge_id}")
            effective_speed = speeds[edge_id] if speeds is not None else (speed if speed > 0 else 30.0)
            free_time = max(1.0, length * 3.6 / max(5.0, effective_speed))
            minimum_time = free_time if speeds is not None or speed > 0 else 1.0
            directed.write(f"{edge_id} {left} {right} {free_time:.12f} {minimum_time:.12f}\n")
            directed_count += 1
            if local[left] and local[right] and left != right:
                weight = max(1, math.floor(minimum_time * 100))
                undirected.write(f"a {local[left]} {local[right]} {weight}\n")
                indexed_count += 1
        if directed_count != edge_count:
            raise ValueError(f"header declares {edge_count} roads; found {directed_count}")
        undirected.seek(0)
        undirected.write(f"{len(selected):020d} {indexed_count:020d}\n")
    report = {"nodes": node_count, "directed_edges": directed_count,
              "indexed_nodes": len(selected), "indexed_edges": indexed_count,
              "weight": "effective fixed speed from state store" if speeds is not None else
                        "known-speed free-flow bound; unknown speed uses 1 second",
              "state_store": str(state_store) if state_store else None}
    (output / "graph.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class H2HRouter:
    """Reusable native router. Keep this object alive for repeated queries."""

    def __init__(self, directory: Path) -> None:
        directory = Path(directory)
        library = ctypes.CDLL(str(directory / "libh2h_router.so"))
        library.h2h_last_error.restype = ctypes.c_char_p
        library.h2h_open.argtypes = [ctypes.c_char_p] * 3
        library.h2h_open.restype = ctypes.c_void_p
        library.h2h_close.argtypes = [ctypes.c_void_p]
        library.h2h_lower_bound.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        library.h2h_lower_bound.restype = ctypes.c_double
        library.h2h_update_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                               ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        library.h2h_update_weights.restype = ctypes.c_int
        library.h2h_reset_weights.argtypes = [ctypes.c_void_p]
        library.h2h_reset_weights.restype = ctypes.c_int
        pointer = ctypes.POINTER(ctypes.c_int)
        library.h2h_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(pointer), ctypes.POINTER(pointer),
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        library.h2h_path.restype = ctypes.c_int
        library.h2h_distance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_double)]
        library.h2h_distance.restype = ctypes.c_int
        library.h2h_plain_distance.argtypes = library.h2h_distance.argtypes
        library.h2h_plain_distance.restype = ctypes.c_int
        library.h2h_free.argtypes = [pointer]
        self._library = library
        self._handle = library.h2h_open(*(str(directory / name).encode() for name in
                                           ("graph.index", "nodes.txt", "directed.txt")))
        if not self._handle:
            raise RuntimeError(library.h2h_last_error().decode())

    def close(self) -> None:
        if self._handle:
            self._library.h2h_close(self._handle)
            self._handle = None

    def __enter__(self) -> "H2HRouter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _error(self) -> RuntimeError:
        return RuntimeError(self._library.h2h_last_error().decode())

    def _check_open(self) -> None:
        if not self._handle:
            raise RuntimeError("H2H router is closed")

    def lower_bound(self, origin: int, target: int) -> float:
        self._check_open()
        result = self._library.h2h_lower_bound(self._handle, origin, target)
        if result < 0:
            raise self._error()
        return result

    def update_travel_times(self, times: Mapping[int, float]) -> None:
        """Apply current edge times before queries; updates are atomic on error."""
        self._check_open()
        ids = (ctypes.c_int * len(times))(*times.keys())
        values = (ctypes.c_double * len(times))(*times.values())
        if self._library.h2h_update_weights(self._handle, ids, values, len(times)) < 0:
            raise self._error()

    def reset_travel_times(self) -> None:
        """Restore exported baseline times before reusing a router in another branch."""
        self._check_open()
        if self._library.h2h_reset_weights(self._handle) < 0:
            raise self._error()

    def route(self, origin: int, target: int) -> tuple[float, tuple[int, ...], tuple[int, ...]] | None:
        """Return exact current cost, directed edge IDs, and path node IDs."""
        self._check_open()
        pointer = ctypes.POINTER(ctypes.c_int)
        edge_ids, nodes = pointer(), pointer()
        count, cost = ctypes.c_int(), ctypes.c_double()
        status = self._library.h2h_path(self._handle, origin, target,
                                        ctypes.byref(edge_ids), ctypes.byref(nodes),
                                        ctypes.byref(count), ctypes.byref(cost))
        if status < 0:
            raise self._error()
        if status == 0:
            return None
        try:
            return (cost.value, tuple(edge_ids[i] for i in range(count.value)),
                    tuple(nodes[i] for i in range(count.value + 1)))
        finally:
            self._library.h2h_free(edge_ids)
            self._library.h2h_free(nodes)

    def distance(self, origin: int, target: int) -> float | None:
        self._check_open()
        cost = ctypes.c_double()
        status = self._library.h2h_distance(self._handle, origin, target, ctypes.byref(cost))
        if status < 0:
            raise self._error()
        return cost.value if status else None

    def baseline_distance(self, origin: int, target: int) -> float | None:
        """Dijkstra on the same native graph, for validation and benchmarks."""
        self._check_open()
        cost = ctypes.c_double()
        status = self._library.h2h_plain_distance(self._handle, origin, target,
                                                   ctypes.byref(cost))
        if status < 0:
            raise self._error()
        return cost.value if status else None


def generate_h2h_grid_candidates(index: POIGridIndex, query: GridQuery,
                                 pois: Mapping[int, object], router: H2HRouter,
                                 max_results: int | None = None):
    """Connect grid POIs to origin and destination with directed routes."""
    def node_id(poi_id: int) -> int:
        poi = pois[poi_id]
        return int(poi["node_id"] if isinstance(poi, dict) else poi.node_id)

    def access(poi_id: int) -> tuple[int, ...] | None:
        route = router.route(query.origin_node_id, node_id(poi_id))
        return route[1] if route else None

    def egress(poi_id: int) -> tuple[int, ...] | None:
        route = router.route(node_id(poi_id), query.destination_node_id)
        return route[1] if route else None

    return generate_grid_candidates(index, query, access, egress, max_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H2H labels for directed NYC routing")
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-store", type=Path,
                        help="full road state store for stronger fixed free-flow labels")
    args = parser.parse_args()
    builder, _ = compile_native(args.output)
    report = prepare_graph(args.edge_file, args.output, args.state_store)
    print(json.dumps(report), flush=True)
    temporary_index = args.output / "graph.index.tmp"
    subprocess.run([str(builder), str(args.output / "undirected.gr"),
                    str(temporary_index)], check=True)
    temporary_index.replace(args.output / "graph.index")
    print(f"H2H index ready: {args.output / 'graph.index'}", flush=True)


if __name__ == "__main__":
    main()
