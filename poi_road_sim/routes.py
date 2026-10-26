"""Deterministic candidate route generation on the directed edge graph."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .recommender import CandidateRoute
from .simulator import RoadState, POIState, Request


@dataclass(frozen=True)
class UserQuery:
    request_id: str
    origin_node_id: int
    people: float = 1.0
    vehicles: float = 1.0
    category: int | None = None
    categories: tuple[int, ...] = ()
    max_routes_per_poi: int = 2
    max_pois: int = 20
    max_detour_ratio: float = 1.8


def _free_time(road: RoadState) -> float:
    speed = max(5.0, road.speed_limit_kmh if road.speed_limit_kmh > 0 else road.speed_kmh)
    return max(1.0, road.length_m * 3.6 / speed)


def shortest_path(roads: Mapping[int, RoadState], origin: int, target: int,
                  blocked_edges: frozenset[int] = frozenset(),
                  blocked_nodes: frozenset[int] = frozenset()) -> tuple[float, tuple[int, ...]] | None:
    """Return a free-flow shortest directed path using Dijkstra."""
    adjacency: dict[int, list[RoadState]] = {}
    for road in roads.values():
        if road.edge_id not in blocked_edges and road.from_node not in blocked_nodes:
            adjacency.setdefault(road.from_node, []).append(road)
    queue: list[tuple[float, int, tuple[int, ...]]] = [(0.0, origin, ())]
    best: dict[int, float] = {origin: 0.0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if cost > best.get(node, math.inf) + 1e-9:
            continue
        if node == target:
            return cost, path
        for road in adjacency.get(node, ()):
            if road.to_node in blocked_nodes:
                continue
            next_cost = cost + _free_time(road)
            if next_cost < best.get(road.to_node, math.inf) - 1e-9:
                best[road.to_node] = next_cost
                heapq.heappush(queue, (next_cost, road.to_node, path + (road.edge_id,)))
    return None


def k_shortest_paths(roads: Mapping[int, RoadState], origin: int, target: int,
                     k: int = 2) -> list[tuple[float, tuple[int, ...]]]:
    """Generate up to ``k`` loop-free alternatives with Yen's algorithm."""
    if k < 1:
        raise ValueError("k must be positive")
    first = shortest_path(roads, origin, target)
    if first is None:
        return []
    accepted = [first]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    seen = {first[1]}
    for _ in range(1, k):
        previous = accepted[-1]
        for index in range(len(previous[1])):
            root = previous[1][:index]
            root_nodes = [origin]
            for edge_id in root:
                root_nodes.append(roads[edge_id].to_node)
            spur_node = root_nodes[-1]
            blocked_edges: set[int] = set()
            for _, path in accepted:
                if path[:index] == root and index < len(path):
                    blocked_edges.add(path[index])
            root_cost = sum(_free_time(roads[edge_id]) for edge_id in root)
            spur = shortest_path(roads, spur_node, target, frozenset(blocked_edges),
                                 frozenset(root_nodes[:-1]))
            if spur is None:
                continue
            total_path = root + spur[1]
            if total_path in seen:
                continue
            seen.add(total_path)
            heapq.heappush(candidates, (root_cost + spur[0], total_path))
        if not candidates:
            break
        accepted.append(heapq.heappop(candidates))
    return accepted


def generate_candidates(roads: Mapping[int, RoadState], pois: Mapping[int, POIState],
                        query: UserQuery) -> list[CandidateRoute]:
    """Generate POI and route candidates satisfying the category and detour bound."""
    if query.people <= 0 or query.vehicles <= 0:
        raise ValueError("people and vehicles must be positive")
    all_candidates: list[tuple[float, int, tuple[int, ...]]] = []
    for poi_id, poi in pois.items():
        if query.category is not None and poi.category != query.category:
            continue
        paths = k_shortest_paths(roads, query.origin_node_id, poi.node_id, query.max_routes_per_poi)
        for cost, path in paths:
            all_candidates.append((cost, poi_id, path))
    all_candidates.sort(key=lambda item: item[0])
    if not all_candidates:
        return []
    reference = all_candidates[0][0]
    selected = [item for item in all_candidates
                if item[0] <= reference * max(1.0, query.max_detour_ratio)][:query.max_pois]
    return [CandidateRoute(Request(
        request_id=f"{query.request_id}-poi-{poi_id}-r{index}",
        origin_node_id=query.origin_node_id, destination_poi_id=poi_id,
        people=query.people, edge_ids=path, vehicles=query.vehicles,
    ), label=f"poi-{poi_id}-route-{index}") for index, (_, poi_id, path) in enumerate(selected)]
