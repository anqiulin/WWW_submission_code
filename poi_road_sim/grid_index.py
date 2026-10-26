"""Fast grid-index candidate POI search for recommendation queries.

The index narrows POI candidates using the straight origin-destination
corridor and expanding grid rings. A route factory then supplies a connected
edge sequence (usually from a cached route table or a local router). No graph
search is performed during POI candidate discovery.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .recommender import CandidateRoute
from .simulator import Request


@dataclass(frozen=True)
class GridQuery:
    request_id: str
    origin_node_id: int
    origin_lat: float
    origin_lon: float
    destination_node_id: int
    destination_lat: float
    destination_lon: float
    people: float = 1.0
    category: int | None = None
    categories: tuple[int, ...] = ()
    vehicles: float = 1.0
    cell_size_deg: float = 0.005
    max_layers: int = 4
    max_pois: int = 20


class POIGridIndex:
    """In-memory spatial hash keyed by ``(latitude_cell, longitude_cell)``."""

    def __init__(self, cell_size_deg: float = 0.005) -> None:
        if cell_size_deg <= 0:
            raise ValueError("cell_size_deg must be positive")
        self.cell_size_deg = cell_size_deg
        self.cells: dict[tuple[int, int], list[tuple[int, float, float, int]]] = defaultdict(list)

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        return (math.floor(lat / self.cell_size_deg), math.floor(lon / self.cell_size_deg))

    def add(self, poi_id: int, latitude: float, longitude: float, category: int) -> None:
        self.cells[self._cell(latitude, longitude)].append(
            (int(poi_id), float(latitude), float(longitude), int(category))
        )

    @classmethod
    def from_records(cls, records: Iterable[Mapping], cell_size_deg: float = 0.005) -> "POIGridIndex":
        result = cls(cell_size_deg)
        for row in records:
            result.add(row["poi_id"], row["latitude"], row["longitude"], row["category"])
        return result

    @classmethod
    def from_poi_mapping(cls, pois: Mapping[int, Mapping], cell_size_deg: float = 0.005) -> "POIGridIndex":
        """Build directly from ``data.load_pois`` output."""
        result = cls(cell_size_deg)
        for poi_id, row in pois.items():
            result.add(poi_id, row["latitude"], row["longitude"], row["category"])
        return result

    def _line_cells(self, query: GridQuery) -> list[tuple[int, int]]:
        start = self._cell(query.origin_lat, query.origin_lon)
        end = self._cell(query.destination_lat, query.destination_lon)
        count = max(abs(end[0] - start[0]), abs(end[1] - start[1])) * 2 + 1
        result = []
        for i in range(count):
            ratio = i / max(1, count - 1)
            lat = query.origin_lat + ratio * (query.destination_lat - query.origin_lat)
            lon = query.origin_lon + ratio * (query.destination_lon - query.origin_lon)
            cell = self._cell(lat, lon)
            if not result or result[-1] != cell:
                result.append(cell)
        return result

    @staticmethod
    def _ring(cells: Iterable[tuple[int, int]], layer: int) -> set[tuple[int, int]]:
        return {(row + dr, col + dc) for row, col in cells
                for dr in range(-layer, layer + 1)
                for dc in range(-layer, layer + 1)
                if max(abs(dr), abs(dc)) == layer}

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat_scale = 111.32
        lon_scale = 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
        return math.hypot((lat1 - lat2) * lat_scale, (lon1 - lon2) * lon_scale)

    def search(self, query: GridQuery) -> list[tuple[int, float, int]]:
        if query.max_layers < 0 or query.max_pois < 1:
            raise ValueError("max_layers must be non-negative and max_pois positive")
        if abs(query.cell_size_deg - self.cell_size_deg) > 1e-12:
            raise ValueError("query cell_size_deg does not match the index")
        corridor = self._line_cells(query)
        found: dict[int, tuple[float, int]] = {}
        for layer in range(query.max_layers + 1):
            cells = corridor if layer == 0 else self._ring(corridor, layer)
            for cell in cells:
                for poi_id, lat, lon, category in self.cells.get(cell, ()):
                    allowed = set(query.categories)
                    if query.category is not None:
                        allowed.add(query.category)
                    if allowed and category not in allowed:
                        continue
                    distance = min(self._distance_km(lat, lon, query.origin_lat, query.origin_lon),
                                   self._distance_km(lat, lon, query.destination_lat, query.destination_lon))
                    found.setdefault(poi_id, (distance, layer))
            if len(found) >= query.max_pois:
                break
        return [(poi_id, value[0], value[1]) for poi_id, value in
                sorted(found.items(), key=lambda item: (item[1][1], item[1][0]))[:query.max_pois]]


def generate_grid_candidates(
    index: POIGridIndex, query: GridQuery,
    route_factory: Callable[[int], tuple[int, ...] | None],
    egress_route_factory: Callable[[int], tuple[int, ...] | None] | None = None,
    max_results: int | None = None,
) -> list[CandidateRoute]:
    """Build connected candidate requests from the grid search result.

    ``route_factory`` is intentionally injectable: it can use cached routes,
    a local router, or precomputed origin/POI paths without slowing down the
    spatial candidate search.
    """
    if query.people <= 0 or query.vehicles <= 0:
        raise ValueError("people and vehicles must be positive")
    if max_results is not None and max_results < 1:
        raise ValueError("max_results must be positive")
    result: list[CandidateRoute] = []
    for index_no, (poi_id, distance, layer) in enumerate(index.search(query)):
        path = route_factory(poi_id)
        if not path:
            continue
        egress_path = egress_route_factory(poi_id) if egress_route_factory else ()
        egress = tuple(egress_path) if egress_path is not None else ()
        if egress_route_factory and not egress:
            continue
        request = Request(
            request_id=f"{query.request_id}-poi-{poi_id}-grid-{index_no}",
            origin_node_id=query.origin_node_id, destination_poi_id=poi_id,
            people=query.people, edge_ids=tuple(path), vehicles=query.vehicles,
            final_destination_node_id=query.destination_node_id if egress else None,
            egress_edge_ids=egress,
        )
        result.append(CandidateRoute(request, f"poi-{poi_id}-grid-layer-{layer}"))
        if max_results is not None and len(result) >= max_results:
            break
    return result
