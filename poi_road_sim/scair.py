"""SCAIR-compatible strategic, crowd-aware assignment baselines.

The original SCAIR paper models a park as a facility graph and assigns each
arriving visitor a feasible itinerary using social utility and overlap-based
queue cost. This module keeps that objective separate from RISE-MPC: it uses
current facility/POI crowding only and never rolls the road state forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .recommender import CandidateRoute
from .simulator import POIState, Request


@dataclass(frozen=True)
class ScairWeights:
    queue_weight: float = 1.0
    travel_weight: float = 0.1
    popularity_weight: float = 1.0
    capacity_floor: float = 1.0


def _route_time_minutes(candidate: CandidateRoute, road_times: Mapping[int, float] | None) -> float:
    if not road_times:
        return float(len(candidate.request.edge_ids))
    edges = candidate.request.edge_ids + candidate.request.egress_edge_ids
    return sum(float(road_times.get(edge_id, 0.0)) for edge_id in edges) / 60.0


class ScairAdapter:
    """Current-state SCAIR adaptation for a common candidate set.

    Each candidate is an itinerary containing one required POI and its access
    route. ``popularity`` is supplied externally because the NYC POI file does
    not contain the park dataset's review/popularity field. The selected
    request is returned; callers execute it in the common simulator.
    """

    def __init__(self, popularity: Mapping[int, float] | None = None,
                 weights: ScairWeights | None = None) -> None:
        self.popularity = dict(popularity or {})
        self.weights = weights or ScairWeights()

    def _queue_cost(self, candidate: CandidateRoute, pois: Mapping[int, POIState],
                    assigned: Sequence[CandidateRoute]) -> float:
        poi_id = candidate.request.destination_poi_id
        poi = pois[poi_id]
        capacity = max(self.weights.capacity_floor, poi.service_people_per_slot)
        existing = float(poi.waiting) + float(poi.occupancy) / capacity
        same_poi = sum(item.request.people for item in assigned
                       if item.request.destination_poi_id == poi_id)
        shared_edges = sum(
            min(candidate.request.edge_ids.count(edge), item.request.edge_ids.count(edge))
            for item in assigned for edge in set(candidate.request.edge_ids)
        )
        return existing + same_poi / capacity + 0.1 * shared_edges

    def score(self, candidate: CandidateRoute, pois: Mapping[int, POIState],
              assigned: Sequence[CandidateRoute] = (),
              road_times: Mapping[int, float] | None = None) -> float:
        queue = self._queue_cost(candidate, pois, assigned)
        travel = _route_time_minutes(candidate, road_times)
        popularity = float(self.popularity.get(candidate.request.destination_poi_id, 0.0))
        # SCAIR's original popularity/queue idea, with travel as the city
        # adaptation cost. Higher score is preferred.
        return self.weights.popularity_weight * popularity - self.weights.queue_weight * queue \
            - self.weights.travel_weight * travel

    def choose(self, candidate_sets: Sequence[Sequence[CandidateRoute]],
               pois: Mapping[int, POIState],
               road_times: Mapping[int, float] | None = None) -> list[CandidateRoute]:
        """Assign each arriving user greedily using current crowd state only."""
        selected: list[CandidateRoute] = []
        for candidates in candidate_sets:
            feasible = [candidate for candidate in candidates if candidate.request.destination_poi_id in pois]
            if not feasible:
                continue
            selected.append(max(feasible, key=lambda candidate: self.score(
                candidate, pois, selected, road_times
            )))
        return selected

    def choose_one(self, candidates: Sequence[CandidateRoute],
                   pois: Mapping[int, POIState],
                   road_times: Mapping[int, float] | None = None) -> CandidateRoute | None:
        selected = self.choose((candidates,), pois, road_times)
        return selected[0] if selected else None
