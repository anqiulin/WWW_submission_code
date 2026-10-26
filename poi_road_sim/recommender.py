"""Model-predictive recommendation over candidate POI routes.

The recommender deliberately sits outside ``CoupledSimulator``.  The
simulator remains the transition model; this module creates counterfactual
copies, scores them over a finite horizon, and executes only the first action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .simulator import CoupledSimulator, Request, STEP_SECONDS
from .metrics import global_load


@dataclass(frozen=True)
class CandidateRoute:
    """A feasible route to one candidate POI for the current request."""

    request: Request
    label: str = ""


@dataclass(frozen=True)
class RecommendationWeights:
    """Weights for system cost in vehicle/person minutes."""

    user_travel: float = 0.0
    user_wait: float = 0.0
    road_externality: float = 1.0
    poi_externality: float = 1.0
    unmet_penalty: float = 10_000.0
    local_scope: bool = False


@dataclass(frozen=True)
class RecommendationDecision:
    candidate: CandidateRoute | None
    score: float
    evaluated: int
    rejected: int
    details: tuple[dict, ...]


class MPCRecommender:
    """Finite-horizon counterfactual route selector.

    ``future_requests`` maps a relative step (0 means the current step) to
    already known requests. They are replayed identically in every candidate
    rollout, so the score measures the selected recommendation's incremental
    effect on later users and the environment.
    """

    def __init__(self, horizon: int = 4, discount: float = 0.95,
                 weights: RecommendationWeights | None = None) -> None:
        if horizon < 1 or not 0 < discount <= 1:
            raise ValueError("horizon must be positive and discount in (0, 1]")
        self.horizon = horizon
        self.discount = discount
        self.weights = weights or RecommendationWeights()

    @staticmethod
    def _free_time_minutes(simulator: CoupledSimulator, edge_id: int) -> float:
        road = simulator.roads[edge_id]
        speed = max(5.0, road.speed_limit_kmh if road.speed_limit_kmh > 0 else road.speed_kmh)
        return max(1.0, road.length_m * 3.6 / speed) / 60.0

    def _environment_cost(self, simulator: CoupledSimulator, result: dict,
                          road_scope: set[int] | None = None,
                          poi_scope: set[int] | None = None) -> tuple[float, float]:
        # Use every road edge and every POI in the simulator.  This keeps the
        # MPC objective aligned with the paper's global-load definition rather
        # than measuring only the candidate route's local subgraph.
        if not self.weights.local_scope:
            load = global_load(simulator, result)
            return load["road_load"], load["poi_load"]
        roads = result.get("roads", {})
        pois = result.get("pois", {})
        road_queue = road_delay = poi_wait = 0.0
        for edge_id in road_scope or ():
            road = simulator.roads.get(edge_id)
            row = roads.get(edge_id, {})
            if road is None:
                continue
            queued = float(row.get("queue_vehicles", road.queue_vehicles))
            road_queue += queued * 15.0
            served = float(row.get("outflow_vehicles", road.outflow_vehicles))
            free = simulator._travel_time(road, 0.0)
            travel = float(row.get("travel_time_sec", road.travel_time_sec))
            road_delay += served * max(0.0, travel - free) / 60.0
        for poi_id in poi_scope or ():
            poi = simulator.pois.get(poi_id)
            if poi is not None:
                poi_wait += float(pois.get(poi_id, {}).get("waiting", poi.waiting)) * 15.0
        mode = getattr(getattr(simulator, "cost_weights", None), "mode", "vehicle_min")
        cost_weights = getattr(simulator, "cost_weights", None)
        road_weight = self.weights.road_externality
        poi_weight = self.weights.poi_externality
        road_reference = getattr(cost_weights, "road_reference_vehicle_min", 1.0)
        poi_reference = getattr(cost_weights, "poi_reference_person_min", 1.0)
        if mode == "queue_wait":
            return (road_weight * road_queue / road_reference,
                    poi_weight * poi_wait / poi_reference)
        return (road_weight * (road_queue + road_delay) / road_reference,
                poi_weight * poi_wait / poi_reference)

    def _rollout(self, simulator: CoupledSimulator, requests: Mapping[int, Sequence[Request]],
                 first: Sequence[Request] = ()) -> tuple[float, float, dict | None]:
        road_cost = poi_cost = 0.0
        first_result = None
        for relative_step in range(self.horizon):
            scheduled = list(requests.get(relative_step, ()))
            if relative_step == 0:
                scheduled = list(first) + scheduled
            result = simulator.step(scheduled)
            if first_result is None:
                first_result = result
            road_scope = set()
            poi_scope = set()
            if self.weights.local_scope:
                for item in scheduled:
                    road_scope.update(item.edge_ids)
                    road_scope.update(item.egress_edge_ids)
                    poi_scope.add(item.destination_poi_id)
            step_road, step_poi = self._environment_cost(
                simulator, result, road_scope, poi_scope)
            factor = self.discount ** relative_step
            road_cost += factor * step_road
            poi_cost += factor * step_poi
        return road_cost, poi_cost, first_result

    def choose(self, simulator: CoupledSimulator, candidates: Sequence[CandidateRoute],
               future_requests: Mapping[int, Sequence[Request]] | None = None) -> RecommendationDecision:
        if not candidates:
            raise ValueError("at least one candidate route is required")
        future_requests = future_requests or {}
        outcomes: list[tuple[CandidateRoute, float, float, dict]] = []
        rejected = 0
        for candidate in candidates:
            trial = simulator.fork()
            try:
                road_cost, poi_cost, first_result = self._rollout(
                    trial, future_requests, (candidate.request,)
                )
            except (KeyError, ValueError):
                rejected += 1
                continue
            if first_result is None:
                rejected += 1
                continue
            outcomes.append((candidate, road_cost, poi_cost, first_result))
        if not outcomes:
            return RecommendationDecision(None, math.inf, 0, rejected, ())
        baseline_road, baseline_poi = outcomes[0][1:3]
        details: list[dict] = []
        best: tuple[float, CandidateRoute] | None = None
        for candidate, road_cost, poi_cost, first_result in outcomes:
            request = candidate.request
            full_route = request.edge_ids + request.egress_edge_ids
            modeled_route_time = sum(first_result["roads"][edge]["travel_time_sec"]
                                     for edge in full_route) / 60.0
            arrival_slots = first_result["request_arrival_slots"]
            scheduled_delay = ((arrival_slots[0] - simulator.slot) * STEP_SECONDS / 60.0
                               if arrival_slots else 0.0)
            # A capacity delay may schedule an edge in a later slot even though
            # that edge is not assigned in the current result row.
            route_time = max(modeled_route_time, scheduled_delay)
            poi_state = first_result["pois"][request.destination_poi_id]
            user_wait = float(poi_state["wait_time_min"])
            road_externality = road_cost - baseline_road
            poi_externality = poi_cost - baseline_poi
            score = (
                self.weights.user_travel * route_time
                + self.weights.user_wait * user_wait
                + self.weights.road_externality * road_externality
                + self.weights.poi_externality * poi_externality
            )
            item = {"label": candidate.label, "poi_id": request.destination_poi_id,
                    "score": score, "route_time_min": route_time,
                    "user_wait_min": user_wait, "road_externality": road_externality,
                    "poi_externality": poi_externality}
            details.append(item)
            if best is None or score < best[0]:
                best = (score, candidate)
        details.sort(key=lambda item: item["score"])
        return RecommendationDecision(best[1], best[0], len(details), rejected, tuple(details))

    def recommend_and_step(self, simulator: CoupledSimulator,
                           candidates: Sequence[CandidateRoute],
                           future_requests: Mapping[int, Sequence[Request]] | None = None) -> tuple[RecommendationDecision, dict]:
        decision = self.choose(simulator, candidates, future_requests)
        if decision.candidate is None:
            raise ValueError("no feasible candidate route")
        return decision, simulator.step([decision.candidate.request])

    def choose_batch(self, simulator: CoupledSimulator,
                     candidate_sets: Sequence[Sequence[CandidateRoute]],
                     future_requests: Mapping[int, Sequence[Request]] | None = None
                     ) -> list[RecommendationDecision]:
        """Greedily assign a batch while accounting for earlier assignments.

        Earlier assignments remain in the planning state's event schedule.
        Each candidate rollout therefore starts from the same provisional
        batch state without replaying the growing request prefix.
        """
        planning = simulator.fork()
        background = {edge_id: planning._background(road, planning.slot)
                      for edge_id, road in planning.roads.items()}
        decisions: list[RecommendationDecision] = []
        for candidates in candidate_sets:
            decision = self.choose(planning, candidates, future_requests)
            if decision.candidate is None:
                decisions.append(decision)
                continue
            decisions.append(decision)
            planning._schedule(decision.candidate.request, background)
        return decisions

    def recommend_batch_and_step(self, simulator: CoupledSimulator,
                                 candidate_sets: Sequence[Sequence[CandidateRoute]],
                                 future_requests: Mapping[int, Sequence[Request]] | None = None
                                 ) -> tuple[list[RecommendationDecision], dict]:
        decisions = self.choose_batch(simulator, candidate_sets, future_requests)
        if not decisions:
            raise ValueError("no feasible candidate route for the batch")
        selected = [decision.candidate.request for decision in decisions
                    if decision.candidate is not None]
        return decisions, simulator.step(selected)

    def choose_query(self, simulator: CoupledSimulator, query: Any,
                     future_requests: Mapping[int, Sequence[Request]] | None = None
                     ) -> RecommendationDecision:
        """Generate graph candidates for a query, then run the MPC selector.

        The import is local so route generation remains optional for callers
        that already provide candidate routes.
        """
        from .routes import generate_candidates
        candidates = generate_candidates(simulator.roads, simulator.pois, query)
        return self.choose(simulator, candidates, future_requests)

    def choose_grid_query(self, simulator: CoupledSimulator, index: Any, query: Any,
                          route_factory: Any,
                          egress_route_factory: Any = None,
                          future_requests: Mapping[int, Sequence[Request]] | None = None
                          ) -> RecommendationDecision:
        """Search POIs by grid rings, then score the resulting route candidates."""
        from .grid_index import generate_grid_candidates
        candidates = generate_grid_candidates(index, query, route_factory, egress_route_factory)
        return self.choose(simulator, candidates, future_requests)

    def choose_h2h_grid_query(self, simulator: CoupledSimulator, index: Any, query: Any,
                              router: Any,
                              future_requests: Mapping[int, Sequence[Request]] | None = None
                              ) -> RecommendationDecision:
        """Use indexed directed routes for both legs of each grid POI candidate."""
        from .h2h import generate_h2h_grid_candidates
        router.reset_travel_times()
        current_times = {edge_id: max(road.travel_time_sec, simulator._travel_time(road, 0.0))
                         for edge_id, road in simulator.roads.items()
                         if road.travel_time_sec > 0}
        router.update_travel_times(current_times)
        candidates = generate_h2h_grid_candidates(index, query, simulator.pois, router)
        return self.choose(simulator, candidates, future_requests)
