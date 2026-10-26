"""Lightweight learnable candidate policy.

The module has no third-party dependency.  It provides the same interfaces a
PyTorch/PPO policy can later replace: feature extraction, candidate scoring,
batch selection, save/load, and supervised imitation from MPC decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .recommender import CandidateRoute


def candidate_features(simulator, candidate: CandidateRoute) -> list[float]:
    request = candidate.request
    poi = simulator.pois[request.destination_poi_id]
    edges = request.edge_ids + request.egress_edge_ids
    if edges:
        reserved = simulator.reserved.get(simulator.slot, {})
        committed = getattr(simulator, "provisional_edge_counts", {})
        ratios = [
            (simulator.roads[e].queue_vehicles + simulator.roads[e].background_vehicles
             + reserved.get(e, 0.0) + committed.get(e, 0.0))
            / max(1.0, simulator.roads[e].capacity_veh_per_slot) for e in edges
        ]
        mean_vc = sum(ratios) / len(ratios)
        max_vc = max(ratios)
        length = sum(simulator.roads[e].length_m for e in edges)
        free_time = sum(simulator._travel_time(simulator.roads[e], 0.0) for e in edges) / 60.0
    else:
        mean_vc = max_vc = length = free_time = 0.0
    capacity = max(1.0, poi.capacity_people)
    return [1.0, math.log1p(length) / 10.0, free_time / 60.0,
            mean_vc, max_vc, poi.occupancy / capacity,
            poi.waiting / capacity,
            poi.service_people_per_slot / capacity,
            len(edges) / 100.0]


def global_state_features(simulator) -> list[float]:
    """Fixed-size summary of the complete loaded road--POI state."""
    roads = list(simulator.roads.values())
    pois = list(simulator.pois.values())
    road_count = max(1, len(roads))
    poi_count = max(1, len(pois))
    ratios = [(r.queue_vehicles + r.background_vehicles) /
              max(1.0, r.capacity_veh_per_slot) for r in roads]
    ratios.sort()
    def quantile(values, fraction):
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]
    poi_queue = [p.waiting / max(1.0, p.capacity_people) for p in pois]
    poi_occ = [p.occupancy / max(1.0, p.capacity_people) for p in pois]
    if getattr(simulator, "dynamics_mode", "reserved") == "event_driven":
        in_transit = {}
        soon = {}
        for traveler in simulator.travelers.values():
            if traveler.phase != "access" or traveler.status not in ("scheduled", "road_queue", "in_transit"):
                continue
            poi_id = traveler.request.destination_poi_id
            in_transit[poi_id] = in_transit.get(poi_id, 0.0) + traveler.request.people
            remaining = traveler.request.edge_ids[traveler.route_position:]
            remaining_slots = sum(simulator._travel_time(simulator.roads[edge_id], 0.0)
                                  for edge_id in remaining) / 900.0
            if remaining_slots <= 4:
                soon[poi_id] = soon.get(poi_id, 0.0) + traveler.request.people
        simulator.policy_in_transit_by_poi = in_transit
        simulator.policy_arriving_soon_by_poi = soon
    return [float(simulator.slot % 96) / 96.0,
            sum(ratios) / road_count, quantile(ratios, 0.5), quantile(ratios, 0.9),
            sum(value > 1.0 for value in ratios) / road_count,
            sum(r.queue_vehicles for r in roads) / road_count,
            sum(poi_queue) / poi_count, quantile(poi_queue, 0.9),
            sum(poi_occ) / poi_count]


def request_features(simulator, candidate: CandidateRoute) -> list[float]:
    """Request context shared by all candidates of one request."""
    request = candidate.request
    poi = simulator.pois[request.destination_poi_id]
    return [float(poi.category) / 100.0,
            float(request.dwell_slots) / 8.0,
            float(len(request.edge_ids)) / 100.0,
            float(len(request.egress_edge_ids)) / 100.0]


def policy_features(simulator, candidate: CandidateRoute,
                    global_features: list[float] | None = None) -> list[float]:
    """Concatenate global state, request context and candidate attributes."""
    features = (global_features if global_features is not None else global_state_features(simulator)) + request_features(simulator, candidate) + candidate_features(simulator, candidate)
    forecast = getattr(simulator, "policy_forecast_by_poi", None)
    if forecast is not None:
        poi_id = candidate.request.destination_poi_id
        total = max(1.0, float(getattr(simulator, "policy_forecast_total", 1.0)))
        assigned = getattr(simulator, "policy_batch_counts", {})
        # These optional features preserve the legacy 22-dimensional input
        # while exposing predictive demand only in explicitly marked scenarios.
        features += [min(1.0, float(forecast.get(poi_id, 0.0)) / 12.0),
                     min(1.0, total / 48.0),
                     min(1.0, float(assigned.get(poi_id, 0.0)) / 12.0)]
    if getattr(simulator, "dynamics_mode", "reserved") == "event_driven":
        poi_id = candidate.request.destination_poi_id
        incoming = getattr(simulator, "policy_in_transit_by_poi", {})
        soon = getattr(simulator, "policy_arriving_soon_by_poi", {})
        features += [min(1.0, incoming.get(poi_id, 0.0) / 12.0),
                     min(1.0, soon.get(poi_id, 0.0) / 12.0),
                     min(1.0, len(simulator.travelers) / 100.0)]
    if getattr(simulator, "policy_feature_schema", "legacy") == "local_v2":
        request = candidate.request
        poi = simulator.pois[request.destination_poi_id]
        edges = request.edge_ids + request.egress_edge_ids
        reserved = simulator.reserved.get(simulator.slot, {})
        committed = getattr(simulator, "provisional_edge_counts", {})
        edge_pressure = [
            (simulator.roads[edge_id].queue_vehicles
             + simulator.roads[edge_id].background_vehicles
             + reserved.get(edge_id, 0.0) + committed.get(edge_id, 0.0)
             + request.vehicles)
            / max(1.0, simulator.roads[edge_id].capacity_veh_per_slot)
            for edge_id in edges]
        incoming = getattr(simulator, "policy_in_transit_by_poi", {})
        service = max(1.0, poi.service_people_per_slot)
        features += [math.log1p(request.people) / 5.0,
                     math.log1p(request.vehicles) / 5.0,
                     math.log1p(poi.capacity_people) / 10.0,
                     math.log1p(poi.service_people_per_slot) / 10.0,
                     min(10.0, poi.wait_time_min / 60.0),
                     min(10.0, (poi.waiting + incoming.get(request.destination_poi_id, 0.0)
                                + request.people) / service) / 10.0,
                     min(10.0, max(edge_pressure, default=0.0)) / 10.0,
                     min(10.0, sum(edge_pressure) / max(1, len(edge_pressure))) / 10.0]
    return features


@dataclass
class LinearCandidatePolicy:
    """Softmax linear scorer trained from MPC expert choices."""

    weights: list[float]
    learning_rate: float = 0.05
    temperature: float = 1.0

    @classmethod
    def create(cls, feature_count: int = 9) -> "LinearCandidatePolicy":
        return cls([0.0] * feature_count)

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension does not match policy weights")
        return sum(w * x for w, x in zip(self.weights, features))

    def choose(self, simulator, candidates: Sequence[CandidateRoute]) -> CandidateRoute:
        if not candidates:
            raise ValueError("at least one candidate is required")
        return max(candidates, key=lambda c: self.score(candidate_features(simulator, c)))

    def choose_batch(self, simulator, candidate_sets: Sequence[Sequence[CandidateRoute]]) -> list[CandidateRoute]:
        # Re-score after each assignment so earlier actions affect later users.
        selected: list[CandidateRoute] = []
        for candidates in candidate_sets:
            if not candidates:
                continue
            selected.append(self.choose(simulator, candidates))
        return selected

    def fit(self, examples: Sequence[tuple[Sequence[float], int]], epochs: int = 10) -> dict[str, float]:
        """Fit a multiclass ranking model from (candidate features, label) pairs.

        Each example contains all candidate feature vectors followed by the
        index selected by MPC.  The update is softmax cross entropy.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive")
        loss = 0.0
        for _ in range(epochs):
            for features, label in examples:
                # ``features`` may be a flattened list for one candidate only;
                # use the compact binary ranking form in that case.
                if isinstance(features[0], (int, float)):
                    vectors = [features]
                else:
                    vectors = features
                scores = [self.score(vector) for vector in vectors]
                scale = max(self.temperature, 1e-6)
                exp_scores = [math.exp(min(50.0, s / scale)) for s in scores]
                denom = sum(exp_scores) or 1.0
                probabilities = [value / denom for value in exp_scores]
                if not 0 <= label < len(vectors):
                    raise ValueError("expert label is outside candidate set")
                loss -= math.log(max(1e-12, probabilities[label]))
                for index, vector in enumerate(vectors):
                    error = probabilities[index] - (1.0 if index == label else 0.0)
                    for j, value in enumerate(vector):
                        self.weights[j] -= self.learning_rate * error * value / scale
        count = max(1, len(examples) * epochs)
        return {"cross_entropy": loss / count, "examples": float(len(examples))}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": self.weights,
                                    "learning_rate": self.learning_rate,
                                    "temperature": self.temperature}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearCandidatePolicy":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls([float(x) for x in value["weights"]],
                   float(value.get("learning_rate", 0.05)),
                   float(value.get("temperature", 1.0)))
