"""Reproducible multi-strategy experiment runner for paper tables and plots.

The scenario supplies a common candidate set for every user. A future routing
index can generate that JSON without changing this runner. Every strategy gets
an independent simulator fork from the same initial state and receives the
same requests, observations, candidates, and random seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .model import CountModel
from .background_profile import DailyBackgroundProfile
from .recommender import CandidateRoute, MPCRecommender, RecommendationWeights
from .scair import ScairAdapter, ScairWeights
from .simulator import CoupledSimulator, POIState, Request, RoadState
from .metrics import GlobalLoadWeights, global_load
from .congestion import congestion_snapshot, congestion_episode
from .policy import global_state_features, policy_features
from .torch_policy import PPOCandidatePolicy


STRATEGIES = ("shortest", "shortest_distance", "shortest_queue", "individual", "scair", "myopic_joint", "rise_mpc", "rise_mpc_sampled", "ppo", "ppo_stress", "ppo_hotspot", "random",
              "no_road", "no_poi", "no_future")


@dataclass(frozen=True)
class ExperimentConfig:
    strategies: tuple[str, ...] = STRATEGIES
    horizon: int = 4
    discount: float = 0.95
    seed: int = 2025
    mpc_per_slot: int = 16
    write_csv: bool = True
    local_mpc: bool = False
    scair_travel_weight: float = 0.1
    slot_log_every: int = 0


def _background_profile_path(profile_path: str, model_dir: Path) -> Path:
    path = Path(profile_path)
    if path.is_file() or not path.is_absolute():
        return path
    # Transferred scenarios may retain an absolute path from the source machine.
    relocated = model_dir / path.parent.name / path.name
    return relocated if relocated.is_file() else path


def _load_simulator(scenario: dict, model_dir: Path) -> CoupledSimulator:
    roads = {int(row["edge_id"]): RoadState(**row) for row in scenario["roads"]}
    pois = {int(row["poi_id"]): POIState(**row) for row in scenario["pois"]}
    normalization = scenario.get("cost_normalization", {})
    cost_weights = GlobalLoadWeights(
        road=float(normalization.get("road_weight", 1.0)),
        poi=float(normalization.get("poi_weight", 1.0)),
        road_reference_vehicle_min=float(normalization.get("road_reference_vehicle_min", 1.0)),
        poi_reference_person_min=float(normalization.get("poi_reference_person_min", 1.0)),
        mode=str(normalization.get("mode", "vehicle_min")),
        road_top_k=int(normalization.get("road_top_k", 0)),
    )
    profile_path = scenario.get("background_profile_file")
    background_profile = (DailyBackgroundProfile.from_file(
        _background_profile_path(profile_path, model_dir), roads, pois)
                          if profile_path else None)
    simulator = CoupledSimulator(
        roads, pois,
        CountModel.load(model_dir / "road_background.json"),
        CountModel.load(model_dir / "poi_arrivals.json"),
        CountModel.load(model_dir / "poi_departures.json"),
        start_slot=int(scenario.get("start_slot", 0)),
        allow_over_capacity=bool(scenario.get("allow_over_capacity", False)),
        cost_weights=cost_weights,
        background_profile=background_profile,
        dynamics_mode=str(scenario.get("dynamics_mode", "event_driven"
                                       if scenario.get("background_profile_file") else "reserved")),
    )
    simulator.policy_feature_schema = scenario.get("policy_feature_schema", "legacy")
    return simulator


def _policy_feature_count(scenario: dict) -> int:
    return (22 + (3 if scenario.get("policy_use_forecast", False) else 0)
            + (3 if scenario.get("dynamics_mode", "event_driven"
                                 if scenario.get("background_profile_file") else "reserved") == "event_driven" else 0)
            + (8 if scenario.get("policy_feature_schema", "legacy") == "local_v2" else 0))


def apply_feature_schema(scenario: dict, schema: str) -> dict:
    if schema not in ("legacy", "local_v2"):
        raise ValueError(f"unsupported policy feature schema: {schema}")
    return {**scenario, "policy_feature_schema": schema}


def apply_cost_mode(scenario: dict, mode: str | None) -> dict:
    if mode is None:
        return scenario
    if mode != "queue_wait":
        raise ValueError(f"unsupported cost override: {mode}")
    updated = dict(scenario)
    existing = dict(scenario.get("cost_normalization", {}))
    road_reference = float(existing.get(
        "road_reference_vehicle_min",
        max(1.0, sum(float(row.get("capacity_veh_per_slot", 0.0))
                     for row in scenario.get("roads", []))
            * int(scenario.get("steps", 8)) * 15.0),
    ))
    poi_reference = float(existing.get(
        "poi_reference_person_min",
        max(1.0, sum(float(row.get("service_people_per_slot", 0.0))
                     for row in scenario.get("pois", []))
            * int(scenario.get("steps", 8)) * 15.0),
    ))
    updated["cost_normalization"] = {
        "mode": "queue_wait", "road_weight": 1.0, "poi_weight": 1.0,
        "road_reference_vehicle_min": road_reference,
        "poi_reference_person_min": poi_reference,
        "reference_basis": existing.get("reference_basis", "selected_network_capacity_minutes"),
    }
    updated["dynamics_mode"] = "event_driven"
    return updated


def _candidate_signature(scenario: dict) -> str:
    payload = [(str(row.get("request_id")), row.get("candidates") or [row])
               for row in scenario.get("requests", [])]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mpc_scenario_signature(scenario: dict) -> str:
    """Cover initial state, observations and request timing as well as candidates."""
    payload = {key: value for key, value in scenario.items()
               if key not in ("mpc_plan_file", "ppo_checkpoint", "write_object_timeseries",
                              "policy_feature_schema")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_mpc_expert_plan(scenario: dict, path: Path, horizon: int,
                         future_request_limit: int) -> dict:
    """Validate a complete action cache before replaying an expert trajectory.

    Legacy v1 plans have no scenario_signature; they still need matching
    candidates, MPC configuration and complete request/slot coverage.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported MPC cache schema: {path}")
    expected = {
        "horizon": horizon,
        "cost_mode": scenario.get("cost_normalization", {}).get("mode", "vehicle_min"),
        "dynamics_mode": scenario.get("dynamics_mode", "event_driven"
                                      if scenario.get("background_profile_file") else "reserved"),
        "candidate_signature": _candidate_signature(scenario),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"MPC cache {key} does not match scenario: {path}")
    # The forecaster caps demand at 16; limit=0 has the same effective bound.
    cached_limit = payload.get("future_request_limit")
    if not isinstance(cached_limit, int) or cached_limit < 0:
        raise ValueError(f"invalid MPC cache future_request_limit: {path}")
    effective_limit = lambda limit: min(16, limit) if limit > 0 else 16
    if effective_limit(cached_limit) != effective_limit(future_request_limit):
        raise ValueError(f"MPC cache future_request_limit does not match scenario: {path}")
    if ("scenario_signature" in payload and
            payload["scenario_signature"] != _mpc_scenario_signature(scenario)):
        raise ValueError(f"MPC cache initial state/observations/scenario do not match: {path}")
    expected_requests: dict[str, dict[str, int]] = defaultdict(dict)
    for row in scenario.get("requests", []):
        expected_requests[str(int(row["departure_slot"]))][str(row["request_id"])] = len(
            row.get("candidates") or [row])
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise ValueError(f"MPC cache has no action plan: {path}")
    if set(expected_requests) != {slot for slot, actions in plan.items() if actions}:
        raise ValueError(f"MPC cache slot coverage does not match scenario: {path}")
    for slot, requests in expected_requests.items():
        actions = plan.get(slot)
        if not isinstance(actions, dict) or set(actions) != set(requests):
            raise ValueError(f"MPC cache request coverage does not match slot {slot}: {path}")
        for request_id, count in requests.items():
            action = actions[request_id]
            if type(action) is not int or not 0 <= action < count:
                raise ValueError(f"invalid MPC cache action for request {request_id}: {path}")
    return payload


def find_mpc_expert_plan(cache: Path, scenario: dict, horizon: int,
                         future_request_limit: int) -> Path:
    """Find a matching plan independently of cache filenames and manifest paths."""
    if cache.is_file():
        load_mpc_expert_plan(scenario, cache, horizon, future_request_limit)
        return cache.resolve()
    if not cache.is_dir():
        raise FileNotFoundError(f"MPC cache directory not found: {cache}; generate training plans first")
    signature = _candidate_signature(scenario)
    errors = []
    matches = []
    for path in sorted(cache.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("candidate_signature") != signature:
            continue
        try:
            validated = load_mpc_expert_plan(scenario, path, horizon, future_request_limit)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        matches.append((path.resolve(), validated["plan"]))
    if not matches:
        if errors:
            raise ValueError("; ".join(errors))
        raise FileNotFoundError(
            f"no MPC cache matches this scenario's requests/candidates under {cache}; "
            "use plans generated from the training split")
    if any(plan != matches[0][1] for _, plan in matches[1:]):
        raise ValueError(f"conflicting MPC plans match the same scenario under {cache}")
    return matches[0][0]


def _observations_by_slot(scenario: dict) -> dict[int, dict]:
    # Profiled scenes use the ID-indexed daily schedule. Explicit overrides
    # represent shocks; legacy fixed observations must not shadow the profile.
    key = "profile_observations" if scenario.get("background_profile_file") else "observations"
    return {int(item["slot"]): item for item in scenario.get(key, [])}


def _request(row: dict, suffix: str = "") -> Request:
    return Request(
        request_id=str(row["request_id"]) + suffix,
        origin_node_id=int(row["origin_node_id"]),
        destination_poi_id=int(row["destination_poi_id"]),
        # User-facing requests are unit demand; people/vehicles are internal
        # simulator quantities and are intentionally not part of the query.
        people=1.0,
        edge_ids=tuple(map(int, row["edge_ids"])),
        vehicles=1.0,
        final_destination_node_id=(int(row["final_destination_node_id"])
                                   if row.get("final_destination_node_id") is not None else None),
        egress_edge_ids=tuple(map(int, row.get("egress_edge_ids", []))),
        dwell_slots=int(row.get("dwell_slots", 1)),
        category=(int(row["category"]) if row.get("category") is not None else None),
    )


def _candidate(row: dict, request_id: str, index: int) -> CandidateRoute:
    payload = dict(row)
    payload["request_id"] = request_id + f"-candidate-{index}"
    return CandidateRoute(_request(payload), str(row.get("label", f"candidate-{index}")))


def _route_minutes(simulator: CoupledSimulator, candidate: CandidateRoute) -> float:
    edges = candidate.request.edge_ids + candidate.request.egress_edge_ids
    total = 0.0
    for edge_id in edges:
        road = simulator.roads[edge_id]
        speed = max(5.0, road.speed_limit_kmh if road.speed_limit_kmh > 0 else road.speed_kmh)
        total += road.length_m * 3.6 / speed / 60.0
    return total


def _future_request_map(simulator: CoupledSimulator, horizon: int,
                        future_request_limit: int = 12) -> dict[int, tuple[Request, ...]]:
    """Project recently observed demand; never inspect unreleased requests."""
    result: dict[int, tuple[Request, ...]] = {}
    recent = getattr(simulator, "policy_recent_requests", ())
    # The projection is a bounded sample of observed demand, never a replay
    # of the complete next slot. In particular, legacy limit=0 stays bounded.
    recent = recent[:min(16, future_request_limit)] if future_request_limit > 0 else recent[:16]
    for relative in range(1, max(1, horizon)):
        if recent:
            result[relative] = tuple(Request(
                request_id=f"forecast-{simulator.slot}-{relative}-{index}",
                origin_node_id=item.origin_node_id,
                destination_poi_id=item.destination_poi_id,
                people=item.people, edge_ids=item.edge_ids, vehicles=item.vehicles,
                final_destination_node_id=item.final_destination_node_id,
                egress_edge_ids=item.egress_edge_ids, dwell_slots=item.dwell_slots,
                category=item.category)
                for index, item in enumerate(recent))
    return result


def _set_policy_context(simulator: CoupledSimulator, scenario: dict,
                        horizon: int) -> None:
    """Attach optional forecast/batch context used by predictive PPO scenes."""
    if not scenario.get("policy_use_forecast", False):
        return
    forecast: dict[int, float] = defaultdict(float)
    for request in getattr(simulator, "policy_recent_requests", ()):
        forecast[request.destination_poi_id] += max(0, horizon - 1)
    simulator.policy_forecast_by_poi = dict(forecast)
    simulator.policy_forecast_total = sum(forecast.values())
    if not hasattr(simulator, "policy_batch_counts"):
        simulator.policy_batch_counts = defaultdict(float)


def _choose_strategy(strategy: str, simulator: CoupledSimulator,
                    candidate_sets: Sequence[Sequence[CandidateRoute]],
                    config: ExperimentConfig,
                    learned_policy: PPOCandidatePolicy | None = None,
                    future_requests: Mapping[int, Sequence[Request]] | None = None
                    ) -> list[CandidateRoute]:
    if strategy == "rise_mpc" and getattr(simulator, "mpc_plan", None) is not None:
        slot_plan = simulator.mpc_plan.get(str(simulator.slot), {})
        selected: list[CandidateRoute] = []
        planning = simulator.fork()
        background = _provisional_background(planning)
        for candidates in candidate_sets:
            if not candidates:
                continue
            request_id = candidates[0].request.request_id.split("-candidate-", 1)[0]
            if request_id not in slot_plan:
                raise ValueError(f"MPC cache has no action for slot {simulator.slot}, request {request_id}")
            index = int(slot_plan[request_id])
            if not 0 <= index < len(candidates):
                raise ValueError(f"MPC cache candidate index is invalid for request {request_id}")
            chosen = candidates[index]
            selected.append(chosen)
            _register_provisional(planning, chosen.request, background)
        return selected
    if strategy in ("shortest", "shortest_distance"):
        return [min(candidates, key=lambda item: _route_minutes(simulator, item))
                for candidates in candidate_sets if candidates]
    if strategy == "shortest_queue":
        return [min(candidates, key=lambda item: (
                    simulator.pois[item.request.destination_poi_id].wait_time_min,
                    simulator.pois[item.request.destination_poi_id].waiting,
                    _route_minutes(simulator, item)))
                for candidates in candidate_sets if candidates]
    if strategy == "random":
        generator = random.Random((config.seed * 100_003) + simulator.slot)
        return [generator.choice(candidates) for candidates in candidate_sets if candidates]
    if strategy == "individual":
        return [min(candidates, key=lambda item: _route_minutes(simulator, item) +
                    simulator.pois[item.request.destination_poi_id].wait_time_min)
                for candidates in candidate_sets if candidates]
    if strategy == "scair":
        popularity = {poi_id: math.log1p(max(0.0, poi.arrival_history[-1] if poi.arrival_history else 0.0))
                      for poi_id, poi in simulator.pois.items()}
        return ScairAdapter(
            popularity=popularity,
            weights=ScairWeights(travel_weight=config.scair_travel_weight),
        ).choose(
            candidate_sets, simulator.pois,
            {edge_id: road.travel_time_sec for edge_id, road in simulator.roads.items()}
        )
    if strategy == "myopic_joint":
        decisions = MPCRecommender(horizon=1, discount=1.0).choose_batch(simulator, candidate_sets)
        return [decision.candidate for decision in decisions if decision.candidate is not None]
    if strategy in ("ppo", "ppo_stress", "ppo_hotspot"):
        if learned_policy is None:
            raise ValueError("ppo strategy requires a loaded policy")
        selected = []
        planning = simulator.fork()
        global_features = global_state_features(planning)
        background = _provisional_background(planning)
        for candidates in candidate_sets:
            if not candidates:
                continue
            vectors = [policy_features(planning, candidate, global_features)
                       for candidate in candidates]
            if hasattr(learned_policy, "choose"):
                action = learned_policy.choose(vectors, action_mask=[True] * len(vectors))
            else:
                action, _, _ = learned_policy.act(
                    vectors, deterministic=True, action_mask=[True] * len(vectors))
            selected.append(candidates[action])
            _register_provisional(planning, candidates[action].request, background)
            if getattr(planning, "policy_forecast_by_poi", None) is not None:
                planning.policy_batch_counts[candidates[action].request.destination_poi_id] += 1.0
        return selected
    if strategy == "rise_mpc_sampled":
        planning = simulator.fork()
        background = _provisional_background(planning)
        recommender = MPCRecommender(
            horizon=config.horizon, discount=config.discount,
            weights=RecommendationWeights(local_scope=config.local_mpc))
        sample_count = min(config.mpc_per_slot, len(candidate_sets))
        sampled = {(index * len(candidate_sets)) // sample_count
                   for index in range(sample_count)} if sample_count else set()
        selected = []
        for index, candidates in enumerate(candidate_sets):
            if not candidates:
                continue
            chosen = None
            if index in sampled:
                chosen = recommender.choose(planning, candidates, future_requests).candidate
            if chosen is None:
                chosen = min(candidates, key=lambda item: (
                    planning.pois[item.request.destination_poi_id].wait_time_min,
                    planning.pois[item.request.destination_poi_id].waiting,
                    _route_minutes(planning, item)))
            selected.append(chosen)
            _register_provisional(planning, chosen.request, background)
        return selected
    if strategy in ("rise_mpc", "no_road", "no_poi", "no_future"):
        weights = RecommendationWeights(
            road_externality=0.0 if strategy == "no_road" else 1.0,
            poi_externality=0.0 if strategy == "no_poi" else 1.0,
        )
        decisions = MPCRecommender(
            horizon=1 if strategy == "no_future" else config.horizon,
            discount=config.discount,
            weights=RecommendationWeights(
                user_travel=weights.user_travel, user_wait=weights.user_wait,
                road_externality=weights.road_externality,
                poi_externality=weights.poi_externality,
                unmet_penalty=weights.unmet_penalty,
                local_scope=config.local_mpc),
        ).choose_batch(simulator, candidate_sets,
                       None if strategy == "no_future" else future_requests)
        return [decision.candidate for decision in decisions if decision.candidate is not None]
    raise ValueError(f"unknown strategy: {strategy}")


def _write_csv(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_step(simulator: CoupledSimulator, selected: Sequence[CandidateRoute], **observations) -> tuple[list[CandidateRoute], dict]:
    """Execute a batch without allowing one infeasible route to abort a run."""
    trial = simulator.fork()
    try:
        result = trial.step([item.request for item in selected], **observations)
        simulator.__dict__.update(trial.__dict__)
        simulator.policy_recent_requests = tuple(item.request for item in selected)
        return list(selected), result
    except (KeyError, ValueError):
        feasible: list[CandidateRoute] = []
        for item in selected:
            trial = simulator.fork()
            try:
                trial.step([chosen.request for chosen in feasible] + [item.request], **observations)
            except (KeyError, ValueError):
                continue
            feasible.append(item)
        trial = simulator.fork()
        result = trial.step([item.request for item in feasible], **observations)
        simulator.__dict__.update(trial.__dict__)
        simulator.policy_recent_requests = tuple(item.request for item in feasible)
        return feasible, result


def _state_vector(simulator: CoupledSimulator, load: Mapping[str, float] | None = None) -> list[float]:
    """Compact global state stored with every PPO transition."""
    load = load or global_load(simulator)
    road_count = max(1, len(simulator.roads))
    poi_count = max(1, len(simulator.pois))
    mean_queue = sum(road.queue_vehicles for road in simulator.roads.values()) / road_count
    mean_occupancy = sum(poi.occupancy / max(1.0, poi.capacity_people)
                         for poi in simulator.pois.values()) / poi_count
    return [float(simulator.slot % 96) / 96.0, load["road_load"], load["poi_load"],
            load["total_load"], load["peak_vc"], load["peak_poi_utilization"],
            mean_queue, mean_occupancy]


def _provisional_background(simulator: CoupledSimulator) -> dict[int, float]:
    return {edge_id: simulator._background(road, simulator.slot)
            for edge_id, road in simulator.roads.items()}


def _register_provisional(simulator: CoupledSimulator, request: Request,
                          background: dict[int, float] | None = None) -> bool:
    """Register a batch action without advancing the real environment clock."""
    try:
        simulator._schedule(request, background if background is not None
                            else _provisional_background(simulator))
        return True
    except (KeyError, ValueError):
        return False


def _dagger_sample_indices(candidate_count: int, sample_count: int,
                           seed: int | None = None) -> set[int]:
    if not sample_count:
        return set()
    if seed is None:
        return {(index * candidate_count) // sample_count
                for index in range(sample_count)}
    sampler = random.Random(seed)
    return {
        sampler.randrange((index * candidate_count) // sample_count,
                          ((index + 1) * candidate_count) // sample_count)
        for index in range(sample_count)
    }


def collect_ppo_trajectory(scenario: dict, model_dir: Path,
                           policy: PPOCandidatePolicy,
                           deterministic: bool = False,
                           progress: Callable[[int, int, int], None] | None = None,
                           dagger_per_slot: int = 0,
                           dagger_horizon: int = 4,
                           dagger_future_limit: int | None = None,
                           dagger_sample_seed: int | None = None) -> list[dict]:
    """Collect one transition per simulator slot with its batch decisions."""
    if dagger_per_slot < 0:
        raise ValueError("dagger_per_slot must be nonnegative")
    simulator = _load_simulator(scenario, model_dir)
    requests_by_slot: dict[int, list[dict]] = defaultdict(list)
    candidates_by_request: dict[str, list[CandidateRoute]] = {}
    for row in scenario.get("requests", []):
        requests_by_slot[int(row["departure_slot"])].append(row)
        candidates = row.get("candidates") or [row]
        candidates_by_request[str(row["request_id"])] = [
            _candidate({**row, **candidate}, str(row["request_id"]), index)
            for index, candidate in enumerate(candidates)
        ]
    observations = _observations_by_slot(scenario)
    trajectory: list[dict] = []
    expert = MPCRecommender(horizon=dagger_horizon) if dagger_per_slot else None
    future_limit = (int(scenario.get("future_request_limit", 12))
                    if dagger_future_limit is None else dagger_future_limit)
    for step_index in range(int(scenario["steps"])):
        slot = simulator.slot
        before_load = global_load(simulator)
        state = _state_vector(simulator, before_load)
        request_rows = requests_by_slot.get(slot, [])
        candidate_sets = [candidates_by_request[str(row["request_id"])] for row in request_rows]
        chosen: list[CandidateRoute] = []
        decisions: list[dict] = []
        planning = simulator.fork()
        _set_policy_context(planning, scenario, int(scenario.get("forecast_horizon", 4)))
        planning.policy_batch_counts = defaultdict(float)
        global_features = global_state_features(planning)
        background = _provisional_background(planning)
        if expert is not None:
            future_requests = _future_request_map(simulator, dagger_horizon, future_limit)
            sample_count = min(dagger_per_slot, len(candidate_sets))
            sample_indices = _dagger_sample_indices(
                len(candidate_sets), sample_count,
                None if dagger_sample_seed is None
                else dagger_sample_seed + slot * 1_000_003,
            )
        else:
            future_requests = None
            sample_indices = set()
        online_experts: list[dict] = []
        for request_index, candidates in enumerate(candidate_sets):
            if not candidates:
                continue
            vectors = [policy_features(planning, candidate, global_features)
                       for candidate in candidates]
            action_mask = [True] * len(vectors)
            if expert is not None:
                if request_index in sample_indices:
                    # Label the state the student actually sees, including its
                    # earlier assignments in this slot.
                    expert_decision = expert.choose(planning.fork(), candidates, future_requests)
                    if expert_decision.candidate is not None:
                        expert_action = next(
                            index for index, candidate in enumerate(candidates)
                            if candidate.request.request_id == expert_decision.candidate.request.request_id)
                        online_experts.append({
                            "state": state, "features": vectors,
                            "action_mask": action_mask, "action": expert_action,
                            "request_id": candidates[0].request.request_id.split("-candidate-", 1)[0],
                            "slot": slot,
                        })
            if deterministic and hasattr(policy, "choose"):
                action = policy.choose(vectors, action_mask=action_mask)
                log_prob = 0.0
            else:
                action, log_prob, _ = policy.act(vectors, deterministic=deterministic,
                                                  action_mask=action_mask)
            chosen.append(candidates[action])
            decisions.append({"features": vectors, "action": action,
                              "old_log_prob": log_prob, "action_mask": action_mask})
            _register_provisional(planning, candidates[action].request, background)
            if getattr(planning, "policy_forecast_by_poi", None) is not None:
                planning.policy_batch_counts[candidates[action].request.destination_poi_id] += 1.0
        observed = observations.get(slot, {})
        observation_kwargs = {
            "observed_road_background": {int(k): float(v) for k, v in observed.get("road_background", {}).items()},
            "observed_poi_arrivals": {int(k): float(v) for k, v in observed.get("poi_arrivals", {}).items()},
            "observed_poi_departures": {int(k): float(v) for k, v in observed.get("poi_departures", {}).items()},
        }
        # Retain the paired no-new-action counterfactual for diagnosis, while
        # optimizing the same absolute system cost used by evaluation.
        baseline_simulator = simulator.fork()
        baseline_result = baseline_simulator.step([], **observation_kwargs)
        baseline_load = global_load(baseline_simulator, baseline_result)
        selected, result = _safe_step(
            simulator, chosen, **observation_kwargs,
        )
        after_load = global_load(simulator, result)
        incremental_load = after_load["total_load"] - baseline_load["total_load"]
        reward = -float(incremental_load)
        if len(selected) < len(chosen):
            reward -= 0.1 * (len(chosen) - len(selected))
        next_state = _state_vector(simulator, after_load)
        done = step_index == int(scenario["steps"]) - 1
        trajectory.append({"slot": slot, "state": state, "decisions": decisions,
                           "expert_examples": online_experts,
                           "reward": reward, "next_state": next_state,
                           "global_load": after_load,
                           "baseline_global_load": baseline_load,
                           "incremental_global_load": incremental_load,
                           "done": done})
        if progress is not None:
            progress(step_index + 1, int(scenario["steps"]), len(decisions))
    return trajectory


def collect_mpc_expert_dataset(scenario: dict, model_dir: Path,
                               horizon: int = 4, max_examples_per_slot: int | None = None,
                               future_request_limit: int | None = None,
                               progress: Callable[[int, int, int], None] | None = None,
                               label_progress: Callable[[int, int, int], None] | None = None,
                               mpc_plan_output: Path | None = None,
                               mpc_plan_input: Path | None = None) -> list[dict]:
    """Collect expert labels online or replay a validated MPC action cache.

    This is the pretraining stage only. It advances a separate simulator with
    MPC-selected actions, but does not update PPO parameters. Cached replay
    reconstructs features in request order without running the MPC recommender;
    legacy action-only caches provide hard labels without mpc_scores.
    """
    limit = (int(scenario.get("future_request_limit", 12))
             if future_request_limit is None else future_request_limit)
    cached = (load_mpc_expert_plan(scenario, mpc_plan_input, horizon, limit)
              if mpc_plan_input is not None else None)
    if cached is not None and mpc_plan_output is not None:
        raise ValueError("cached replay reads mpc_plan_input; omit mpc_plan_output")
    simulator = _load_simulator(scenario, model_dir)
    requests_by_slot: dict[int, list[dict]] = defaultdict(list)
    candidates_by_request: dict[str, list[CandidateRoute]] = {}
    for row in scenario.get("requests", []):
        requests_by_slot[int(row["departure_slot"])].append(row)
        candidates = row.get("candidates") or [row]
        candidates_by_request[str(row["request_id"])] = [
            _candidate({**row, **candidate}, str(row["request_id"]), index)
            for index, candidate in enumerate(candidates)
        ]
    observations = _observations_by_slot(scenario)
    examples: list[dict] = []
    mpc_plan: dict[str, dict[str, int]] = {}
    recommender = MPCRecommender(horizon=horizon) if cached is None else None
    for step_index in range(int(scenario["steps"])):
        slot = simulator.slot
        request_rows = requests_by_slot.get(slot, [])
        candidate_sets = [candidates_by_request[str(row["request_id"])] for row in request_rows]
        selected: list[CandidateRoute] = []
        planning = simulator.fork()
        _set_policy_context(planning, scenario, horizon)
        planning.policy_batch_counts = defaultdict(float)
        global_features = global_state_features(planning)
        background = _provisional_background(planning)
        future_requests = _future_request_map(simulator, horizon, limit) if cached is None else None
        sample_indices = set(range(len(candidate_sets)))
        if max_examples_per_slot is not None:
            if max_examples_per_slot < 1:
                raise ValueError("max_examples_per_slot must be positive")
            sample_count = min(max_examples_per_slot, len(candidate_sets))
            sample_indices = {(index * len(candidate_sets)) // sample_count
                              for index in range(sample_count)}
        plan_indices = set(range(len(candidate_sets))) if mpc_plan_output else sample_indices
        for request_index, candidates in enumerate(candidate_sets):
            if not candidates:
                continue
            chosen = None
            if cached is not None:
                base_id = candidates[0].request.request_id.split("-candidate-", 1)[0]
                expert_index = cached["plan"][str(slot)][base_id]
                chosen = candidates[expert_index]
                if request_index in sample_indices:
                    examples.append({
                        "state": _state_vector(simulator),
                        "features": [policy_features(planning, candidate, global_features)
                                     for candidate in candidates],
                        "action_mask": [True] * len(candidates),
                        "action": expert_index, "request_id": base_id, "slot": slot,
                    })
                    if label_progress is not None:
                        label_progress(step_index + 1, int(scenario["steps"]), len(examples))
            elif request_index in plan_indices:
                decision = recommender.choose(planning, candidates, future_requests)
                if decision.candidate is not None:
                    chosen = decision.candidate
                    score_by_key = {(item["label"], int(item["poi_id"])): float(item["score"])
                                    for item in decision.details}
                    scores = [score_by_key.get((candidate.label,
                                                candidate.request.destination_poi_id), float("inf"))
                              for candidate in candidates]
                    expert_index = next(i for i, candidate in enumerate(candidates)
                                        if candidate.request.request_id == chosen.request.request_id)
                    if request_index in sample_indices:
                        examples.append({
                        "state": _state_vector(simulator),
                        "features": [policy_features(planning, candidate, global_features)
                                     for candidate in candidates],
                        "action_mask": [True] * len(candidates),
                        "action": expert_index,
                        "mpc_scores": scores,
                        "request_id": candidates[0].request.request_id.split("-candidate-", 1)[0],
                        "slot": slot,
                        })
                        if label_progress is not None:
                            label_progress(step_index + 1, int(scenario["steps"]), len(examples))
                    if mpc_plan_output is not None:
                        base_id = candidates[0].request.request_id.split("-candidate-", 1)[0]
                        mpc_plan.setdefault(str(slot), {})[base_id] = int(
                            next(i for i, candidate in enumerate(candidates)
                                 if candidate.request.request_id == chosen.request.request_id))
            if chosen is None and max_examples_per_slot is not None:
                chosen = min(candidates, key=lambda item: (
                    planning.pois[item.request.destination_poi_id].wait_time_min,
                    planning.pois[item.request.destination_poi_id].waiting,
                    _route_minutes(planning, item)))
            if chosen is not None:
                selected.append(chosen)
                registered = _register_provisional(planning, chosen.request, background)
                if cached is not None and not registered:
                    raise ValueError(f"cached MPC action is infeasible at slot {slot}: {base_id}")
                planning.policy_batch_counts[chosen.request.destination_poi_id] += 1.0
        observed = observations.get(slot, {})
        executed, _ = _safe_step(
            simulator, selected,
            observed_road_background={int(k): float(v) for k, v in observed.get("road_background", {}).items()},
            observed_poi_arrivals={int(k): float(v) for k, v in observed.get("poi_arrivals", {}).items()},
            observed_poi_departures={int(k): float(v) for k, v in observed.get("poi_departures", {}).items()},
        )
        if cached is not None and len(executed) != len(selected):
            raise ValueError(f"cached MPC batch could not be replayed at slot {slot}")
        if progress is not None:
            progress(step_index + 1, int(scenario["steps"]), len(examples))
    if mpc_plan_output is not None:
        mpc_plan_output.parent.mkdir(parents=True, exist_ok=True)
        mpc_plan_output.write_text(json.dumps({
            "schema_version": 1, "horizon": horizon,
            "future_request_limit": limit,
            "cost_mode": scenario.get("cost_normalization", {}).get("mode", "vehicle_min"),
            "dynamics_mode": scenario.get("dynamics_mode", "event_driven"
                                          if scenario.get("background_profile_file") else "reserved"),
            "candidate_signature": _candidate_signature(scenario),
            "scenario_signature": _mpc_scenario_signature(scenario),
            "plan": mpc_plan,
        }, separators=(",", ":")), encoding="utf-8")
    return examples


def train_ppo_scenarios(scenarios: Sequence[Path], model_dir: Path,
                        output_model: Path, epochs: int = 10,
                        updates: int = 1, seed: int = 0,
                        validation_scenarios: Sequence[Path] = (),
                        rollout_size: int = 2048, batch_size: int = 64,
                        gamma: float = 0.95, init_model: Path | None = None,
                        expert_examples: Sequence[dict] = (), bc_coef: float = 0.1,
                        dagger_per_slot: int = 0, dagger_horizon: int = 4,
                        dagger_future_limit: int | None = None,
                        dagger_bc_coef: float = 0.5,
                        learning_rate: float = 3e-4,
                        clip_ratio: float = 0.2,
                        value_coef: float = 0.5,
                        entropy_coef: float = 0.01,
                        progress: Callable[[dict], None] | None = None,
                        slot_progress: Callable[[str, int, int, int, int], None] | None = None,
                        cost_mode: str | None = None,
                        feature_schema: str = "legacy") -> list[dict]:
    """Train on complete passes over the scenarios before each PPO update."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if rollout_size < 1:
        raise ValueError("rollout_size must be positive")
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    if (dagger_per_slot < 0 or dagger_bc_coef < 0 or learning_rate <= 0
            or clip_ratio <= 0 or value_coef < 0 or entropy_coef < 0):
        raise ValueError("invalid PPO or DAgger coefficient")
    # Torch seeding is optional at import time but available here because this
    # function explicitly requests the neural training path.
    # Seed before module construction so initial weights are reproducible.
    # PPOCandidatePolicy imports torch lazily but exposes it after creation.
    import torch
    torch.manual_seed(seed)
    dimensions = {_policy_feature_count(apply_feature_schema(apply_cost_mode(
        json.loads(Path(path).read_text(encoding="utf-8")), cost_mode), feature_schema))
                  for path in scenarios}
    if len(dimensions) != 1:
        raise ValueError("training scenarios must use one policy feature schema")
    feature_count = dimensions.pop()
    if any(len(vector) != feature_count for item in expert_examples
           for vector in item["features"]):
        raise ValueError("expert dataset feature schema does not match training scenarios")
    policy = PPOCandidatePolicy(
        feature_count=feature_count, clip_ratio=clip_ratio,
        value_coef=value_coef, entropy_coef=entropy_coef)
    if init_model is not None:
        policy.load(init_model)
    reports: list[dict] = []
    best_validation = float("inf")
    generator = random.Random(seed)
    metrics_path = output_model.with_name("training_metrics.json")

    def validate(update_number: int) -> dict:
        totals = {}
        steps = 0
        reward_sum = 0.0
        for scenario_path in validation_scenarios:
            scenario = apply_feature_schema(apply_cost_mode(
                json.loads(Path(scenario_path).read_text(encoding="utf-8")), cost_mode),
                feature_schema)
            trajectory = collect_ppo_trajectory(
                scenario, model_dir, policy, deterministic=True,
                progress=(lambda done, total, count: slot_progress(
                    "validation", update_number, done, total, count))
                if slot_progress else None)
            totals[str(scenario_path)] = sum(
                float(item["global_load"]["total_load"]) for item in trajectory)
            steps += len(trajectory)
            reward_sum += sum(float(item["reward"]) for item in trajectory)
        total_load = sum(totals.values())
        return {"validation_reward": reward_sum / max(1, steps),
                "validation_mean_absolute_load": total_load / max(1, steps),
                "validation_total_load": total_load,
                "validation_load_by_scenario": totals}

    baseline = None
    if init_model is not None and validation_scenarios:
        baseline = validate(0)
        best_validation = baseline["validation_total_load"]
        policy.save(output_model.with_name(output_model.stem + ".best" + output_model.suffix))
        metrics_path.write_text(json.dumps({"baseline": baseline, "updates": []}, indent=2),
                                encoding="utf-8")
    for update in range(updates):
        transitions: list[dict] = []
        dagger_examples: list[dict] = []
        episodes = 0
        decision_count = 0
        scenario_episodes: dict[str, int] = defaultdict(int)
        while decision_count < rollout_size:
            # Finish the whole pass even when an early episode exceeds the
            # decision threshold. Every update then covers every scene equally.
            for scenario_path in generator.sample(list(scenarios), len(scenarios)):
                episodes += 1
                scenario_episodes[str(scenario_path)] += 1
                scenario = apply_feature_schema(apply_cost_mode(
                    json.loads(Path(scenario_path).read_text(encoding="utf-8")), cost_mode),
                    feature_schema)
                episode = collect_ppo_trajectory(
                    scenario, model_dir, policy,
                    progress=(lambda done, total, count: slot_progress(
                        "train", update + 1, done, total, count)) if slot_progress else None,
                    dagger_per_slot=dagger_per_slot,
                    dagger_horizon=dagger_horizon,
                    dagger_future_limit=dagger_future_limit,
                    dagger_sample_seed=seed + update * 1_000_003 + episodes)
                dagger_examples.extend(item for slot in episode
                                       for item in slot.get("expert_examples", ()))
                transitions.extend(episode)
                decision_count += sum(len(slot["decisions"]) for slot in episode)
            if decision_count == 0:
                raise ValueError("scenarios produced no requests with valid candidates")
        if not transitions:
            raise ValueError("scenarios produced no requests with valid candidates")
        report = policy.update(
            transitions, epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, gamma=gamma,
            expert_examples=expert_examples, bc_coef=bc_coef,
            dagger_examples=dagger_examples, dagger_bc_coef=dagger_bc_coef,
            progress=(lambda done, total: slot_progress(
                "optimize", update + 1, done, total, decision_count))
            if slot_progress else None)
        report.update({"update": update, "transitions": decision_count,
                       "dagger_examples": len(dagger_examples),
                       "environment_steps": len(transitions), "episodes": episodes,
                       "scenario_episodes": dict(scenario_episodes),
                       "feature_count": feature_count,
                       "cost_mode": cost_mode or "scenario"})
        if validation_scenarios:
            report.update(validate(update + 1))
            if report["validation_total_load"] < best_validation:
                best_validation = report["validation_total_load"]
                policy.save(output_model.with_name(output_model.stem + ".best" + output_model.suffix))
        policy.save(output_model.with_name(
            f"{output_model.stem}.update_{update + 1:03d}{output_model.suffix}"))
        reports.append(report)
        metrics_path.write_text(json.dumps({"baseline": baseline, "updates": reports}, indent=2),
                                encoding="utf-8")
        if progress is not None:
            progress(report)
    policy.save(output_model)
    return reports


def run_experiment(scenario: dict, model_dir: Path, output: Path,
                   config: ExperimentConfig) -> dict:
    unknown = set(config.strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    requests_by_slot: dict[int, list[dict]] = defaultdict(list)
    for row in scenario.get("requests", []):
        requests_by_slot[int(row["departure_slot"])].append(row)
    candidates_by_request: dict[str, list[CandidateRoute]] = {}
    for row in scenario.get("requests", []):
        candidates = row.get("candidates")
        if not candidates:
            candidates = [row]
        candidates_by_request[str(row["request_id"])] = [
            _candidate({**row, **candidate}, str(row["request_id"]), index)
            for index, candidate in enumerate(candidates)
        ]
    future_request_limit = int(scenario.get("future_request_limit", 12))
    assignments: list[dict] = []
    timeseries: list[dict] = []
    poi_series: list[dict] = []
    road_series: list[dict] = []
    summaries: list[dict] = []
    observations = _observations_by_slot(scenario)
    mpc_plan = None
    plan_path = scenario.get("mpc_plan_file")
    if plan_path and "rise_mpc" in config.strategies:
        path = Path(plan_path)
        if not path.is_file():
            path = output.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"MPC plan cache not found: {plan_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported MPC plan cache schema")
        if int(payload.get("horizon", config.horizon)) != config.horizon:
            raise ValueError("MPC cache horizon does not match evaluation horizon")
        expected_mode = scenario.get("cost_normalization", {}).get("mode", "vehicle_min")
        if payload.get("cost_mode") != expected_mode:
            raise ValueError("MPC cache cost mode does not match evaluation scenario")
        expected_dynamics = scenario.get("dynamics_mode", "reserved")
        if payload.get("dynamics_mode") != expected_dynamics:
            raise ValueError("MPC cache dynamics mode does not match evaluation scenario")
        if payload.get("candidate_signature") != _candidate_signature(scenario):
            raise ValueError("MPC cache candidate set does not match evaluation scenario")
        mpc_plan = payload["plan"]
    learned_policy = None
    if "ppo" in config.strategies or "ppo_stress" in config.strategies or "ppo_hotspot" in config.strategies:
        # A benchmark may pin its checkpoint in the scenario manifest.  This
        # keeps evaluation reproducible without replacing the default NYC
        # checkpoint selection used by earlier experiments.
        scenario_checkpoint = scenario.get("ppo_checkpoint")
        custom_checkpoint = Path(scenario_checkpoint) if scenario_checkpoint else None
        if custom_checkpoint:
            resolved = (custom_checkpoint if custom_checkpoint.is_file()
                        else model_dir / custom_checkpoint)
            if not resolved.is_file():
                raise FileNotFoundError(f"PPO checkpoint not found: {scenario_checkpoint}")
            candidates = (resolved,)
        elif "ppo_hotspot" in config.strategies:
            candidates = (model_dir / "grid_hotspot_training_20261013" /
                          "global_policy_ppo_hotspot.best.pt",)
        elif "ppo_stress" in config.strategies:
            candidates = (model_dir / "ppo_stress_training_v2_20261013" /
                          "global_policy_ppo_stress.best.pt",)
        else:
            candidates = (
                model_dir / "ppo_training_stage2_20261012" / "global_policy_ppo_stage2.best.pt",
                model_dir / "ppo_training_stage2_20261012" / "global_policy_ppo_stage2.pt",
            )
        checkpoint = next((path for path in candidates if path.exists()), None)
        if checkpoint is None:
            raise FileNotFoundError("no PPO checkpoint found; expected stage2 best model")
        feature_count = _policy_feature_count(scenario)
        learned_policy = PPOCandidatePolicy(feature_count=feature_count)
        learned_policy.load(checkpoint)
    for strategy in config.strategies:
        simulator = _load_simulator(scenario, model_dir)
        if mpc_plan is not None:
            simulator.mpc_plan = mpc_plan
        strategy_start = time.perf_counter()
        user_rows: list[dict] = []
        previous_total_load: float | None = None
        cumulative_eval_requests = 0
        cumulative_eval_load = 0.0
        for step_index in range(int(scenario["steps"])):
            slot_start = time.perf_counter()
            slot = simulator.slot
            request_rows = requests_by_slot.get(slot, [])
            candidate_sets = [candidates_by_request[str(row["request_id"])] for row in request_rows]
            future_requests = _future_request_map(simulator, config.horizon,
                                                  future_request_limit)
            # Forecast and provisional batch counts are visible only to the
            # explicitly forecast-enabled PPO policy.
            _set_policy_context(simulator, scenario, config.horizon)
            simulator.policy_batch_counts = defaultdict(float)
            selected = _choose_strategy(strategy, simulator, candidate_sets, config,
                                        learned_policy, future_requests)
            observed = observations.get(slot, {})
            observation_kwargs = {
                "observed_road_background": {int(k): float(v) for k, v in observed.get("road_background", {}).items()},
                "observed_poi_arrivals": {int(k): float(v) for k, v in observed.get("poi_arrivals", {}).items()},
                "observed_poi_departures": {int(k): float(v) for k, v in observed.get("poi_departures", {}).items()},
            }
            no_action = simulator.fork()
            no_action_result = no_action.step([], **observation_kwargs)
            selected, result = _safe_step(
                simulator, selected, **observation_kwargs,
            )
            selected_by_id = {item.request.request_id.rsplit("-candidate-", 1)[0]: item for item in selected}
            load = global_load(simulator, result)
            no_action_load = global_load(no_action, no_action_result)
            if scenario.get("write_object_timeseries", False):
                for poi_id, state in result["pois"].items():
                    poi = simulator.pois[poi_id]
                    poi_series.append({"strategy": strategy, "slot": slot,
                                       "poi_id": poi_id, "category": poi.category,
                                       "capacity_people": poi.capacity_people,
                                       "service_people_per_slot": poi.service_people_per_slot,
                                       **state})
                for edge_id, state in result["roads"].items():
                    road_series.append({"strategy": strategy, "slot": slot,
                                        "edge_id": edge_id, **state})
            road_delay = load["road_travel_delay_vehicle_min"]
            poi_queue = sum(float(state["waiting"]) for state in result["pois"].values())
            road_queue = sum(float(state["queue_vehicles"]) for state in result["roads"].values())
            poi_occupancy = sum(float(state["occupancy"]) for state in result["pois"].values())
            congestion = congestion_snapshot(simulator, result)
            road_count = max(1, len(result["roads"]))
            poi_count = max(1, len(result["pois"]))
            road_vc = [float(state["vc_ratio"]) for state in result["roads"].values()]
            poi_utilization = [
                float(state["occupancy"]) / max(1.0, float(simulator.pois[int(poi_id)].capacity_people))
                for poi_id, state in result["pois"].items()]
            selected_count = len(selected)
            slot_requests = len(request_rows)
            recommendation_count = sum(bool(candidates) for candidates in candidate_sets)
            no_candidate_count = slot_requests - recommendation_count
            failed_count = max(0, recommendation_count - selected_count)
            route_times = [_route_minutes(simulator, item) for item in selected]
            timeseries.append({"strategy": strategy, "slot": slot,
                               **congestion,
                               "road_congestion_ratio": congestion["congested_road_edges"] / road_count,
                               "poi_congestion_ratio": congestion["congested_pois"] / poi_count,
                               "mean_vc_ratio": sum(road_vc) / len(road_vc) if road_vc else 0.0,
                               "max_vc_ratio": max(road_vc, default=0.0),
                               "mean_poi_utilization": sum(poi_utilization) / len(poi_utilization) if poi_utilization else 0.0,
                               "max_poi_utilization": max(poi_utilization, default=0.0),
                               "max_road_queue_vehicles": max((float(state["queue_vehicles"])
                                                                for state in result["roads"].values()), default=0.0),
                               "max_poi_waiting_people": max((float(state["waiting"])
                                                               for state in result["pois"].values()), default=0.0),
                               "requests": slot_requests,
                               "recommendation_requests": recommendation_count,
                               "executed_requests": selected_count,
                               "failed_requests": failed_count,
                               "no_candidate_requests": no_candidate_count,
                               "execution_rate": selected_count / max(1, recommendation_count),
                               # Backward-compatible aliases used by older reports.
                               "selected_requests": selected_count,
                               "unselected_requests": failed_count + no_candidate_count,
                               "selection_rate": selected_count / max(1, slot_requests),
                               "mean_route_time_min": sum(route_times) / len(route_times) if route_times else 0.0,
                               "slot_runtime_sec": time.perf_counter() - slot_start,
                               "road_delay_min": road_delay, "poi_queue_people": poi_queue,
                               "road_queue_vehicles": road_queue,
                               "road_inflow_vehicles": sum(float(state["total_inflow_vehicles"]) for state in result["roads"].values()),
                               "road_outflow_vehicles": sum(float(state["outflow_vehicles"]) for state in result["roads"].values()),
                               "poi_occupancy_people": poi_occupancy,
                               "poi_arrivals_people": sum(float(state["arrivals"]) for state in result["pois"].values()),
                               "poi_departures_people": sum(float(state["departures"]) for state in result["pois"].values()),
                               "poi_admitted_people": sum(float(state["admitted"]) for state in result["pois"].values()),
                               "active_road_edges": sum(1 for state in result["roads"].values()
                                                         if state["vc_ratio"] > 1.0),
                               "global_road_load": load["road_load"],
                               "global_poi_load": load["poi_load"],
                               "global_total_load": load["total_load"],
                               "road_queue_vehicle_min": load["road_queue_vehicle_min"],
                               "road_travel_delay_vehicle_min": load["road_travel_delay_vehicle_min"],
                               "poi_wait_person_min": load["poi_wait_person_min"],
                               "raw_road_vehicle_min": load["raw_road_vehicle_min"],
                               "raw_topk_road_vehicle_min": load["raw_topk_road_vehicle_min"],
                               "raw_poi_person_min": load["raw_poi_person_min"],
                               "incremental_global_total_load": load["total_load"] - no_action_load["total_load"],
                               "incremental_road_load": load["road_load"] - no_action_load["road_load"],
                               "incremental_poi_load": load["poi_load"] - no_action_load["poi_load"],
                               "global_total_load_delta": (load["total_load"] - previous_total_load
                                                            if previous_total_load is not None else 0.0),
                               "peak_vc": load["peak_vc"],
                               "peak_poi_utilization": load["peak_poi_utilization"]})
            previous_total_load = load["total_load"]
            cumulative_eval_requests += slot_requests
            cumulative_eval_load += load["total_load"]
            if (config.slot_log_every > 0 and
                    ((step_index + 1) % config.slot_log_every == 0 or
                     step_index + 1 == int(scenario["steps"]))):
                print(
                    f"[eval-progress] strategy={strategy} slot={step_index + 1}/"
                    f"{int(scenario['steps'])} requests={cumulative_eval_requests} "
                    f"cumulative_total_load={cumulative_eval_load:.6f}",
                    flush=True,
                )
            for row in request_rows:
                chosen = selected_by_id.get(str(row["request_id"]))
                poi_id = chosen.request.destination_poi_id if chosen else ""
                assignment = {"strategy": strategy, "slot": slot,
                              "request_id": row["request_id"], "selected": bool(chosen),
                              "poi_id": poi_id, "label": chosen.label if chosen else "",
                              "candidate_count": len(candidates_by_request[str(row["request_id"])]),
                              "route_time_min": _route_minutes(simulator, chosen) if chosen else ""}
                assignments.append(assignment)
                user_rows.append(assignment)
        elapsed = time.perf_counter() - strategy_start
        strategy_series = [row for row in timeseries if row["strategy"] == strategy]
        selected_times = sorted(float(row["route_time_min"]) for row in user_rows if row["selected"])
        p95_index = math.ceil(0.95 * len(selected_times)) - 1
        poi_loads = [sum(row["poi_id"] == poi_id for row in user_rows if row["selected"])
                     for poi_id in simulator.pois]
        ordered_loads = sorted(poi_loads)
        total_load = sum(ordered_loads)
        gini = (sum((2 * index - len(ordered_loads) - 1) * load
                    for index, load in enumerate(ordered_loads, 1)) /
                (len(ordered_loads) * total_load)) if total_load else 0.0
        summary = {"strategy": strategy, "steps": len(strategy_series),
                   "requests": len(user_rows),
                   "completed_requests": (sum(
                       traveler.status == "complete" for traveler in simulator.travelers.values())
                       if simulator.dynamics_mode == "event_driven" else None),
                   "unfinished_requests": (sum(
                       traveler.status != "complete" for traveler in simulator.travelers.values())
                       if simulator.dynamics_mode == "event_driven" else None),
                   **congestion_episode(strategy_series),
                   "mean_road_delay_min": sum(row["road_delay_min"] for row in strategy_series) / max(1, len(strategy_series)),
                   "mean_poi_queue_people": sum(row["poi_queue_people"] for row in strategy_series) / max(1, len(strategy_series)),
                   "mean_active_congested_edges": sum(row["active_road_edges"] for row in strategy_series) / max(1, len(strategy_series)),
                   "mean_road_congestion_ratio": sum(row["road_congestion_ratio"] for row in strategy_series) / max(1, len(strategy_series)),
                   "mean_poi_congestion_ratio": sum(row["poi_congestion_ratio"] for row in strategy_series) / max(1, len(strategy_series)),
                   "peak_road_congestion_ratio": max((row["road_congestion_ratio"] for row in strategy_series), default=0.0),
                   "peak_poi_congestion_ratio": max((row["poi_congestion_ratio"] for row in strategy_series), default=0.0),
                   "mean_vc_ratio": sum(row["mean_vc_ratio"] for row in strategy_series) / max(1, len(strategy_series)),
                   "peak_vc_ratio": max((row["max_vc_ratio"] for row in strategy_series), default=0.0),
                   "mean_poi_utilization": sum(row["mean_poi_utilization"] for row in strategy_series) / max(1, len(strategy_series)),
                   "peak_poi_utilization_observed": max((row["max_poi_utilization"] for row in strategy_series), default=0.0),
                   "max_road_queue_vehicles": max((row["max_road_queue_vehicles"] for row in strategy_series), default=0.0),
                   "max_poi_waiting_people": max((row["max_poi_waiting_people"] for row in strategy_series), default=0.0),
                   "mean_absolute_load_change": sum(abs(float(row["global_total_load_delta"])) for row in strategy_series) / max(1, len(strategy_series)),
                   "peak_absolute_load_change": max((abs(float(row["global_total_load_delta"])) for row in strategy_series), default=0.0),
                   "selected_requests": sum(row["selected_requests"] for row in strategy_series),
                   "selection_rate": sum(row["selected_requests"] for row in strategy_series) / max(1, sum(row["requests"] for row in strategy_series)),
                   "recommendation_requests": sum(row["recommendation_requests"] for row in strategy_series),
                   "executed_requests": sum(row["executed_requests"] for row in strategy_series),
                   "failed_requests": sum(row["failed_requests"] for row in strategy_series),
                   "no_candidate_requests": sum(row["no_candidate_requests"] for row in strategy_series),
                   "execution_rate": sum(row["executed_requests"] for row in strategy_series) / max(1, sum(row["recommendation_requests"] for row in strategy_series)),
                   "mean_route_time_min": sum(row["mean_route_time_min"] for row in strategy_series) / max(1, len(strategy_series)),
                   "mean_slot_runtime_ms": 1000.0 * sum(row["slot_runtime_sec"] for row in strategy_series) / max(1, len(strategy_series)),
                   "p95_slot_runtime_ms": (1000.0 * sorted(row["slot_runtime_sec"] for row in strategy_series)[
                       max(0, math.ceil(0.95 * len(strategy_series)) - 1)]
                       if strategy_series else 0.0),
                   "requests_per_second": len(user_rows) / max(elapsed, 1e-9),
                   "cumulative_road_delay_min": sum(row["road_delay_min"] for row in strategy_series),
                   "cumulative_poi_queue_people_slots": sum(row["poi_queue_people"] for row in strategy_series),
                   "cumulative_global_road_load": sum(row["global_road_load"] for row in strategy_series),
                   "cumulative_global_poi_load": sum(row["global_poi_load"] for row in strategy_series),
                   "cumulative_global_total_load": sum(row["global_total_load"] for row in strategy_series),
                   "cumulative_road_queue_vehicle_min": sum(row["road_queue_vehicle_min"] for row in strategy_series),
                   "cumulative_road_travel_delay_vehicle_min": sum(row["road_travel_delay_vehicle_min"] for row in strategy_series),
                   "cumulative_poi_wait_person_min": sum(row["poi_wait_person_min"] for row in strategy_series),
                   "cumulative_raw_road_vehicle_min": sum(row["raw_road_vehicle_min"] for row in strategy_series),
                   "cumulative_raw_topk_road_vehicle_min": sum(row["raw_topk_road_vehicle_min"] for row in strategy_series),
                   "cumulative_raw_poi_person_min": sum(row["raw_poi_person_min"] for row in strategy_series),
                   "cumulative_incremental_global_total_load": sum(row["incremental_global_total_load"] for row in strategy_series),
                   "cumulative_incremental_road_load": sum(row["incremental_road_load"] for row in strategy_series),
                   "cumulative_incremental_poi_load": sum(row["incremental_poi_load"] for row in strategy_series),
                   "peak_vc": max((row["peak_vc"] for row in strategy_series), default=0.0),
                   "peak_poi_utilization": max((row["peak_poi_utilization"] for row in strategy_series), default=0.0),
                   "p95_free_flow_route_min": selected_times[p95_index] if selected_times else None,
                   "poi_assignment_gini": gini,
                   "runtime_sec": elapsed}
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    if config.write_csv:
        _write_csv(output / "assignments.csv", assignments,
               ("strategy", "slot", "request_id", "selected", "poi_id", "label", "candidate_count", "route_time_min"))
        _write_csv(output / "timeseries.csv", timeseries,
               ("strategy", "slot", "congested_road_edges", "congested_pois",
                "road_congestion_ratio", "poi_congestion_ratio", "mean_vc_ratio", "max_vc_ratio",
                "mean_poi_utilization", "max_poi_utilization", "max_road_queue_vehicles",
                "max_poi_waiting_people", "requests", "recommendation_requests", "executed_requests",
                "failed_requests", "no_candidate_requests", "execution_rate", "selected_requests",
                "unselected_requests", "selection_rate", "mean_route_time_min", "slot_runtime_sec",
                "global_total_load_delta",
                "road_delay_min", "poi_queue_people",
                "road_queue_vehicles", "road_inflow_vehicles", "road_outflow_vehicles",
                "poi_occupancy_people", "poi_arrivals_people",
                "poi_departures_people", "poi_admitted_people", "active_road_edges",
                "global_road_load", "global_poi_load", "global_total_load",
                "road_queue_vehicle_min", "road_travel_delay_vehicle_min", "poi_wait_person_min",
                "raw_road_vehicle_min", "raw_topk_road_vehicle_min", "raw_poi_person_min",
                "incremental_global_total_load", "incremental_road_load", "incremental_poi_load",
                "peak_vc", "peak_poi_utilization"))
        if poi_series:
            _write_csv(output / "per_poi_timeseries.csv", poi_series,
                   ("strategy", "slot", "poi_id", "category", "capacity_people",
                    "service_people_per_slot", "base_arrivals", "recommendation_arrivals",
                    "scheduled_recommendation_departures", "arrivals", "departures",
                    "admitted", "occupancy", "waiting", "wait_time_min"))
        if road_series:
            _write_csv(output / "per_road_timeseries.csv", road_series,
                   ("strategy", "slot", "edge_id", "background_vehicles",
                    "recommended_vehicles", "total_inflow_vehicles", "total_load_vehicles",
                    "outflow_vehicles", "queue_vehicles", "capacity_vehicles",
                    "vc_ratio", "speed_kmh", "travel_time_sec"))
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    performance = [{
        "strategy": row["strategy"],
        "runtime_sec": row["runtime_sec"],
        "requests": row["requests"],
        "requests_per_second": row["requests_per_second"],
        "recommendation_requests": row["recommendation_requests"],
        "executed_requests": row["executed_requests"],
        "failed_requests": row["failed_requests"],
        "no_candidate_requests": row["no_candidate_requests"],
        "execution_rate": row["execution_rate"],
        "selected_requests": row["selected_requests"],
        "selection_rate": row["selection_rate"],
        "mean_route_time_min": row["mean_route_time_min"],
        "p95_free_flow_route_min": row["p95_free_flow_route_min"],
        "mean_slot_runtime_ms": row["mean_slot_runtime_ms"],
        "p95_slot_runtime_ms": row["p95_slot_runtime_ms"],
        "mean_road_congestion_ratio": row["mean_road_congestion_ratio"],
        "peak_road_congestion_ratio": row["peak_road_congestion_ratio"],
        "mean_poi_congestion_ratio": row["mean_poi_congestion_ratio"],
        "peak_poi_congestion_ratio": row["peak_poi_congestion_ratio"],
        "cumulative_global_total_load": row["cumulative_global_total_load"],
    } for row in summaries]
    (output / "performance_metrics.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "strategies": config.strategies, "horizon": config.horizon,
        "discount": config.discount, "seed": config.seed,
        "provenance": "same scenario, candidates, initial state, and simulator per strategy",
        "cost_normalization": scenario.get("cost_normalization", {}),
        "background_profile_file": scenario.get("background_profile_file"),
        "future_request_limit": int(scenario.get("future_request_limit", 12)),
        "mpc_per_slot": config.mpc_per_slot if "rise_mpc_sampled" in config.strategies else None,
        "write_csv": config.write_csv,
        "local_mpc": config.local_mpc,
    }, indent=2), encoding="utf-8")
    return {"summary": summaries, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--mpc-plan", type=Path,
                        help="precomputed exact MPC plan; skips online MPC rollouts")
    parser.add_argument("--background-profile", type=Path,
                        help="ID-indexed daily background; legacy observations are ignored")
    parser.add_argument("--cost-mode", choices=("queue_wait",),
                        help="use event-driven dynamics and physical waiting-minute cost")
    parser.add_argument("--future-request-limit", type=int,
                        help="observed-demand forecast sample size per future slot (capped at 16)")
    parser.add_argument("--mpc-per-slot", type=int, default=16,
                        help="current requests evaluated by rise_mpc_sampled per slot")
    parser.add_argument("--compact-output", action="store_true",
                        help="write summary and manifest only")
    parser.add_argument("--local-mpc", action="store_true",
                        help="scope MPC load aggregation to candidate routes")
    args = parser.parse_args()
    scenario = apply_cost_mode(json.loads(args.scenario.read_text(encoding="utf-8")),
                               args.cost_mode)
    if args.ppo_checkpoint is not None:
        scenario["ppo_checkpoint"] = str(args.ppo_checkpoint)
    if args.mpc_plan is not None:
        scenario["mpc_plan_file"] = str(args.mpc_plan.resolve())
    if args.background_profile is not None:
        scenario["background_profile_file"] = str(args.background_profile.resolve())
    if args.future_request_limit is not None:
        if args.future_request_limit < 0:
            parser.error("future-request-limit must be nonnegative")
        scenario["future_request_limit"] = args.future_request_limit
    if args.mpc_per_slot < 1:
        parser.error("mpc-per-slot must be positive")
    report = run_experiment(scenario, args.model_dir, args.output,
                            ExperimentConfig(tuple(args.strategies), args.horizon,
                                             args.discount, mpc_per_slot=args.mpc_per_slot,
                                             write_csv=not args.compact_output,
                                             local_mpc=args.local_mpc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
