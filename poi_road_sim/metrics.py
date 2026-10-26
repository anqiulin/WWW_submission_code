"""Road--POI load metrics shared by planning, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
from typing import Mapping


@dataclass(frozen=True)
class GlobalLoadWeights:
    road: float = 1.0
    poi: float = 1.0
    road_reference_vehicle_min: float = 1.0
    poi_reference_person_min: float = 1.0
    mode: str = "vehicle_min"
    road_top_k: int = 0

    def __post_init__(self):
        if min(self.road_reference_vehicle_min, self.poi_reference_person_min) <= 0:
            raise ValueError("cost reference scales must be positive")
        if self.mode not in ("vehicle_min", "bounded_topk", "queue_wait"):
            raise ValueError(f"unknown cost mode: {self.mode}")
        if self.mode == "bounded_topk" and (self.road_top_k < 1 or
                                             not 0 <= self.road <= 1 or
                                             not 0 <= self.poi <= 1):
            raise ValueError("bounded_topk needs road_top_k >= 1 and weights in [0, 1]")


def global_load(simulator, result: Mapping | None = None,
                weights: GlobalLoadWeights | None = None) -> dict[str, float]:
    """Return global road, POI and combined load for one simulator snapshot."""
    weights = weights or getattr(simulator, "cost_weights", GlobalLoadWeights())
    aggregate = result.get("aggregate") if result else None
    if aggregate is not None and weights.mode == "queue_wait":
        road_queue = aggregate["road_queue_vehicle_min"]
        road_delay = aggregate["road_travel_delay_vehicle_min"]
        poi_wait = aggregate["poi_wait_person_min"]
        road_component = weights.road * road_queue / weights.road_reference_vehicle_min
        poi_component = weights.poi * poi_wait / weights.poi_reference_person_min
        return {"road_load": road_component, "poi_load": poi_component,
                "total_load": road_component + poi_component,
                "peak_vc": aggregate["peak_vc"],
                "peak_poi_utilization": aggregate["peak_poi_utilization"],
                "road_queue_vehicle_min": road_queue,
                "road_travel_delay_vehicle_min": road_delay,
                "poi_wait_person_min": poi_wait,
                "raw_road_vehicle_min": road_queue,
                "raw_poi_person_min": poi_wait,
                "raw_topk_road_vehicle_min": road_queue,
                "road_top_k": len(simulator.roads)}
    roads = result.get("roads", {}) if result else {}
    pois = result.get("pois", {}) if result else {}
    road_load = 0.0
    road_queue_vehicle_min = 0.0
    road_travel_delay_vehicle_min = 0.0
    edge_costs = []
    peak_vc = 0.0
    for edge_id, road in simulator.roads.items():
        row = roads.get(edge_id, {})
        load = float(row.get("total_load_vehicles", road.queue_vehicles))
        ratio = load / max(1.0, road.capacity_veh_per_slot)
        free_time = simulator._travel_time(road, 0.0)
        served = float(row.get("outflow_vehicles", road.outflow_vehicles))
        queued = float(row.get("queue_vehicles", road.queue_vehicles))
        travel_time = float(row.get("travel_time_sec", road.travel_time_sec))
        delay = served * max(0.0, travel_time - free_time) / 60.0
        queue = queued * 15.0
        road_travel_delay_vehicle_min += delay
        road_queue_vehicle_min += queue
        edge_costs.append(delay + queue)
        peak_vc = max(peak_vc, ratio)
    road_load = road_travel_delay_vehicle_min + road_queue_vehicle_min
    poi_load = 0.0
    peak_poi = 0.0
    for poi_id, poi in simulator.pois.items():
        row = pois.get(poi_id, {})
        waiting = float(row.get("waiting", poi.waiting))
        occupancy = float(row.get("occupancy", poi.occupancy))
        capacity = max(1.0, poi.capacity_people)
        poi_load += waiting * 15.0
        peak_poi = max(peak_poi, occupancy / capacity)
    road_topk = sum(nlargest(weights.road_top_k, edge_costs)) if weights.mode == "bounded_topk" else road_load
    road_basis = road_queue_vehicle_min if weights.mode == "queue_wait" else road_load
    if weights.mode == "bounded_topk":
        road_component = weights.road * road_topk / (road_topk + weights.road_reference_vehicle_min)
        poi_component = weights.poi * poi_load / (poi_load + weights.poi_reference_person_min)
    else:
        road_component = weights.road * road_basis / weights.road_reference_vehicle_min
        poi_component = weights.poi * poi_load / weights.poi_reference_person_min
    total = road_component + poi_component
    return {"road_load": road_component, "poi_load": poi_component,
            "total_load": total, "peak_vc": peak_vc,
            "peak_poi_utilization": peak_poi,
            "road_queue_vehicle_min": road_queue_vehicle_min,
            "road_travel_delay_vehicle_min": road_travel_delay_vehicle_min,
            "poi_wait_person_min": poi_load,
            "raw_road_vehicle_min": road_basis,
            "raw_poi_person_min": poi_load,
            "raw_topk_road_vehicle_min": road_topk,
            "road_top_k": min(weights.road_top_k, len(edge_costs)) if weights.mode == "bounded_topk" else len(edge_costs)}
