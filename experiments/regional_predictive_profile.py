"""Auditable predictive stress interventions on routed NYC regional scenes."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict


PROFILE = "nyc_regional_predictive_v2"


def _route_length(candidate: dict, lengths: dict[int, float]) -> float:
    return sum(lengths[int(edge)] for edge in
               candidate["edge_ids"] + candidate["egress_edge_ids"])


def apply_profile(source: dict, version: str = "predictive_v2") -> dict:
    """Configure three distinct mechanisms without changing topology or routes.

    Scenarios are synthetic interventions on real routed edges. This function
    does not select costs or thresholds using held-out strategy results.
    """
    scenario = copy.deepcopy(source)
    if version not in ("predictive_v2", "predictive_v3"):
        raise ValueError(f"unknown predictive profile version: {version}")
    condition = scenario["condition"]
    if condition not in ("regional_delayed", "regional_coupling", "regional_shock"):
        raise ValueError(f"unknown predictive condition: {condition}")
    lengths = {int(road["edge_id"]): float(road["length_m"]) for road in scenario["roads"]}
    poi_by_id = {int(poi["poi_id"]): poi for poi in scenario["pois"]}
    requests_by_region = defaultdict(list)
    for request in scenario["requests"]:
        region = int(request["request_id"].rsplit("-", 2)[-2])
        requests_by_region[region].append(request)
        # The stored query permits either category present in the candidate
        # pool; the dataset contains a category set, not an impossible label.
        request["categories"] = sorted({int(poi_by_id[c["destination_poi_id"]]["category"])
                                        for c in request["candidates"]})

    # Identify a first-leg edge that the shortest candidate uses often while
    # at least one alternative avoids it. Penalizing only these shared edges
    # makes route choice, rather than citywide background, drive road queues.
    bottlenecks = set()
    hot_pois = {}
    for region, requests in requests_by_region.items():
        example = {request["origin_node_id"]: request for request in requests}.values()
        counts = Counter()
        for request in example:
            choices = request["candidates"]
            shortest = min(choices, key=lambda c: _route_length(c, lengths))
            for edge in shortest["edge_ids"][:6]:
                if any(int(edge) not in c["edge_ids"] for c in choices):
                    counts[int(edge)] += 1
        if counts:
            bottlenecks.add(counts.most_common(1)[0][0])
        poi_counts = Counter(int(c["destination_poi_id"]) for request in example
                             for c in request["candidates"])
        if poi_counts:
            hot_pois[region] = poi_counts.most_common(1)[0][0]
    if not bottlenecks:
        raise ValueError("no avoidable shortest-route bottleneck in this regional scene")

    for road in scenario["roads"]:
        edge_id = int(road["edge_id"])
        if edge_id in bottlenecks:
            road["capacity_veh_per_slot"] = 6.0
        else:
            road["capacity_veh_per_slot"] = max(4.0, float(road["capacity_veh_per_slot"]))
        road["queue_vehicles"] = road["recommended_vehicles"] = 0.0
        road["background_vehicles"] = 0.0
        road["count_history"] = [0.0] * 4
    for poi in scenario["pois"]:
        poi["capacity_people"] = 24.0
        poi["service_people_per_slot"] = 5.0
        poi["occupancy"] = 2.0
        poi["waiting"] = 0.0
        poi["arrival_history"] = [0.0] * 4
        poi["departure_history"] = [2.0] * 4

    for request in scenario["requests"]:
        slot = int(request["departure_slot"])
        region = int(request["request_id"].rsplit("-", 2)[-2])
        if condition == "regional_delayed" and 24 <= slot < 40:
            hot = hot_pois.get(region)
            if hot is not None and any(int(c["destination_poi_id"]) == hot
                                   for c in request["candidates"]):
                request["forecast_poi_id"] = hot
    for observation in scenario["observations"]:
        slot = int(observation["slot"])
        background = {str(road["edge_id"]): 0.20 for road in scenario["roads"]}
        if condition == "regional_coupling":
            for edge_id in bottlenecks:
                background[str(edge_id)] = 0.5
        observation["road_background"] = background
        observation["poi_arrivals"] = {str(poi_id): 0.05 for poi_id in poi_by_id}
        if condition == "regional_delayed" and 24 <= slot < 40:
            for poi_id in set(hot_pois.values()):
                observation["poi_arrivals"][str(poi_id)] = 8.0
        observation["poi_departures"] = {str(poi_id): 2.5 for poi_id in poi_by_id}

    profile_name = f"nyc_regional_{version}"
    scenario["profile"] = profile_name
    scenario["policy_use_forecast"] = True
    scenario["provenance"].update({
        "profile": profile_name,
        "bottleneck_edge_ids": sorted(bottlenecks),
        "hot_poi_by_region": hot_pois,
        "mechanism": {
            "regional_delayed": "low current queue, forecast regional POI demand and later exogenous arrivals",
            "regional_coupling": "near shortest routes share low-capacity edges with road background",
            "regional_shock": "abrupt regional request peak on the common route pool",
        }[condition],
        "scope_note": "real NYC route-local road subgraph and POI candidates; synthetic demand/capacity intervention",
        "not_a_claim": "configuration does not guarantee superiority of MPC or PPO",
    })
    return scenario
