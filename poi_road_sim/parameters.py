"""Reproducible simulation parameters for every POI and directed road edge.

These values describe an experimental environment, not measured physical
capacities. The same object ID, seed, and distribution settings give the same
parameters in full-store and small-scenario initialization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterConfig:
    seed: int = 2025
    road_capacity_scale: float = 1.5
    poi_capacity_scale: float = 1.25
    poi_service_scale: float = 1.5
    spread: float = 0.2

    def __post_init__(self) -> None:
        if min(self.road_capacity_scale, self.poi_capacity_scale, self.poi_service_scale) <= 0:
            raise ValueError("parameter scales must be positive")
        if not 0 <= self.spread < 1:
            raise ValueError("spread must be in [0, 1)")


def _factor(identifier: int, salt: int, config: ParameterConfig) -> float:
    # Stable integer mixing avoids Python's process-randomized hash().
    value = (int(identifier) ^ config.seed ^ salt) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return 1.0 + config.spread * (2.0 * (value / 0xFFFFFFFFFFFFFFFF) - 1.0)


def poi_parameters(
    poi_id: int, static_capacity: float | None,
    history: list[tuple[float, ...]], config: ParameterConfig,
) -> dict[str, float | str]:
    if not history:
        raise ValueError("POI history must not be empty")
    peak_occupancy = max(0.0, *(row[2] for row in history))
    peak_arrivals = max(0.0, *(row[0] for row in history))
    valid_static = static_capacity is not None and math.isfinite(static_capacity) and static_capacity > 0
    observed_floor = math.ceil(peak_occupancy)
    if valid_static and static_capacity >= observed_floor:
        capacity = float(static_capacity)
        source = "static"
    else:
        capacity = float(max(observed_floor, math.ceil(
            max(20.0, peak_occupancy * config.poi_capacity_scale) * _factor(poi_id, 1, config)
        )))
        source = "simulated"
    service = float(max(1, math.ceil(
        max(1.0, peak_arrivals * config.poi_service_scale) * _factor(poi_id, 2, config)
    )))
    return {
        "capacity_people": capacity,
        "service_people_per_slot": service,
        "capacity_source": source,
        "service_source": "simulated",
    }


def road_parameters(
    edge_id: int, speed_limit_kmh: float,
    history: list[tuple[float, ...]], config: ParameterConfig,
) -> dict[str, float | str]:
    if not history:
        raise ValueError("road history must not be empty")
    peak_vehicles = max(0.0, *(row[0] for row in history))
    capacity = float(max(1, math.ceil(
        max(1.0, peak_vehicles * config.road_capacity_scale) * _factor(edge_id, 3, config)
    )))
    valid_speed = math.isfinite(speed_limit_kmh) and speed_limit_kmh > 0
    observed_speeds = [row[2] for row in history if math.isfinite(row[2]) and row[2] > 0]
    fallback_speed = max(observed_speeds) if observed_speeds else 30.0
    return {
        "capacity_veh_per_slot": capacity,
        "speed_limit_kmh": float(speed_limit_kmh if valid_speed else fallback_speed),
        "capacity_source": "simulated",
        "speed_source": "static" if valid_speed else "observed_or_default",
    }
