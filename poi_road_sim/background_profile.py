"""Compact, ID-indexed daily background demand for a fixed simulation graph."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DailyBackgroundProfile:
    road_ratios: tuple[float, ...]
    poi_ratios: dict[str, tuple[float, ...]]
    road_regions: dict[int, int]
    poi_regions: dict[int, int]
    road_factors: dict[int, float]
    poi_factors: dict[int, float]
    poi_profiles: dict[int, str]
    poi_departure_fractions: dict[int, float]
    region_road_factors: dict[int, float]
    region_poi_factors: dict[int, float]
    max_road_ratio: float
    max_poi_arrival_service_ratio: float

    def __deepcopy__(self, memo):
        # The profile is read-only after loading; MPC forks share its large ID maps.
        return self

    @classmethod
    def from_file(cls, path: Path, road_ids=None, poi_ids=None) -> "DailyBackgroundProfile":
        path = Path(path)
        if path.suffix in (".sqlite3", ".db"):
            if road_ids is None or poi_ids is None:
                raise ValueError("citywide profiles require the loaded road and POI IDs")
            road_ids, poi_ids = tuple(map(int, road_ids)), tuple(map(int, poi_ids))
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                row = connection.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
                if row is None:
                    raise ValueError("citywide profile has no metadata")
                payload = json.loads(row[0])

                def fetch(table: str, field: str, ids: tuple[int, ...]):
                    result = []
                    for offset in range(0, len(ids), 500):
                        chunk = ids[offset:offset + 500]
                        if chunk:
                            marks = ",".join("?" for _ in chunk)
                            result.extend(connection.execute(
                                f"SELECT * FROM {table} WHERE {field} IN ({marks})", chunk))
                    return result

                road_rows = fetch("road_profile", "edge_id", road_ids)
                poi_rows = fetch("poi_profile", "poi_id", poi_ids)
            payload.update({
                "road_regions": {row[0]: row[1] for row in road_rows},
                "road_factors": {row[0]: row[2] for row in road_rows},
                "poi_regions": {row[0]: row[1] for row in poi_rows},
                "poi_factors": {row[0]: row[2] for row in poi_rows},
                "poi_profiles": {row[0]: row[3] for row in poi_rows},
                "poi_departure_fractions": {row[0]: row[4] for row in poi_rows},
            })
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("slots_per_day") != 96:
            raise ValueError("background profile requires schema_version=1 and 96 slots/day")
        profile = cls(
            road_ratios=tuple(map(float, payload["road_ratios"])),
            poi_ratios={name: tuple(map(float, values)) for name, values in payload["poi_ratios"].items()},
            road_regions={int(k): int(v) for k, v in payload["road_regions"].items()},
            poi_regions={int(k): int(v) for k, v in payload["poi_regions"].items()},
            road_factors={int(k): float(v) for k, v in payload["road_factors"].items()},
            poi_factors={int(k): float(v) for k, v in payload["poi_factors"].items()},
            poi_profiles={int(k): str(v) for k, v in payload["poi_profiles"].items()},
            poi_departure_fractions={int(k): float(v) for k, v in payload["poi_departure_fractions"].items()},
            region_road_factors={int(k): float(v) for k, v in payload["region_road_factors"].items()},
            region_poi_factors={int(k): float(v) for k, v in payload["region_poi_factors"].items()},
            max_road_ratio=float(payload["max_road_ratio"]),
            max_poi_arrival_service_ratio=float(payload["max_poi_arrival_service_ratio"]),
        )
        profile._validate()
        if road_ids is not None and poi_ids is not None:
            profile.validate_ids(road_ids, poi_ids)
        return profile

    def _validate(self) -> None:
        curves = (self.road_ratios, *self.poi_ratios.values())
        if not self.poi_ratios or any(len(curve) != 96 for curve in curves):
            raise ValueError("road and POI curves must each contain 96 values")
        if any(not math.isfinite(value) or value < 0 for curve in curves for value in curve):
            raise ValueError("background curves must be finite and non-negative")
        if not 0 < self.max_road_ratio or not 0 < self.max_poi_arrival_service_ratio:
            raise ValueError("background ratio limits must be positive")
        for regions, factors, region_factors in (
            (self.road_regions, self.road_factors, self.region_road_factors),
            (self.poi_regions, self.poi_factors, self.region_poi_factors),
        ):
            if regions.keys() != factors.keys() or not set(regions.values()) <= region_factors.keys():
                raise ValueError("background ID/region/factor maps are inconsistent")
            if any(not math.isfinite(value) or value <= 0 for value in
                   (*factors.values(), *region_factors.values())):
                raise ValueError("background factors must be finite and positive")
        if self.poi_regions.keys() != self.poi_profiles.keys() or not set(self.poi_profiles.values()) <= self.poi_ratios.keys():
            raise ValueError("POI profile assignments are inconsistent")
        if self.poi_regions.keys() != self.poi_departure_fractions.keys() or any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in self.poi_departure_fractions.values()):
            raise ValueError("POI departure fractions must be in [0, 1] for every POI")

    def validate_ids(self, road_ids, poi_ids) -> None:
        if set(road_ids) != self.road_regions.keys() or set(poi_ids) != self.poi_regions.keys():
            raise ValueError("background profile must cover exactly the loaded road and POI IDs")

    def road_flow(self, road, slot: int) -> float:
        edge_id = road.edge_id
        ratio = (self.road_ratios[slot % 96] * self.road_factors[edge_id] *
                 self.region_road_factors[self.road_regions[edge_id]])
        return road.capacity_veh_per_slot * min(ratio, self.max_road_ratio)

    def poi_arrivals(self, poi, slot: int) -> float:
        poi_id = poi.poi_id
        ratio = (self.poi_ratios[self.poi_profiles[poi_id]][slot % 96] * self.poi_factors[poi_id] *
                 self.region_poi_factors[self.poi_regions[poi_id]])
        return poi.service_people_per_slot * min(ratio, self.max_poi_arrival_service_ratio)

    def poi_departures(self, poi, eligible_occupancy: float) -> float:
        return max(0.0, eligible_occupancy) * self.poi_departure_fractions[poi.poi_id]
