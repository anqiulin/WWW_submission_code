"""Stream every NYC static edge and POI into a ten-core daily background SQLite file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.build_background_profile import _curve


def _factor(identifier: int, seed: int, salt: int, spread: float) -> float:
    if not 0 <= spread < 1:
        raise ValueError("object factor spread must be in [0, 1)")
    value = (identifier ^ seed ^ salt) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return 1.0 + spread * (2.0 * value / 0xFFFFFFFFFFFFFFFF - 1.0)


def _node_coordinates(path: Path) -> tuple[array, array]:
    with path.open(encoding="utf-8") as stream:
        count = int(stream.readline().split("\t")[0])
        latitudes = array("f", [math.nan]) * (count + 1)
        longitudes = array("f", [math.nan]) * (count + 1)
        for row in csv.reader(stream, delimiter="\t"):
            node_id = int(row[0])
            if not 0 < node_id <= count:
                raise ValueError(f"node ID outside declared range: {node_id}")
            latitudes[node_id], longitudes[node_id] = float(row[1]), float(row[2])
    return latitudes, longitudes


def _nearest_core(lat: float, lon: float, centers: list[tuple[float, float]]) -> tuple[int, float]:
    region = min(range(len(centers)), key=lambda index:
                 (lat - centers[index][0]) ** 2 + (lon - centers[index][1]) ** 2)
    center_lat, center_lon = centers[region]
    distance_km = math.hypot((lat - center_lat) * 111.0,
                             (lon - center_lon) * 111.0 * math.cos(math.radians(center_lat)))
    return region, distance_km


def build(root: Path, catalog: Path, config: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"profile already exists; choose a new output: {output}")
    manifest = json.loads((catalog / "manifest.json").read_text(encoding="utf-8"))
    count = int(config["region_count"])
    if count != int(manifest["regions"]) or len(manifest["region_summary"]) != count:
        raise ValueError("background region_count must match the POI catalog")
    centers = [(float(row["center_lat"]), float(row["center_lon"]))
               for row in manifest["region_summary"]]
    core_radius = float(config["core_radius_km"])
    if not math.isfinite(core_radius) or core_radius <= 0:
        raise ValueError("core_radius_km must be positive")
    hot = config.get("hot_region_ids")
    hot = list(range(int(config.get("hot_region_count", 0)))) if hot is None else list(map(int, hot))
    if len(hot) != len(set(hot)) or any(not 0 <= region < count for region in hot):
        raise ValueError("hot region IDs must be unique and inside region_count")
    road_config, poi_config = config["road"], config["poi"]
    road_curve = _curve(road_config["capacity_ratio_anchors"])
    poi_curves = {name: _curve(anchors) for name, anchors in poi_config["arrival_ratio_anchors"].items()}
    if "default" not in poi_curves:
        raise ValueError("POI arrival curves require default")
    seed = int(config.get("seed", 0))
    road_spread = float(road_config.get("edge_factor_spread", 0))
    poi_spread = float(poi_config.get("poi_factor_spread", 0))
    road_hot, poi_hot = float(road_config["hot_region_multiplier"]), float(poi_config["hot_region_multiplier"])
    road_outer = float(road_config["outer_region_multiplier"])
    poi_outer = float(poi_config["outer_region_multiplier"])
    if any(not math.isfinite(value) or value <= 0 for value in
           (road_hot, poi_hot, road_outer, poi_outer)):
        raise ValueError("core and outer background multipliers must be positive")
    road_overrides = {int(k): float(v) for k, v in road_config.get("edge_factor_overrides", {}).items()}
    poi_overrides = {int(k): float(v) for k, v in poi_config.get("poi_factor_overrides", {}).items()}
    profile_overrides = {int(k): str(v) for k, v in poi_config.get("poi_profile_overrides", {}).items()}
    unknown_road_overrides = set(road_overrides)
    unknown_poi_overrides = set(poi_overrides) | set(profile_overrides)
    categories = {int(k): str(v) for k, v in poi_config.get("category_profiles", {}).items()}
    departures = {str(k): float(v) for k, v in poi_config["departure_fraction_by_profile"].items()}
    if not set(departures) <= set(poi_curves) or any(not 0 <= value <= 1 for value in departures.values()):
        raise ValueError("invalid POI departure fractions")
    latitudes, longitudes = _node_coordinates(root / "static/node.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unfinished temporary profile exists: {temporary}")
    road_count = poi_count = 0
    road_regions, poi_regions = Counter(), Counter()
    road_cores, poi_cores = Counter(), Counter()
    try:
        with sqlite3.connect(temporary) as database:
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            database.execute("CREATE TABLE road_profile(edge_id INTEGER PRIMARY KEY, region INTEGER NOT NULL, factor REAL NOT NULL)")
            database.execute("CREATE TABLE poi_profile(poi_id INTEGER PRIMARY KEY, region INTEGER NOT NULL, factor REAL NOT NULL, profile TEXT NOT NULL, departure REAL NOT NULL)")
            edge_file = root / "static/edge.txt"
            with edge_file.open(encoding="utf-8") as stream:
                declared_edges = int(stream.readline().split("\t")[1])
                batch = []
                for row in csv.reader(stream, delimiter="\t"):
                    edge_id, from_node = int(row[0]), int(row[1])
                    unknown_road_overrides.discard(edge_id)
                    if from_node >= len(latitudes) or not math.isfinite(latitudes[from_node]):
                        raise ValueError(f"edge {edge_id} has no source-node coordinates")
                    region, distance = _nearest_core(latitudes[from_node], longitudes[from_node], centers)
                    is_core = region in hot and distance <= core_radius
                    multiplier = road_hot if is_core else road_outer
                    factor = road_overrides.get(edge_id, _factor(edge_id, seed, 17, road_spread)) * multiplier
                    if not math.isfinite(factor) or factor <= 0:
                        raise ValueError(f"invalid road factor at edge {edge_id}")
                    batch.append((edge_id, region, factor))
                    road_regions[region] += 1
                    road_cores[region] += is_core
                    road_count += 1
                    if len(batch) >= 10000:
                        database.executemany("INSERT INTO road_profile VALUES(?,?,?)", batch)
                        batch.clear()
                        if road_count % 500000 == 0:
                            print(f"[background] roads {road_count}/{declared_edges}", flush=True)
                if batch:
                    database.executemany("INSERT INTO road_profile VALUES(?,?,?)", batch)
            if road_count != declared_edges:
                raise ValueError(f"road count mismatch: {road_count} != {declared_edges}")
            poi_file = root / "static/POI.txt"
            with poi_file.open(encoding="utf-8") as stream:
                declared_pois = int(stream.readline().split("\t")[0])
                batch = []
                for row in csv.reader(stream, delimiter="\t"):
                    poi_id, lat, lon, category = int(row[0]), float(row[1]), float(row[2]), int(row[3])
                    unknown_poi_overrides.discard(poi_id)
                    region, distance = _nearest_core(lat, lon, centers)
                    is_core = region in hot and distance <= core_radius
                    multiplier = poi_hot if is_core else poi_outer
                    factor = poi_overrides.get(poi_id, _factor(poi_id, seed, 29, poi_spread)) * multiplier
                    profile = profile_overrides.get(poi_id, categories.get(category, "default"))
                    if profile not in poi_curves or profile not in departures:
                        raise ValueError(f"unknown POI profile for {poi_id}: {profile}")
                    if not math.isfinite(factor) or factor <= 0:
                        raise ValueError(f"invalid POI factor at {poi_id}")
                    batch.append((poi_id, region, factor, profile, departures[profile]))
                    poi_regions[region] += 1
                    poi_cores[region] += is_core
                    poi_count += 1
                    if len(batch) >= 10000:
                        database.executemany("INSERT INTO poi_profile VALUES(?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    database.executemany("INSERT INTO poi_profile VALUES(?,?,?,?,?)", batch)
            if poi_count != declared_pois:
                raise ValueError(f"POI count mismatch: {poi_count} != {declared_pois}")
            if unknown_road_overrides:
                raise ValueError("road override references an unknown edge ID")
            if unknown_poi_overrides:
                raise ValueError("POI override references an unknown ID")
            payload = {"schema_version": 1, "slots_per_day": 96,
                "road_ratios": road_curve, "poi_ratios": poi_curves,
                "region_road_factors": {region: 1.0 for region in range(count)},
                "region_poi_factors": {region: 1.0 for region in range(count)},
                "max_road_ratio": float(road_config["max_capacity_ratio"]),
                "max_poi_arrival_service_ratio": float(poi_config["max_arrival_service_ratio"])}
            database.execute("INSERT INTO metadata VALUES('profile',?)", (json.dumps(payload),))
            database.commit()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {"profile_file": str(output.resolve()), "static_root": str(root.resolve()),
        "catalog": str(catalog.resolve()), "edge_count": road_count, "poi_count": poi_count,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "region_count": count, "hot_region_ids": hot, "core_radius_km": core_radius,
        "road_counts_by_region": dict(road_regions), "poi_counts_by_region": dict(poi_regions),
        "core_road_counts_by_region": dict(road_cores), "core_poi_counts_by_region": dict(poi_cores),
        "centers": centers, "background_type": "synthetic periodic, not measured traffic"}
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=here.parent / "experiment_data/nyc_v4_final")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new citywide .sqlite3 file")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(args.root, args.catalog, config, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
