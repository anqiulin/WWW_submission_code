"""Select dense NYC POI regions from the complete v4 static dataset.

This is a catalog step, not yet a route-candidate expansion.  It reads all
136k POIs, selects dense spatial centers, and writes selected POIs per region
with their real nearest road node IDs. An optional eligible-ID filter supports
regions drawn from the largest routable road component.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_pois(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        declared = int(stream.readline().strip().split("\t")[0])
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 4:
                continue
            rows.append({"poi_id": int(row[0]), "lat": float(row[1]),
                         "lon": float(row[2]), "category": int(row[3])})
    if len(rows) != declared:
        raise ValueError(f"POI count mismatch: header={declared}, rows={len(rows)}")
    return rows


def read_mapping(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8") as stream:
        return {int(row["poi_id"]): int(row["nearest_node_id"])
                for row in csv.DictReader(stream, delimiter="\t")}


def initialize(points: list[dict], k: int) -> list[tuple[float, float]]:
    # Start at deterministic quantiles of longitude and latitude in the
    # central 70% to avoid selecting sparse airport/water-edge points.
    lats = sorted(p["lat"] for p in points)
    lons = sorted(p["lon"] for p in points)
    centers = []
    for index in range(k):
        q = 0.15 + 0.7 * (index + 0.5) / k
        centers.append((lats[int(q * (len(lats) - 1))],
                        lons[int(q * (len(lons) - 1))]))
    # Spread coincident diagonal initial points using longitude ranks.
    return centers


def dense_centers(points: list[dict], k: int, cell_size: float = 0.05,
                  min_separation: float = 0.07) -> list[tuple[float, float]]:
    """Pick separated high-density cells instead of clustering suburbs."""
    cells: Counter[tuple[int, int]] = Counter(
        (int(point["lat"] / cell_size), int(point["lon"] / cell_size))
        for point in points)
    ranked = sorted(cells.items(), key=lambda item: (-item[1], item[0]))
    centers: list[tuple[float, float]] = []
    for (lat_cell, lon_cell), _ in ranked:
        center = ((lat_cell + 0.5) * cell_size, (lon_cell + 0.5) * cell_size)
        if all(math.dist(center, other) >= min_separation for other in centers):
            centers.append(center)
        if len(centers) == k:
            break
    if len(centers) < k:
        raise ValueError(f"only {len(centers)} separated dense regions found")
    return centers


def cluster(points: list[dict], k: int, iterations: int = 12) -> tuple[list[int], list[tuple[float, float]]]:
    centers = initialize(points, k)
    labels = [0] * len(points)
    for _ in range(iterations):
        for index, point in enumerate(points):
            labels[index] = min(range(k), key=lambda cluster_id:
                                (point["lat"] - centers[cluster_id][0]) ** 2 +
                                (point["lon"] - centers[cluster_id][1]) ** 2)
        updated = []
        for cluster_id in range(k):
            members = [points[i] for i, label in enumerate(labels) if label == cluster_id]
            if not members:
                updated.append(centers[cluster_id])
            else:
                updated.append((sum(p["lat"] for p in members) / len(members),
                                sum(p["lon"] for p in members) / len(members)))
        if max(math.dist(a, b) for a, b in zip(centers, updated)) < 1e-7:
            centers = updated
            break
        centers = updated
    return labels, centers


def build(poi_file: Path, mapping_file: Path, output: Path,
          regions: int, pois_per_region: int, seed: int,
          eligible_poi_ids: set[int] | None = None,
          center_cell_size: float = 0.05,
          center_min_separation: float = 0.07) -> dict:
    points = read_pois(poi_file)
    mapping = read_mapping(mapping_file)
    source_count = len(points)
    if eligible_poi_ids is not None:
        points = [point for point in points if point["poi_id"] in eligible_poi_ids]
    centers = dense_centers(points, regions, center_cell_size, center_min_separation)
    labels = [min(range(regions), key=lambda region_id:
                   (point["lat"] - centers[region_id][0]) ** 2 +
                   (point["lon"] - centers[region_id][1]) ** 2)
              for point in points]
    output.mkdir(parents=True, exist_ok=True)
    region_rows = []
    selected_ids = set()
    for region_id in range(regions):
        members = [point for point, label in zip(points, labels) if label == region_id
                   and point["poi_id"] in mapping]
        center = centers[region_id]
        members.sort(key=lambda p: ((p["lat"] - center[0]) ** 2 +
                                    (p["lon"] - center[1]) ** 2,
                                    p["poi_id"]))
        chosen = members[:pois_per_region]
        if len(chosen) < pois_per_region:
            raise ValueError(f"region {region_id} has only {len(chosen)} mapped POIs")
        selected_ids.update(p["poi_id"] for p in chosen)
        region_dir = output / f"region_{region_id:02d}"
        region_dir.mkdir(parents=True, exist_ok=True)
        with (region_dir / "pois.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("poi_id", "nearest_node_id", "lat", "lon", "category"))
            writer.writeheader()
            for point in chosen:
                writer.writerow({**point, "nearest_node_id": mapping[point["poi_id"]]})
        region_rows.append({"region_id": region_id, "poi_count": len(chosen),
                            "center_lat": center[0], "center_lon": center[1],
                            "lat_min": min(p["lat"] for p in chosen), "lat_max": max(p["lat"] for p in chosen),
                            "lon_min": min(p["lon"] for p in chosen), "lon_max": max(p["lon"] for p in chosen),
                            "category_count": dict(Counter(p["category"] for p in chosen))})
    manifest = {"source_poi_file": str(poi_file), "source_mapping_file": str(mapping_file),
                "source_poi_count": source_count, "eligible_poi_count": len(points),
                "regions": regions,
                "pois_per_region": pois_per_region, "selected_poi_count": len(selected_ids),
                "seed": seed, "selection": "density-cell centers then nearest-to-center POIs",
                "region_summary": region_rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "region_summary.csv").write_text(
        "region_id,poi_count,center_lat,center_lon,lat_min,lat_max,lon_min,lon_max\n" +
        "\n".join(",".join(str(row[key]) for key in
                           ("region_id", "poi_count", "center_lat", "center_lon", "lat_min", "lat_max", "lon_min", "lon_max"))
                 for row in region_rows) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi-file", type=Path,
                        default=Path("experiment_data/nyc_v4_final/static/POI.txt"))
    parser.add_argument("--mapping-file", type=Path,
                        default=Path("experiment_data/nyc_v4_final/interfaces/poi_node_map.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--pois-per-region", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    print(json.dumps(build(args.poi_file, args.mapping_file, args.output,
                           args.regions, args.pois_per_region, args.seed), indent=2))


if __name__ == "__main__":
    main()
