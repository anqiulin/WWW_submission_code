"""One-command full-NYC background and ten-core train/validation/test request setup."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.build_citywide_background import build as build_background
from experiments.build_citywide_core_requests import build as build_requests
from experiments.build_nyc_regional_poi_catalog import build as build_catalog
from experiments.run_nyc_regional_experiment import _calibrate_bounded_topk


def _routable_poi_ids(root: Path) -> tuple[set[int], int]:
    with (root / "static/node.txt").open(encoding="utf-8") as stream:
        node_count = int(stream.readline().split("\t")[0])
    parent = array("i", range(node_count + 1))
    size = array("i", [1]) * (node_count + 1)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    with (root / "static/edge.txt").open(encoding="utf-8") as stream:
        stream.readline()
        for row in csv.reader(stream, delimiter="\t"):
            left, right = find(int(row[1])), find(int(row[2]))
            if left != right:
                if size[left] < size[right]:
                    left, right = right, left
                parent[right] = left
                size[left] += size[right]
    mapping = {}
    with (root / "interfaces/poi_node_map.txt").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            mapping[int(row["poi_id"])] = find(int(row["nearest_node_id"]))
    largest = Counter(mapping.values()).most_common(1)[0][0]
    return {poi_id for poi_id, component in mapping.items() if component == largest}, size[largest]


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(output: Path, root: Path, routing_index: Path, state_store: Path,
            background_config: dict, request_config: dict, smoke: bool = False,
            model_dir: Path | None = None,
            split_seeds: dict[str, int] | None = None) -> dict:
    if int(background_config["region_count"]) != 10 or int(request_config["region_count"]) != 10:
        raise ValueError("this command requires ten regions in both configs")
    if float(background_config["core_radius_km"]) != float(request_config["core_radius_km"]):
        raise ValueError("background and request core radii must match")
    default_seeds = {"train": 9101, "validation": 9102, "test": 9103}
    split_seeds = default_seeds if split_seeds is None else split_seeds
    if set(split_seeds) != {"train", "validation", "test"} or len(set(split_seeds.values())) != 3:
        raise ValueError("train, validation, and test require three distinct seeds")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "preparation_manifest.json"
    keys = {"background_config_sha256": _digest(background_config),
            "request_config_sha256": _digest(request_config), "smoke": smoke,
            "split_seeds": split_seeds}
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (any(previous.get(key) != keys[key] for key in
                ("background_config_sha256", "request_config_sha256", "smoke")) or
                previous.get("split_seeds", default_seeds) != split_seeds):
            raise ValueError("existing output uses a different config; choose a new output directory")
    for name, config in (("background_config.json", background_config),
                         ("request_config.json", request_config)):
        snapshot = output / name
        if not snapshot.exists():
            snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    catalog = output / "catalog"
    if not (catalog / "manifest.json").exists():
        print("[prepare] selecting ten dense cores from the largest routable component", flush=True)
        eligible, component_nodes = _routable_poi_ids(root)
        print(f"[prepare] routable component: {component_nodes} nodes, {len(eligible)} POIs", flush=True)
        build_catalog(root / "static/POI.txt", root / "interfaces/poi_node_map.txt",
                      catalog, 10, int(request_config.get("catalog_pois_per_region", 400)),
                      int(request_config.get("catalog_seed", 20261015)), eligible,
                      center_cell_size=0.025, center_min_separation=0.05)
    catalog_manifest = json.loads((catalog / "manifest.json").read_text(encoding="utf-8"))
    if (catalog_manifest["regions"] != 10 or catalog_manifest["pois_per_region"] !=
            int(request_config.get("catalog_pois_per_region", 400)) or
            catalog_manifest.get("eligible_poi_count", 0) < 4000 or
            catalog_manifest["seed"] != int(request_config.get("catalog_seed", 20261015))):
        raise ValueError("existing catalog does not match the request config")
    background = output / "citywide_background.sqlite3"
    if not background.exists():
        print("[prepare] indexing every static road edge and POI", flush=True)
        build_background(root, catalog, background_config, background)
    report = json.loads(background.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (report["region_count"] != 10 or report["catalog"] != str(catalog.resolve()) or
            report.get("config_sha256") != keys["background_config_sha256"]):
        raise ValueError("existing citywide background does not match this catalog")
    scenarios = []
    slots = 4 if smoke else int(request_config.get("simulation_steps", 96))
    start_slot = int(request_config.get("start_slot", 0))
    if not 1 <= slots <= 96 or not 0 <= start_slot < 96 or start_slot + slots > 96:
        raise ValueError("request simulation_steps and start_slot must fit inside one 96-slot day")
    for split, seed in split_seeds.items():
        path = output / "dataset" / split / "citywide_ten_cores" / f"seed_{seed}" / "scenario.json"
        if path.exists():
            scene = json.loads(path.read_text(encoding="utf-8"))
            if (scene.get("background_profile_file") != str(background.resolve()) or
                    int(scene["steps"]) != slots or int(scene.get("start_slot", 0)) != start_slot):
                raise ValueError(f"existing scenario does not match this run: {path}")
            row = {"scenario": str(path), "split": split, "requests": len(scene["requests"]),
                   "requests_by_region": scene["provenance"]["requests_by_region"],
                   "selected_edges": len(scene["roads"]), "selected_pois": len(scene["pois"])}
        else:
            row = build_requests(catalog, routing_index, state_store, background,
                                 request_config, path, split, seed, slots, start_slot)
        scenarios.append(row)
    calibration = None
    if not smoke and request_config.get("cost_mode") == "bounded_topk":
        model_dir = model_dir or Path(__file__).resolve().parents[1] / "artifacts"
        train = [Path(row["scenario"]) for row in scenarios if row["split"] == "train"]
        all_paths = [Path(row["scenario"]) for row in scenarios]
        print("[prepare] calibrating road and POI costs on the training split", flush=True)
        calibration = _calibrate_bounded_topk(output, model_dir, train, all_paths)
    result = {**keys, "catalog": str(catalog), "background": str(background),
              "background_config_snapshot": str(output / "background_config.json"),
              "request_config_snapshot": str(output / "request_config.json"),
              "background_counts": {"edges": report["edge_count"], "pois": report["poi_count"]},
              "scenarios": scenarios, "cost_calibration": calibration,
              "note": "Full NYC background coverage; simulator loads route-local state only. "
                      "Request intensity, core centers and capacity scales are synthetic."}
    marker.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=here.parent / "experiment_data/nyc_v4_final")
    parser.add_argument("--routing-index", type=Path, default=here / "routing_index_calibrated")
    parser.add_argument("--state-store", type=Path, default=here / "state_store")
    parser.add_argument("--model-dir", type=Path, default=here / "artifacts")
    parser.add_argument("--background-config", type=Path,
                        default=here / "configs/citywide_background_10_regions.json")
    parser.add_argument("--request-config", type=Path,
                        default=here / "configs/citywide_core_requests_10_regions.json")
    parser.add_argument("--smoke", action="store_true", help="generate only the first four time slots")
    parser.add_argument("--train-seed", type=int, default=9101)
    parser.add_argument("--validation-seed", type=int, default=9102)
    parser.add_argument("--test-seed", type=int, default=9103)
    args = parser.parse_args()
    background_config = json.loads(args.background_config.read_text(encoding="utf-8"))
    request_config = json.loads(args.request_config.read_text(encoding="utf-8"))
    print(json.dumps(prepare(args.output.resolve(), args.root, args.routing_index,
                             args.state_store, background_config, request_config,
                             args.smoke, args.model_dir,
                             {"train": args.train_seed, "validation": args.validation_seed,
                              "test": args.test_seed}), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
