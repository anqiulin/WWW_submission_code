"""Fit background counts from complete 96-slot object histories."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .data import (
    iter_histories,
    iter_road_histories_with_edges,
    load_edges,
    load_mta_signal,
    load_pois,
    sample_histories,
)
from .model import CountModel, poi_extra, road_extra


ROOT = Path(__file__).resolve().parents[2] / "experiment_data" / "nyc_v4_final"


def _split(identifier: int, seed: int) -> bool:
    """Keep all time slots for an object in the same train or validation split."""
    return ((identifier * 2654435761 + seed) % 5) == 0


def _evaluate_poi(
    histories: list[tuple[int, list[tuple[float, ...]]]],
    pois: dict[int, dict[str, float | int | None]],
    mta: dict[int, tuple[float, ...]],
    arrivals_model: CountModel,
    departures_model: CountModel,
) -> dict:
    sums = {name: 0.0 for name in ("arrival_mae", "departure_mae", "arrival_persistence_mae", "departure_persistence_mae")}
    n = 0
    for poi_id, rows in histories:
        group = int(pois[poi_id]["category"])
        signal = mta.get(poi_id, (0.0,) * 4)
        arrivals = [row[0] for row in rows]
        departures = [row[1] for row in rows]
        for slot in range(1, 96):
            extra = poi_extra(rows[slot - 1][2], signal[slot // 24])
            sums["arrival_mae"] += abs(arrivals_model.predict(arrivals[max(0, slot - 4):slot], slot, extra, group) - arrivals[slot])
            sums["departure_mae"] += abs(departures_model.predict(departures[max(0, slot - 4):slot], slot, extra, group) - departures[slot])
            sums["arrival_persistence_mae"] += abs(arrivals[slot - 1] - arrivals[slot])
            sums["departure_persistence_mae"] += abs(departures[slot - 1] - departures[slot])
            n += 1
    return {name: round(value / max(1, n), 4) for name, value in sums.items()} | {"validation_rows": n}


def _limit(objects: str) -> int | None:
    return None if objects.lower() == "all" else int(objects)


def _poi_histories(root: Path, objects: str, seed: int, epoch: int):
    path = root / "dynamic" / "POI_state_mta_15min.txt"
    if objects.lower() == "all":
        return iter_histories(path, "poi")
    # Sampling remains useful for quick smoke tests. Full training is sequential.
    return sample_histories(path, "poi", int(objects), seed + epoch)


def _log(message: str, log_path: Path | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def train_poi(root: Path, output: Path, objects: str, epochs: int, seed: int,
              progress_every: int, log_path: Path | None = None) -> dict:
    pois = load_pois(root / "static" / "POI.txt")
    mta = load_mta_signal(root / "calibration" / "mta_station_period.csv", root / "interfaces" / "poi_mta_links.csv")
    arrival_model = CountModel.create("poi_base_arrivals_15min", 2)
    departure_model = CountModel.create("poi_departures_15min", 2)
    for epoch in range(epochs):
        train_count = validation_count = 0
        for poi_id, rows in _poi_histories(root, objects, seed, epoch):
            if poi_id not in pois or _split(poi_id, seed):
                validation_count += 1
                continue
            train_count += 1
            if progress_every and train_count % progress_every == 0:
                _log(f"POI epoch {epoch + 1}/{epochs}: processed {train_count:,} training objects", log_path)
            group = int(pois[poi_id]["category"])
            signal = mta.get(poi_id, (0.0,) * 4)
            arrivals = [row[0] for row in rows]
            departures = [row[1] for row in rows]
            for slot in range(1, 96):
                extra = poi_extra(rows[slot - 1][2], signal[slot // 24])
                arrival_model.update(arrivals[max(0, slot - 4):slot], slot, extra, arrivals[slot], group, 0.006)
                departure_model.update(departures[max(0, slot - 4):slot], slot, extra, departures[slot], group, 0.006)
        _log(f"POI epoch {epoch + 1}/{epochs} complete; train={train_count:,}, validation={validation_count:,}", log_path)
    arrival_model.save(output / "poi_arrivals.json")
    departure_model.save(output / "poi_departures.json")
    validation_rows = (
        (poi_id, rows) for poi_id, rows in _poi_histories(root, objects, seed, epochs)
        if poi_id in pois and _split(poi_id, seed)
    )
    metrics = _evaluate_poi(validation_rows, pois, mta, arrival_model, departure_model)
    total = train_count + validation_count
    return {"kind": "poi", "objects_mode": objects, "sampled_objects": total,
            "train_objects": train_count, "validation_objects": validation_count,
            "epochs": epochs, **metrics}


def _evaluate_road(histories, edges, model):
    error = baseline_error = n = 0
    for edge_id, rows in histories:
        edge = edges[edge_id]
        counts = [row[0] for row in rows]
        for slot in range(1, 96):
            extra = road_extra(float(edge["length_m"]), float(edge["speed_limit_kmh"]), rows[slot - 1][2])
            prediction = model.predict(counts[max(0, slot - 4):slot], slot, extra)
            error += abs(prediction - counts[slot])
            baseline_error += abs(counts[slot - 1] - counts[slot])
            n += 1
    return {"vehicle_mae": round(error / max(1, n), 4),
            "persistence_mae": round(baseline_error / max(1, n), 4), "validation_rows": n}


def train_road(root: Path, output: Path, objects: str, epochs: int, seed: int,
                progress_every: int, log_path: Path | None = None) -> dict:
    model = CountModel.create("road_background_vehicles_15min", 3)
    for epoch in range(epochs):
        train_count = validation_count = 0
        if objects.lower() == "all":
            stream = iter_road_histories_with_edges(
                root / "dynamic" / "road_state_15min.txt", root / "static" / "edge.txt"
            )
        else:
            histories = list(sample_histories(root / "dynamic" / "road_state_15min.txt", "road", int(objects), seed + epoch + 1))
            edges = load_edges(root / "static" / "edge.txt", {identifier for identifier, _ in histories})
            stream = ((edge_id, edges[edge_id], rows) for edge_id, rows in histories if edge_id in edges)
        for edge_id, edge, rows in stream:
            if _split(edge_id, seed):
                validation_count += 1
                continue
            train_count += 1
            if progress_every and train_count % progress_every == 0:
                _log(f"road epoch {epoch + 1}/{epochs}: processed {train_count:,} training edges", log_path)
            counts = [row[0] for row in rows]
            for slot in range(1, 96):
                extra = road_extra(float(edge["length_m"]), float(edge["speed_limit_kmh"]), rows[slot - 1][2])
                model.update(counts[max(0, slot - 4):slot], slot, extra, counts[slot], 0, 0.003)
        _log(f"road epoch {epoch + 1}/{epochs} complete; train={train_count:,}, validation={validation_count:,}", log_path)
    model.save(output / "road_background.json")
    if objects.lower() == "all":
        validation_rows = ((edge_id, edge, rows) for edge_id, edge, rows in iter_road_histories_with_edges(
            root / "dynamic" / "road_state_15min.txt", root / "static" / "edge.txt"
        ) if _split(edge_id, seed))
    else:
        histories = list(sample_histories(root / "dynamic" / "road_state_15min.txt", "road", int(objects), seed + epochs + 1))
        edges = load_edges(root / "static" / "edge.txt", {identifier for identifier, _ in histories})
        validation_rows = ((edge_id, edges[edge_id], rows) for edge_id, rows in histories if edge_id in edges and _split(edge_id, seed))
    # Evaluate one object at a time to keep the full run bounded in memory.
    error = baseline = n = 0
    for edge_id, edge, rows in validation_rows:
        counts = [row[0] for row in rows]
        for slot in range(1, 96):
            extra = road_extra(float(edge["length_m"]), float(edge["speed_limit_kmh"]), rows[slot - 1][2])
            error += abs(model.predict(counts[max(0, slot - 4):slot], slot, extra) - counts[slot])
            baseline += abs(counts[slot - 1] - counts[slot])
            n += 1
    metrics = {"vehicle_mae": round(error / max(1, n), 4), "persistence_mae": round(baseline / max(1, n), 4), "validation_rows": n}
    total = train_count + validation_count
    return {"kind": "road", "objects_mode": objects, "sampled_objects": total,
            "train_objects": train_count, "validation_objects": validation_count,
            "epochs": epochs, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--kind", choices=("poi", "road", "both"), default="both")
    parser.add_argument("--objects", type=str, default="3000",
                        help="number of objects for a smoke run, or 'all' for sequential full training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="write a progress line every N training objects")
    parser.add_argument("--log", type=Path, default=None,
                        help="append timestamped progress to this file")
    args = parser.parse_args()
    if args.epochs < 1 or (args.objects.lower() != "all" and int(args.objects) < 10):
        parser.error("--epochs must be positive and --objects at least 10, or 'all'")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("", encoding="utf-8")
    _log(f"starting training: kind={args.kind}, objects={args.objects}, epochs={args.epochs}, seed={args.seed}", args.log)
    results = []
    if args.kind in ("poi", "both"):
        _log("starting POI full-stream pass", args.log)
        results.append(train_poi(args.root, args.output, args.objects, args.epochs, args.seed,
                                 args.progress_every, args.log))
    if args.kind in ("road", "both"):
        _log("starting road full-stream pass", args.log)
        results.append(train_road(args.root, args.output, args.objects, args.epochs, args.seed,
                                  args.progress_every, args.log))
    report = {"dataset": "nyc_v4_final", "time_step_min": 15,
              "provenance": "synthetic disaggregation of four six-hour proxy states",
              "evaluation_limit": "one undated synthetic day; held-out objects do not validate real-world forecasting",
              "seed": args.seed, "results": results}
    path = args.output / "training_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log("training finished", args.log)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
