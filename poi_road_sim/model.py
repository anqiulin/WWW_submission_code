"""Small shared log-count models for the observation-calibrated baseline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def count_features(history: list[float], slot: int, extra: tuple[float, ...]) -> list[float]:
    if not history:
        raise ValueError("At least one prior count is required")
    recent = history[-4:]
    phase = 2 * math.pi * (slot % 96) / 96
    return [
        1.0,
        math.log1p(max(0.0, history[-1])) / 5,
        math.log1p(max(0.0, sum(recent) / len(recent))) / 5,
        math.sin(phase), math.cos(phase),
        math.sin(2 * phase), math.cos(2 * phase),
        *extra,
    ]


@dataclass
class CountModel:
    name: str
    weights: list[float]
    group_bias: dict[int, float] = field(default_factory=dict)
    training_provenance: str = "synthetic_15min_disaggregation"

    @classmethod
    def create(cls, name: str, extra_count: int) -> "CountModel":
        return cls(name=name, weights=[0.0] * (7 + extra_count))

    def predict(self, history: list[float], slot: int, extra: tuple[float, ...], group: int = 0) -> float:
        features = count_features(history, slot, extra)
        if len(features) != len(self.weights):
            raise ValueError(f"{self.name} expects {len(self.weights) - 7} extra features")
        log_mean = sum(w * x for w, x in zip(self.weights, features)) + self.group_bias.get(group, 0.0)
        return max(0.0, math.expm1(min(12.0, max(0.0, log_mean))))

    def update(
        self, history: list[float], slot: int, extra: tuple[float, ...],
        target: float, group: int, learning_rate: float,
    ) -> None:
        features = count_features(history, slot, extra)
        prediction = sum(w * x for w, x in zip(self.weights, features)) + self.group_bias.get(group, 0.0)
        error = max(-4.0, min(4.0, prediction - math.log1p(max(0.0, target))))
        for index, feature in enumerate(features):
            self.weights[index] -= learning_rate * (error * feature + 0.0001 * self.weights[index])
        if group:
            current = self.group_bias.get(group, 0.0)
            self.group_bias[group] = current - learning_rate * (error + 0.01 * current)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "name": self.name, "weights": self.weights, "group_bias": self.group_bias,
            "training_provenance": self.training_provenance,
        }, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CountModel":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=value["name"], weights=[float(x) for x in value["weights"]],
            group_bias={int(k): float(v) for k, v in value["group_bias"].items()},
            training_provenance=value["training_provenance"],
        )


def poi_extra(occupancy: float, mta_ridership: float) -> tuple[float, float]:
    return (math.log1p(max(0.0, occupancy)) / 7, math.log1p(max(0.0, mta_ridership)) / 10)


def road_extra(length_m: float, speed_limit_kmh: float, last_speed_kmh: float) -> tuple[float, float, float]:
    return (
        math.log1p(max(0.0, length_m)) / 8,
        math.log1p(max(0.0, speed_limit_kmh)) / 5,
        math.log1p(max(0.0, last_speed_kmh)) / 5,
    )
