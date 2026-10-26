"""MPC expert data generation for the learnable recommendation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .policy import candidate_features
from .recommender import CandidateRoute, MPCRecommender


@dataclass(frozen=True)
class ExpertExample:
    features: tuple[tuple[float, ...], ...]
    label: int


def collect_mpc_examples(simulator, candidate_sets: Sequence[Sequence[CandidateRoute]],
                         horizon: int = 4) -> list[ExpertExample]:
    """Collect one supervised example per request using the current MPC."""
    expert = MPCRecommender(horizon=horizon)
    examples: list[ExpertExample] = []
    for candidates in candidate_sets:
        if not candidates:
            continue
        decision = expert.choose(simulator, candidates)
        if decision.candidate is None:
            continue
        selected_id = decision.candidate.request.request_id
        label = next((i for i, c in enumerate(candidates)
                      if c.request.request_id == selected_id), None)
        if label is not None:
            examples.append(ExpertExample(
                tuple(tuple(candidate_features(simulator, c)) for c in candidates), label))
    return examples


def flatten_examples(examples: Sequence[ExpertExample]) -> list[tuple[list[list[float]], int]]:
    """Convert immutable examples to the format accepted by policy.fit."""
    return [( [list(vector) for vector in item.features], item.label) for item in examples]
