"""Optional PyTorch implementation of the learned candidate policy.

This module is deliberately optional: the simulator and the MPC baseline use
only the Python standard library.  Install the CPU wheel listed in
``requirements-neural.txt`` before importing this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Neural policy requires PyTorch. Install code/requirements-neural.txt"
        ) from exc
    return torch, nn


class MLPPolicy:
    """Shared MLP scorer for variable-size candidate sets."""

    def __init__(self, feature_count: int = 9, hidden_size: int = 64):
        torch, nn = _torch()
        self.torch = torch
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def scores(self, feature_vectors: Sequence[Sequence[float]]):
        values = self.torch.tensor(feature_vectors, dtype=self.torch.float32)
        return self.network(values).squeeze(-1)

    def fit_imitation(self, examples, epochs: int = 10, learning_rate: float = 1e-3) -> dict[str, float]:
        """Fit cross entropy to MPC-selected candidate indices."""
        torch = self.torch
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        total = 0.0
        count = 0
        for _ in range(epochs):
            for vectors, label in examples:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    self.scores(vectors).unsqueeze(0), torch.tensor([label]))
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                count += 1
        return {"cross_entropy": total / max(1, count), "examples": float(count)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.network.state_dict(), path)

    def load(self, path: Path) -> None:
        self.network.load_state_dict(self.torch.load(path, map_location="cpu"))


class PPOCandidatePolicy:
    """PPO policy for variable-size POI--route candidate sets.

    A transition stores all request decisions within one physical simulator
    slot and one slot reward. Request ratios are clipped separately; the value
    head pools the valid candidate encodings across the whole slot.
    """

    def __init__(self, feature_count: int = 22, hidden_size: int = 64,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        torch, nn = _torch()
        self.feature_count = feature_count
        self.torch = torch
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.encoder = nn.Sequential(nn.Linear(feature_count, hidden_size), nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.policy_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def _forward(self, feature_vectors, action_mask=None):
        values = self.torch.tensor(feature_vectors, dtype=self.torch.float32)
        hidden = self.encoder(values)
        logits = self.policy_head(hidden).squeeze(-1)
        if action_mask is not None:
            mask = self.torch.tensor(action_mask, dtype=self.torch.bool)
            logits = logits.masked_fill(~mask, -1e9)
        value = self.value_head(hidden.mean(dim=0)).squeeze(-1)
        return logits, value

    def distribution(self, feature_vectors, action_mask=None):
        return self.torch.distributions.Categorical(logits=self._forward(feature_vectors, action_mask)[0])

    def act(self, feature_vectors, deterministic: bool = False, action_mask=None):
        with self.torch.no_grad():
            logits, value = self._forward(feature_vectors, action_mask)
            if deterministic:
                return int(logits.argmax().item()), 0.0, float(value)
            distribution = self.torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            return int(action.item()), float(distribution.log_prob(action)), float(value)

    def choose(self, feature_vectors, action_mask=None) -> int:
        """Select a deterministic action without constructing a distribution."""
        with self.torch.no_grad():
            values = self.torch.tensor(feature_vectors, dtype=self.torch.float32)
            logits = self.policy_head(self.encoder(values)).squeeze(-1)
            if action_mask is not None:
                mask = self.torch.tensor(action_mask, dtype=self.torch.bool)
                logits = logits.masked_fill(~mask, -1e9)
            return int(logits.argmax().item())

    def _pack_slot(self, decisions):
        torch = self.torch
        width = max(len(item["features"]) for item in decisions)
        feature_count = len(decisions[0]["features"][0])
        features = torch.zeros((len(decisions), width, feature_count), dtype=torch.float32)
        masks = torch.zeros((len(decisions), width), dtype=torch.bool)
        actions = torch.tensor([int(item["action"]) for item in decisions], dtype=torch.long)
        for index, item in enumerate(decisions):
            count = len(item["features"])
            features[index, :count] = torch.as_tensor(item["features"], dtype=torch.float32)
            masks[index, :count] = torch.as_tensor(
                item.get("action_mask", [True] * count), dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self._slot_forward(features, masks)
            initial_log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
        old_log_probs = torch.tensor([
            float(item.get("old_log_prob", initial_log_probs[index]))
            for index, item in enumerate(decisions)
        ], dtype=torch.float32)
        return features, masks, actions, old_log_probs

    def _slot_forward(self, features, masks):
        hidden = self.encoder(features)
        logits = self.policy_head(hidden).squeeze(-1).masked_fill(~masks, -1e9)
        pooled = hidden[masks].mean(dim=0)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def _clipped_policy_loss(self, log_probs, old_log_probs, advantage):
        """Average clipped request surrogates for a shared slot advantage."""
        ratio = self.torch.exp((log_probs - old_log_probs).clamp(-20.0, 20.0))
        return -self.torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage).mean()

    @staticmethod
    def _time_step_targets(rewards, dones, values, gamma, gae_lambda):
        """GAE on simulator slots, including slots with no requests."""
        advantages = [0.0] * len(rewards)
        running = 0.0
        for index in reversed(range(len(rewards))):
            next_value = values[index + 1] if index + 1 < len(values) else 0.0
            mask = 0.0 if dones[index] else 1.0
            delta = rewards[index] + gamma * next_value * mask - values[index]
            running = delta + gamma * gae_lambda * mask * running
            advantages[index] = running
        return advantages, [adv + value for adv, value in zip(advantages, values)]

    def _expert_batch_loss(self, examples):
        torch = self.torch
        width = max(len(item["features"]) for item in examples)
        features = torch.zeros((len(examples), width, self.feature_count), dtype=torch.float32)
        masks = torch.zeros((len(examples), width), dtype=torch.bool)
        actions = torch.empty(len(examples), dtype=torch.long)
        for index, item in enumerate(examples):
            count = len(item["features"])
            features[index, :count] = torch.as_tensor(item["features"], dtype=torch.float32)
            masks[index, :count] = torch.as_tensor(
                item.get("action_mask", [True] * count), dtype=torch.bool)
            actions[index] = int(item["action"])
        logits, _ = self._slot_forward(features, masks)
        return torch.nn.functional.cross_entropy(logits, actions)

    def update(self, transitions, epochs: int = 10, learning_rate: float = 3e-4,
               gamma: float = 0.95, gae_lambda: float = 0.95,
               batch_size: int = 64,
               progress: Callable[[int, int], None] | None = None,
               expert_examples=None, bc_coef: float = 0.1,
               dagger_examples=None, dagger_bc_coef: float = 0.5) -> dict[str, float]:
        """Update PPO with static MPC demonstrations and on-policy DAgger labels."""
        if not transitions:
            raise ValueError("PPO requires at least one transition")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if bc_coef < 0 or dagger_bc_coef < 0:
            raise ValueError("BC coefficients must be nonnegative")
        expert_examples = expert_examples or ()
        dagger_examples = dagger_examples or ()
        torch = self.torch
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.policy_head.parameters()) +
                                     list(self.value_head.parameters()), lr=learning_rate)
        # Legacy one-decision inputs remain accepted for compact policy tests.
        slots = [item if "decisions" in item else {**item, "decisions": [item]}
                 for item in transitions]
        rewards = [float(item["reward"]) for item in slots]
        dones = [bool(item.get("done", False)) for item in slots]
        active = [index for index, item in enumerate(slots) if item["decisions"]]
        if not active:
            raise ValueError("PPO requires at least one slot with a decision")
        packed = {index: self._pack_slot(slots[index]["decisions"]) for index in active}
        with torch.no_grad():
            values = [float(self._slot_forward(packed[index][0], packed[index][1])[1])
                      if item["decisions"] else 0.0 for index, item in enumerate(slots)]
        advantages, returns = self._time_step_targets(rewards, dones, values, gamma, gae_lambda)
        advantage_tensor = torch.tensor([advantages[i] for i in active], dtype=torch.float32)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
        total_loss = 0.0
        total_bc_loss = 0.0
        total_dagger_loss = 0.0
        bc_batches = 0
        minibatches = 0
        for epoch in range(epochs):
            order = torch.randperm(len(active)).tolist()
            for start in range(0, len(order), batch_size):
                optimizer.zero_grad()
                batch = order[start:start + batch_size]
                batch_loss = 0.0
                for active_index in batch:
                    index = active[active_index]
                    features, masks, actions, old_log_probs = packed[index]
                    logits, value = self._slot_forward(features, masks)
                    dist = torch.distributions.Categorical(logits=logits)
                    adv = advantage_tensor[active_index]
                    policy_loss = self._clipped_policy_loss(
                        dist.log_prob(actions), old_log_probs, adv)
                    value_loss = (value - returns[index]) ** 2
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * dist.entropy().mean()
                    (loss / len(batch)).backward()
                    batch_loss += float(loss.detach())
                if expert_examples and bc_coef:
                    sample_count = min(batch_size, len(expert_examples))
                    expert_indices = torch.randperm(len(expert_examples))[:sample_count].tolist()
                    bc_loss = self._expert_batch_loss(
                        [expert_examples[index] for index in expert_indices])
                    (bc_coef * bc_loss).backward()
                    total_bc_loss += float(bc_loss.detach())
                    bc_batches += 1
                if dagger_examples and dagger_bc_coef:
                    sample_count = min(batch_size, len(dagger_examples))
                    dagger_indices = torch.randperm(len(dagger_examples))[:sample_count].tolist()
                    dagger_loss = self._expert_batch_loss(
                        [dagger_examples[index] for index in dagger_indices])
                    (dagger_bc_coef * dagger_loss).backward()
                    total_dagger_loss += float(dagger_loss.detach())
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                               list(self.policy_head.parameters()) +
                                               list(self.value_head.parameters()), 0.5)
                optimizer.step()
                total_loss += batch_loss / len(batch)
                minibatches += 1
            if progress is not None:
                progress(epoch + 1, epochs)
        return {"ppo_loss": total_loss / max(1, minibatches),
                "bc_auxiliary_loss": total_bc_loss / max(1, bc_batches),
                "bc_coef": float(bc_coef),
                "dagger_auxiliary_loss": total_dagger_loss / max(1, minibatches),
                "dagger_bc_coef": float(dagger_bc_coef),
                "mean_reward": sum(rewards) / len(rewards),
                "mean_return": sum(returns) / len(returns),
                "environment_steps": float(len(slots)),
                "decision_steps": float(sum(len(item["decisions"]) for item in slots)),
                "epochs": float(epochs), "batch_size": float(batch_size),
                "minibatches": float(minibatches), "gamma": float(gamma)}

    def fit_behavior_cloning(self, examples, epochs: int = 10,
                             learning_rate: float = 1e-3,
                             batch_size: int = 128) -> dict[str, float]:
        """Warm-start the actor from MPC expert actions before PPO."""
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        torch = self.torch
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.policy_head.parameters()), lr=learning_rate)
        total = 0.0
        count = 0
        for _ in range(epochs):
            order = torch.randperm(len(examples)).tolist()
            for start in range(0, len(order), batch_size):
                batch = [examples[index] for index in order[start:start + batch_size]]
                optimizer.zero_grad()
                loss = self._expert_batch_loss(batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                count += 1
        report = self.evaluate_behavior_cloning(examples, batch_size=batch_size)
        return {**report, "bc_updates": float(count),
                "bc_examples": float(len(examples)),
                "bc_batch_size": float(batch_size)}

    def evaluate_behavior_cloning(self, examples, batch_size: int = 128) -> dict[str, float]:
        """Measure imitation on labeled observations without changing the actor."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        torch = self.torch
        final_correct = 0
        final_loss = 0.0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            width = max(len(item["features"]) for item in batch)
            features = torch.zeros((len(batch), width, self.feature_count), dtype=torch.float32)
            masks = torch.zeros((len(batch), width), dtype=torch.bool)
            labels = torch.empty(len(batch), dtype=torch.long)
            for index, item in enumerate(batch):
                candidate_count = len(item["features"])
                features[index, :candidate_count] = torch.as_tensor(
                    item["features"], dtype=torch.float32)
                masks[index, :candidate_count] = torch.as_tensor(
                    item.get("action_mask", [True] * candidate_count), dtype=torch.bool)
                labels[index] = int(item["action"])
            with torch.no_grad():
                logits, _ = self._slot_forward(features, masks)
                final_correct += int((logits.argmax(dim=1) == labels).sum())
                final_loss += float(torch.nn.functional.cross_entropy(
                    logits, labels, reduction="sum"))
        return {"bc_cross_entropy": final_loss / max(1, len(examples)),
                "bc_accuracy": final_correct / max(1, len(examples))}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"feature_count": self.feature_count,
                         "encoder": self.encoder.state_dict(),
                         "policy_head": self.policy_head.state_dict(),
                         "value_head": self.value_head.state_dict()}, path)

    def load(self, path: Path) -> None:
        state = self.torch.load(path, map_location="cpu")
        stored = state.get("feature_count", state["encoder"]["0.weight"].shape[1])
        if stored != self.feature_count:
            raise ValueError(f"checkpoint has {stored} features; current scenario needs {self.feature_count}")
        self.encoder.load_state_dict(state["encoder"])
        self.policy_head.load_state_dict(state["policy_head"])
        self.value_head.load_state_dict(state["value_head"])
