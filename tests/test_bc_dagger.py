import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BCDaggerTests(unittest.TestCase):
    def test_dagger_sampling_covers_each_request_block(self):
        from poi_road_sim.experiment import _dagger_sample_indices

        sampled = _dagger_sample_indices(64, 4, seed=17)
        self.assertEqual(sampled, _dagger_sample_indices(64, 4, seed=17))
        self.assertEqual(len(sampled), 4)
        self.assertEqual({index // 16 for index in sampled}, {0, 1, 2, 3})
        self.assertEqual(_dagger_sample_indices(64, 4), {0, 16, 32, 48})

    def test_evaluation_does_not_update_actor(self):
        import torch
        from poi_road_sim.torch_policy import PPOCandidatePolicy

        policy = PPOCandidatePolicy(feature_count=2, hidden_size=8)
        examples = [
            {"features": [[1.0, 0.0], [0.0, 1.0]], "action": 0},
            {"features": [[0.0, 1.0], [1.0, 0.0]], "action": 1},
        ]
        before = [weight.clone() for weight in policy.encoder.parameters()]
        report = policy.evaluate_behavior_cloning(examples, batch_size=1)
        self.assertTrue(0 <= report["bc_accuracy"] <= 1)
        self.assertGreaterEqual(report["bc_cross_entropy"], 0)
        self.assertTrue(all(torch.equal(old, new)
                            for old, new in zip(before, policy.encoder.parameters())))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            policy.evaluate_behavior_cloning(examples, batch_size=0)

    def test_validation_requires_every_scenario_to_beat_scair(self):
        from train_bc_dagger import _validate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in ("scenario_A", "scenario_B"):
                path = root / name / "test" / "seed_1" / "scenario.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}")
                paths.append(path)
            loads = iter((8.0, 12.0))
            with patch("train_bc_dagger._load_scenario", return_value={}), \
                 patch("train_bc_dagger.collect_ppo_trajectory",
                       side_effect=lambda *args, **kwargs: [
                           {"global_load": {"total_load": next(loads)}}]):
                result = _validate(
                    object(), paths, root, "local_v2",
                    {"scenario_A": {"rise_mpc": 5.0, "scair": 10.0},
                     "scenario_B": {"rise_mpc": 6.0, "scair": 11.0}},
                )
            self.assertEqual(result["beats_scair_by_scenario"],
                             {"scenario_A": True, "scenario_B": False})
            self.assertFalse(result["passes_scair_gate"])
            self.assertFalse(result["passes_scair_5pct_margin"])
            self.assertAlmostEqual(result["worst_scair_ratio"], 12.0 / 11.0)
            self.assertAlmostEqual(result["mean_mpc_ratio"], 1.8)


if __name__ == "__main__":
    unittest.main()
