import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.model import CountModel
from poi_road_sim.recommender import CandidateRoute, MPCRecommender
from poi_road_sim.routes import UserQuery
from poi_road_sim.simulator import CoupledSimulator, POIState, Request, RoadState


def constant_model(name, value, extra_count):
    model = CountModel.create(name, extra_count)
    model.weights[0] = math.log1p(value)
    return model


class RecommenderTests(unittest.TestCase):
    def simulator(self):
        roads = {
            1: RoadState(1, 1, 2, 100, 36, 5, [0], 36),
            2: RoadState(2, 1, 3, 100, 36, 50, [0], 36),
        }
        pois = {
            2: POIState(2, 2, 7, 100, 20, occupancy=0),
            3: POIState(3, 3, 7, 100, 20, occupancy=0),
        }
        return CoupledSimulator(roads, pois, constant_model("road", 4, 3),
                                constant_model("arrivals", 0, 2), constant_model("departures", 0, 2))

    def test_counterfactual_choice_keeps_base_unchanged(self):
        sim = self.simulator()
        candidates = [
            CandidateRoute(Request("busy", 1, 2, 2, (1,)), "busy"),
            CandidateRoute(Request("free", 1, 3, 2, (2,)), "free"),
        ]
        decision = MPCRecommender(horizon=3).choose(sim, candidates)
        self.assertEqual(decision.candidate.label, "free")
        self.assertEqual(sim.slot, 0)
        self.assertEqual(decision.evaluated, 2)
        self.assertLess(decision.details[0]["score"], decision.details[1]["score"])
        reference = next(item for item in decision.details if item["label"] == "busy")
        self.assertAlmostEqual(reference["road_externality"], 0.0)
        self.assertAlmostEqual(reference["poi_externality"], 0.0)

    def test_selected_action_is_the_only_action_executed(self):
        sim = self.simulator()
        candidates = [CandidateRoute(Request("free", 1, 3, 2, (2,)), "free")]
        decision, result = MPCRecommender(horizon=2).recommend_and_step(sim, candidates)
        self.assertEqual(decision.candidate.label, "free")
        self.assertEqual(result["roads"][2]["recommended_vehicles"], 1)
        self.assertEqual(sim.slot, 1)

    def test_query_generates_candidates_before_scoring(self):
        sim = self.simulator()
        decision = MPCRecommender(horizon=2).choose_query(
            sim, UserQuery("generated", origin_node_id=1, people=2, category=7)
        )
        self.assertIsNotNone(decision.candidate)

    def test_sampled_mpc_keeps_full_request_batch(self):
        from poi_road_sim.experiment import _choose_strategy, ExperimentConfig
        sim = self.simulator()
        candidate_sets = [
            [CandidateRoute(Request(f"user-{index}-busy", 1, 2, 1, (1,)), "busy"),
             CandidateRoute(Request(f"user-{index}-free", 1, 3, 1, (2,)), "free")]
            for index in range(4)
        ]
        selected = _choose_strategy(
            "rise_mpc_sampled", sim, candidate_sets,
            ExperimentConfig(mpc_per_slot=1))
        self.assertEqual(len(selected), 4)
        self.assertEqual(sim.slot, 0)
        self.assertEqual(len(sim.active_requests), 0)


if __name__ == "__main__":
    unittest.main()
