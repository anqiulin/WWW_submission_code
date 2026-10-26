import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.metrics import GlobalLoadWeights, global_load
from poi_road_sim.policy import LinearCandidatePolicy
from poi_road_sim.simulator import CoupledSimulator, POIState, RoadState
from poi_road_sim.model import CountModel
from experiments.run_nyc_regional_experiment import _route_mean_top_k


def zero_model(name, extras):
    return CountModel.create(name, extras)


class GlobalPolicyTests(unittest.TestCase):
    def test_global_load_counts_all_objects(self):
        roads = {1: RoadState(1, 1, 2, 100, 36, 1, [0], 36),
                 2: RoadState(2, 2, 3, 100, 36, 10, [0], 36)}
        pois = {2: POIState(2, 2, 1, 10, 1, occupancy=10, waiting=2),
                3: POIState(3, 3, 1, 10, 1, occupancy=0, waiting=0)}
        sim = CoupledSimulator(roads, pois, zero_model("r", 3),
                               zero_model("a", 2), zero_model("d", 2))
        result = {"roads": {1: {"total_load_vehicles": 2,
                                "outflow_vehicles": 1, "queue_vehicles": 1,
                                "travel_time_sec": 11.5},
                            2: {"total_load_vehicles": 0,
                                "outflow_vehicles": 0, "queue_vehicles": 0,
                                "travel_time_sec": 10}},
                  "pois": {2: {"occupancy": 10, "waiting": 2},
                           3: {"occupancy": 0, "waiting": 0}}}
        load = global_load(sim, result)
        self.assertAlmostEqual(load["road_load"], 15.025)
        self.assertAlmostEqual(load["poi_load"], 30.0)
        self.assertAlmostEqual(load["total_load"], 45.025)

    def test_reference_scales_normalize_components_without_losing_raw_units(self):
        roads = {1: RoadState(1, 1, 2, 100, 36, 1)}
        pois = {2: POIState(2, 2, 1, 10, 1, waiting=2)}
        sim = CoupledSimulator(roads, pois, zero_model("r", 3),
                               zero_model("a", 2), zero_model("d", 2),
                               cost_weights=GlobalLoadWeights(
                                   road_reference_vehicle_min=15.025,
                                   poi_reference_person_min=30))
        result = {"roads": {1: {"total_load_vehicles": 2,
                                "outflow_vehicles": 1, "queue_vehicles": 1,
                                "travel_time_sec": 11.5}},
                  "pois": {2: {"occupancy": 0, "waiting": 2}}}
        load = global_load(sim, result)
        self.assertAlmostEqual(load["raw_road_vehicle_min"], 15.025)
        self.assertAlmostEqual(load["raw_poi_person_min"], 30)
        self.assertAlmostEqual(load["road_load"], 1)
        self.assertAlmostEqual(load["poi_load"], 1)
        self.assertAlmostEqual(load["total_load"], 2)

    def test_bounded_topk_uses_hot_edges_and_separate_component_scales(self):
        roads = {edge_id: RoadState(edge_id, edge_id, edge_id + 1, 100, 36, 10)
                 for edge_id in range(1, 5)}
        pois = {1: POIState(1, 2, 1, 10, 1, waiting=2)}
        sim = CoupledSimulator(roads, pois, zero_model("r", 3),
                               zero_model("a", 2), zero_model("d", 2),
                               cost_weights=GlobalLoadWeights(mode="bounded_topk",
                                   road_top_k=2, road_reference_vehicle_min=45,
                                   poi_reference_person_min=30))
        result = {"roads": {1: {"total_load_vehicles": 12, "outflow_vehicles": 10,
                                "queue_vehicles": 2, "travel_time_sec": 20},
                             2: {"total_load_vehicles": 10, "outflow_vehicles": 10,
                                 "queue_vehicles": 1, "travel_time_sec": 10},
                             3: {"total_load_vehicles": 10, "outflow_vehicles": 10,
                                 "queue_vehicles": 0.5, "travel_time_sec": 10},
                             4: {"total_load_vehicles": 10, "outflow_vehicles": 10,
                                 "queue_vehicles": 0, "travel_time_sec": 10}},
                  "pois": {1: {"waiting": 2, "occupancy": 10}}}
        load = global_load(sim, result)
        self.assertAlmostEqual(load["raw_topk_road_vehicle_min"], 30 + 10 * 10 / 60 + 15)
        self.assertAlmostEqual(load["raw_road_vehicle_min"] - load["raw_topk_road_vehicle_min"], 7.5)
        self.assertAlmostEqual(load["road_load"], load["raw_topk_road_vehicle_min"] /
                               (load["raw_topk_road_vehicle_min"] + 45))
        self.assertAlmostEqual(load["poi_load"], 0.5)
        self.assertLess(load["road_load"], 1)
        self.assertLess(load["total_load"], 2)

    def test_top_k_uses_full_candidate_routes(self):
        scene = {"requests": [{"candidates": [
            {"edge_ids": [1, 2], "egress_edge_ids": [3]},
            {"edge_ids": [4], "egress_edge_ids": [5, 6, 7, 8]}]},
            {"candidates": [{"edge_ids": [9, 10, 11, 12], "egress_edge_ids": []}]}]}
        self.assertEqual(_route_mean_top_k(scene), 4)

    def test_policy_fit_and_round_trip(self):
        policy = LinearCandidatePolicy.create()
        report = policy.fit([([[1.0] + [0.0] * 8,
                               [0.0] * 9], 0)], epochs=2)
        self.assertEqual(report["examples"], 1.0)
        path = Path(self.id().replace("/", "_") + ".json")
        try:
            policy.save(path)
            loaded = LinearCandidatePolicy.load(path)
            self.assertEqual(len(loaded.weights), 9)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
