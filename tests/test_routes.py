import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.model import CountModel
from poi_road_sim.recommender import MPCRecommender
from poi_road_sim.routes import UserQuery, generate_candidates, k_shortest_paths
from poi_road_sim.simulator import CoupledSimulator, POIState, RoadState


def model(name, value, extras):
    result = CountModel.create(name, extras)
    result.weights[0] = math.log1p(value)
    return result


class RouteTests(unittest.TestCase):
    def roads(self):
        return {
            1: RoadState(1, 1, 2, 100, 36, 100, [0], 36),
            2: RoadState(2, 2, 4, 100, 36, 100, [0], 36),
            3: RoadState(3, 1, 3, 250, 36, 100, [0], 36),
            4: RoadState(4, 3, 4, 100, 36, 100, [0], 36),
        }

    def test_k_shortest_paths_are_connected_and_distinct(self):
        paths = k_shortest_paths(self.roads(), 1, 4, 2)
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0][1], paths[1][1])
        self.assertLess(paths[0][0], paths[1][0])

    def test_candidate_generation_filters_category(self):
        pois = {4: POIState(4, 4, 7, 50, 5), 3: POIState(3, 3, 8, 50, 5)}
        candidates = generate_candidates(self.roads(), pois, UserQuery("u", 1, 2, category=7))
        self.assertTrue(candidates)
        self.assertTrue(all(item.request.destination_poi_id == 4 for item in candidates))

    def test_batch_assignments_execute_together(self):
        roads = {1: RoadState(1, 1, 2, 100, 36, 50, [0], 36)}
        pois = {2: POIState(2, 2, 7, 100, 20)}
        sim = CoupledSimulator(roads, pois, model("r", 0, 3), model("a", 0, 2), model("d", 0, 2))
        candidates = []
        from poi_road_sim.recommender import CandidateRoute
        from poi_road_sim.simulator import Request
        for index in (1, 2):
            candidates.append([CandidateRoute(Request(f"u{index}", 1, 2, 1, (1,)), str(index))])
        decisions, result = MPCRecommender(horizon=2).recommend_batch_and_step(sim, candidates)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(result["roads"][1]["recommended_vehicles"], 2)
        self.assertEqual(sim.slot, 1)


if __name__ == "__main__":
    unittest.main()
