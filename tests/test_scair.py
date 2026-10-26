import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.recommender import CandidateRoute
from poi_road_sim.scair import ScairAdapter
from poi_road_sim.simulator import POIState, Request


class ScairTests(unittest.TestCase):
    def test_current_crowd_changes_assignment_without_future_rollout(self):
        pois = {
            1: POIState(1, 1, 7, 100, 10, occupancy=0, waiting=0),
            2: POIState(2, 2, 7, 100, 10, occupancy=0, waiting=20),
        }
        candidates = [
            CandidateRoute(Request("u-a", 0, 1, 1, (1,)), "a"),
            CandidateRoute(Request("u-b", 0, 2, 1, (2,)), "b"),
        ]
        selected = ScairAdapter(popularity={1: 10, 2: 10}).choose_one(candidates, pois)
        self.assertEqual(selected.label, "a")

    def test_city_adapter_uses_batch_overlap(self):
        pois = {1: POIState(1, 1, 7, 100, 10), 2: POIState(2, 2, 7, 100, 10)}
        sets = [[
            CandidateRoute(Request("u1-a", 0, 1, 1, (1,)), "a1"),
            CandidateRoute(Request("u1-b", 0, 2, 1, (2,)), "b1"),
        ], [
            CandidateRoute(Request("u2-a", 0, 1, 1, (1,)), "a2"),
            CandidateRoute(Request("u2-b", 0, 2, 1, (2,)), "b2"),
        ]]
        selected = ScairAdapter(popularity={1: 10, 2: 10}).choose(sets, pois)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
