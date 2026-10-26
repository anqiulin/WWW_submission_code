import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.grid_index import GridQuery, POIGridIndex, generate_grid_candidates


class GridIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = POIGridIndex(0.01)
        self.index.add(10, 40.704, -74.004, 7)  # corridor
        self.index.add(20, 40.745, -74.000, 7)  # one expansion ring
        self.index.add(30, 40.710, -74.000, 8)  # wrong category
        self.query = GridQuery("grid-u", 1, 40.700, -74.005, 9, 40.720, -74.005,
                               2, category=7, cell_size_deg=0.01, max_layers=3, max_pois=5)

    def test_corridor_then_ring_and_category(self):
        result = self.index.search(self.query)
        self.assertEqual([row[0] for row in result], [10, 20])
        self.assertEqual(result[0][2], 0)
        self.assertGreaterEqual(result[1][2], result[0][2])

    def test_grid_candidates_use_cached_route_factory(self):
        candidates = generate_grid_candidates(self.index, self.query, lambda poi_id: (1,))
        self.assertEqual([item.request.destination_poi_id for item in candidates], [10, 20])
        self.assertTrue(all(item.request.edge_ids == (1,) for item in candidates))

    def test_grid_candidates_can_attach_cached_egress(self):
        candidates = generate_grid_candidates(self.index, self.query, lambda poi_id: (1,),
                                              lambda poi_id: (2,))
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].request.final_destination_node_id, 9)
        self.assertEqual(candidates[0].request.egress_edge_ids, (2,))


if __name__ == "__main__":
    unittest.main()
