import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poi_road_sim.state_store import StateStore


class StateStoreTests(unittest.TestCase):
    def test_full_batch_update_and_observation_override(self):
        with tempfile.TemporaryDirectory() as directory, StateStore(Path(directory), shards=3) as store:
            store.upsert_pois(((1, {"occupancy": 2}, 0), (2, {"occupancy": 4}, 0)))
            store.upsert_roads(((10, {"vehicles": 3}, 0),))
            self.assertEqual(store.counts(), {"pois": 2, "roads": 1, "in_transit": 0})
            store.update_observation("poi", 1, {"occupancy": 9}, 4)
            self.assertEqual(store.get("poi", 1), ({"occupancy": 9}, 4))
            self.assertEqual(store.snapshot("poi", [1, 2, 3])[1], ({"occupancy": 9}, 4))
            self.assertEqual(sum(len(batch) for batch in store.iter_batches("poi", 1)), 2)

    def test_in_transit_is_persistent_and_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory, StateStore(Path(directory), shards=2) as store:
            store.add_in_transit("r1", 3, {"poi_id": 7, "people": 2})
            self.assertEqual(store.pop_arrivals(2), [])
            self.assertEqual(store.pop_arrivals(3)[0]["people"], 2)
            self.assertEqual(store.pop_arrivals(3), [])


if __name__ == "__main__":
    unittest.main()
