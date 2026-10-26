import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.build_store import build
from poi_road_sim.parameters import ParameterConfig, poi_parameters, road_parameters
from poi_road_sim.state_store import StateStore


class ParameterTests(unittest.TestCase):
    def test_all_parameters_are_positive_and_reproducible(self):
        config = ParameterConfig(seed=21, spread=0.3)
        poi_history = [(3.0, 1.0, 18.0), (8.0, 3.0, 30.0)]
        road_history = [(4.0, 0.0, 25.0), (12.0, 0.0, 20.0)]
        poi = poi_parameters(7, -1, poi_history, config)
        road = road_parameters(9, -1, road_history, config)
        self.assertEqual(poi, poi_parameters(7, -1, poi_history, config))
        self.assertEqual(road, road_parameters(9, -1, road_history, config))
        self.assertGreaterEqual(poi["capacity_people"], 30)
        self.assertGreater(poi["service_people_per_slot"], 0)
        self.assertGreater(road["capacity_veh_per_slot"], 0)
        self.assertGreater(road["speed_limit_kmh"], 0)
        self.assertEqual(road["speed_source"], "observed_or_default")
        self.assertEqual(poi_parameters(7, 100, poi_history, config)["capacity_source"], "static")
        self.assertNotEqual(road["capacity_veh_per_slot"], road_parameters(
            9, -1, road_history, ParameterConfig(seed=21, spread=0.3, road_capacity_scale=3)
        )["capacity_veh_per_slot"])

    def test_full_store_assigns_parameters_to_every_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "static").mkdir(parents=True)
            (root / "dynamic").mkdir()
            (root / "static" / "POI.txt").write_text(
                "2\t0\n10\t0\t0\t7\t-1\t-1\t-1\t-1\n20\t0\t0\t7\t50\t-1\t-1\t-1\n"
            )
            (root / "static" / "edge.txt").write_text(
                "3\t2\n1\t1\t2\t10\t-1\n2\t2\t3\t10\t40\n"
            )
            with (root / "dynamic" / "POI_state_mta_15min.txt").open("w") as out:
                out.write("2\ntime\tpoi_id\tarrivals\tdepartures\toccupancy\tqueue_length\twait_time_min\tcrowd_proxy\n")
                for poi_id in (10, 20):
                    for slot in range(96):
                        out.write(f"{slot // 4:02d}:{slot % 4 * 15:02d}\t{poi_id}\t2\t1\t12\t0\t0\t0\n")
            with (root / "dynamic" / "road_state_15min.txt").open("w") as out:
                out.write("2\ntime\tedge_id\tvehicles\tflow\tspeed\ttravel\n")
                for edge_id in (1, 2):
                    for slot in range(96):
                        out.write(f"{slot // 4:02d}:{slot % 4 * 15:02d}\t{edge_id}\t3\t12\t25\t2\n")
            output = Path(directory) / "store"
            self.assertEqual(build(root, output, 2, 1), {"pois": 2, "roads": 2, "in_transit": 0})
            with StateStore(output, shards=2) as store:
                for poi_id in (10, 20):
                    state, _ = store.get("poi", poi_id)
                    self.assertGreater(state["capacity_people"], 0)
                    self.assertGreater(state["service_people_per_slot"], 0)
                for edge_id in (1, 2):
                    state, _ = store.get("road", edge_id)
                    self.assertGreater(state["capacity_veh_per_slot"], 0)
                    self.assertGreater(state["speed_limit_kmh"], 0)
            self.assertEqual(json.loads((output / "parameter_manifest.json").read_text())
                             ["road_unit"], "directed edge")


if __name__ == "__main__":
    unittest.main()
