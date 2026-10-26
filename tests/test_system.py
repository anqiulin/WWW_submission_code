import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.data import sample_histories, slot_of, time_of
from poi_road_sim.model import CountModel
from poi_road_sim.simulator import CoupledSimulator, POIState, Request, RoadState, bpr_travel_time


def constant_model(name, value, extra_count):
    model = CountModel.create(name, extra_count)
    model.weights[0] = math.log1p(value)
    return model


class DataTests(unittest.TestCase):
    def test_slot_roundtrip_and_bad_time(self):
        for slot in range(96):
            self.assertEqual(slot_of(time_of(slot)), slot)
        with self.assertRaises(ValueError):
            slot_of("24:00")

    def test_random_access_keeps_complete_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.txt"
            with path.open("w") as out:
                out.write("2\ntime\tpoi_id\tarrivals\tdepartures\toccupancy\tqueue_length\twait_time_min\tcrowd_proxy\n")
                for poi_id in (10, 20):
                    for slot in range(96):
                        out.write(f"{time_of(slot)}\t{poi_id}\t1\t0\t2\t0\t0\t0.1\n")
            items = list(sample_histories(path, "poi", 2, 4))
            self.assertEqual({item[0] for item in items}, {10, 20})
            self.assertTrue(all(len(item[1]) == 96 for item in items))


class ModelTests(unittest.TestCase):
    def test_count_model_learns_and_roundtrips(self):
        model = CountModel.create("test", 2)
        for _ in range(300):
            model.update([3, 4, 3, 4], 12, (0.1, 0.2), 4, 10, 0.01)
        self.assertLess(abs(model.predict([3, 4, 3, 4], 12, (0.1, 0.2), 10) - 4), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertAlmostEqual(model.predict([3, 4], 12, (0.1, 0.2), 10),
                                   CountModel.load(path).predict([3, 4], 12, (0.1, 0.2), 10))


class SimulatorTests(unittest.TestCase):
    def simulator(self):
        roads = {1: RoadState(1, 1, 2, 100, 36, 20, [0], 36)}
        pois = {2: POIState(2, 2, 7, 5, 3, occupancy=2, waiting=0)}
        return CoupledSimulator(roads, pois, constant_model("road", 0, 3),
                                constant_model("arrivals", 2, 2), constant_model("departures", 1, 2))

    def test_route_arrival_and_conservation(self):
        sim = self.simulator()
        result = sim.step([Request("r1", 1, 2, 3, (1,))])
        self.assertEqual(result["request_arrival_slots"], [0])
        self.assertEqual(result["roads"][1]["recommended_vehicles"], 1)
        poi = result["pois"][2]
        self.assertAlmostEqual(poi["arrivals"], 5)
        self.assertAlmostEqual(poi["occupancy"], 4)
        self.assertAlmostEqual(poi["waiting"], 2)
        self.assertAlmostEqual(2 + poi["admitted"] - poi["departures"], poi["occupancy"])
        later = sim.step()["pois"][2]
        self.assertLessEqual(later["occupancy"], 5)
        self.assertAlmostEqual(poi["waiting"] + later["arrivals"] - later["admitted"], later["waiting"])

    def test_rejects_disconnected_route_and_negative_observation(self):
        sim = self.simulator()
        with self.assertRaises(ValueError):
            sim.step([Request("bad", 3, 2, 1, (1,))])
        with self.assertRaises(ValueError):
            sim.step(observed_poi_arrivals={2: -1})

    def test_new_observation_changes_state(self):
        observed = self.simulator().step(observed_poi_arrivals={2: 4})["pois"][2]
        baseline = self.simulator().step()["pois"][2]
        self.assertGreater(observed["arrivals"], baseline["arrivals"])
        self.assertGreaterEqual(observed["occupancy"], baseline["occupancy"])

    def test_full_road_delays_vehicle_and_poi_arrival(self):
        sim = self.simulator()
        route = Request("delayed", 1, 2, 3, (1,))
        first = sim.step([route], observed_road_background={1: 20})
        self.assertEqual(first["request_arrival_slots"], [1])
        self.assertEqual(first["roads"][1]["recommended_vehicles"], 0)
        self.assertEqual(first["pois"][2]["recommendation_arrivals"], 0)
        second = sim.step()
        self.assertEqual(second["roads"][1]["recommended_vehicles"], 1)
        self.assertEqual(second["pois"][2]["recommendation_arrivals"], 3)

    def test_fork_keeps_counterfactuals_independent(self):
        base = self.simulator()
        policy_a = base.fork()
        policy_b = base.fork()
        policy_a.step([Request("a", 1, 2, 3, (1,))])
        self.assertEqual(base.slot, 0)
        self.assertEqual(policy_b.slot, 0)
        self.assertEqual(policy_b.pois[2].occupancy, 2)

    def test_bpr_is_monotone_and_reports_link_queue(self):
        free = bpr_travel_time(10, 0, 20)
        congested = bpr_travel_time(10, 40, 20)
        self.assertEqual(free, 10)
        self.assertGreater(congested, free)
        sim = self.simulator()
        result = sim.step(observed_road_background={1: 30})["roads"][1]
        self.assertEqual(result["total_inflow_vehicles"], 30)
        self.assertEqual(result["outflow_vehicles"], 20)
        self.assertEqual(result["queue_vehicles"], 10)
        self.assertGreater(result["vc_ratio"], 1)


if __name__ == "__main__":
    unittest.main()
