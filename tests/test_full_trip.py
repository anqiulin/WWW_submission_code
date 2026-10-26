import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.model import CountModel
from poi_road_sim.simulator import CoupledSimulator, POIState, Request, RoadState


def model(name, value, extra):
    result = CountModel.create(name, extra)
    result.weights[0] = math.log1p(value)
    return result


class FullTripTests(unittest.TestCase):
    def test_poi_stop_schedules_egress_and_departure(self):
        roads = {
            1: RoadState(1, 1, 2, 100, 36, 20, [0], 36),
            2: RoadState(2, 2, 3, 100, 36, 20, [0], 36),
        }
        pois = {2: POIState(2, 2, 7, 20, 20, occupancy=0)}
        sim = CoupledSimulator(roads, pois, model("r", 0, 3),
                               model("a", 0, 2), model("d", 0, 2))
        request = Request("trip", 1, 2, 3, (1,), 1, 3, (2,), 1)
        first = sim.step([request])
        self.assertEqual(first["request_arrival_slots"], [1])
        self.assertEqual(first["pois"][2]["recommendation_arrivals"], 3)
        second = sim.step()
        self.assertEqual(second["pois"][2]["scheduled_recommendation_departures"], 3)
        self.assertEqual(second["roads"][2]["recommended_vehicles"], 1)
        self.assertEqual(sim.step()["pois"][2]["scheduled_recommendation_departures"], 0)

    def test_observed_background_departures_add_to_scheduled_visitors(self):
        roads = {1: RoadState(1, 1, 2, 100, 36, 20, [0], 36),
                 2: RoadState(2, 2, 3, 100, 36, 20, [0], 36)}
        pois = {2: POIState(2, 2, 7, 20, 20, occupancy=2)}
        sim = CoupledSimulator(roads, pois, model("r", 0, 3),
                               model("a", 0, 2), model("d", 0, 2))
        request = Request("trip", 1, 2, 3, (1,), 1, 3, (2,), 1)
        sim.step([request], observed_road_background={1: 0, 2: 0},
                 observed_poi_arrivals={2: 0}, observed_poi_departures={2: 0})
        second = sim.step(observed_road_background={1: 0, 2: 0},
                          observed_poi_arrivals={2: 0}, observed_poi_departures={2: 1})
        self.assertEqual(second["pois"][2]["scheduled_recommendation_departures"], 3)
        self.assertEqual(second["pois"][2]["departures"], 4)


if __name__ == "__main__":
    unittest.main()
