import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_road_sim.experiment import _future_request_map, _set_policy_context
from poi_road_sim.metrics import GlobalLoadWeights, global_load
from poi_road_sim.model import CountModel
from poi_road_sim.simulator import CoupledSimulator, POIState, Request, RoadState
from experiments.build_citywide_core_requests import _route_variants


def model(name, extras):
    result = CountModel.create(name, extras)
    result.weights[0] = math.log1p(0.0)
    return result


class EventDynamicsTests(unittest.TestCase):
    def test_alternate_routes_are_distinct_and_router_is_restored(self):
        class Router:
            def __init__(self):
                self.penalty = None
                self.resets = 0

            def update_travel_times(self, times):
                self.penalty = next(iter(times))

            def route(self, origin, target):
                edges = (1, 4) if self.penalty == 2 else (1, 2)
                return 1.0, edges, (origin, 2, target)

            def reset_travel_times(self):
                self.penalty = None
                self.resets += 1

        router = Router()
        self.assertEqual(_route_variants(router, 1, 3, (1, 2), 2),
                         [(1, 2), (1, 4)])
        self.assertIsNone(router.penalty)
        self.assertGreater(router.resets, 0)

    def simulator(self):
        roads = {edge: RoadState(edge, edge, edge + 1, 100, 36, 1)
                 for edge in (1, 2, 3)}
        pois = {3: POIState(3, 3, 7, 1, 1)}
        return CoupledSimulator(roads, pois, model("road", 3), model("arrival", 2),
                                model("departure", 2), dynamics_mode="event_driven",
                                cost_weights=GlobalLoadWeights(mode="queue_wait"))

    def test_delay_propagates_to_poi_and_egress(self):
        sim = self.simulator()
        request = Request("one", 1, 3, 1, (1, 2), 1, 4, (3,), 1)
        zero = dict(observed_poi_arrivals={3: 0}, observed_poi_departures={3: 0})
        first = sim.step([request], observed_road_background={1: 1, 2: 0, 3: 0}, **zero)
        self.assertEqual(first["roads"][1]["recommended_vehicles"], 0)
        self.assertEqual(first["roads"][2]["recommended_vehicles"], 0)
        self.assertEqual(global_load(sim, first)["road_load"], 15)
        self.assertEqual(global_load(sim, first)["total_load"],
                         global_load(sim)["total_load"])
        second = sim.step(observed_road_background={1: 0, 2: 0, 3: 0}, **zero)
        self.assertEqual(second["roads"][1]["recommended_vehicles"], 1)
        self.assertEqual(second["roads"][2]["recommended_vehicles"], 0)
        third = sim.step(observed_road_background={1: 0, 2: 0, 3: 0}, **zero)
        self.assertEqual(third["roads"][2]["recommended_vehicles"], 1)
        fourth = sim.step(observed_road_background={1: 0, 2: 0, 3: 0}, **zero)
        self.assertEqual(fourth["pois"][3]["recommendation_arrivals"], 1)
        self.assertEqual(fourth["roads"][3]["recommended_vehicles"], 0)
        fifth = sim.step(observed_road_background={1: 0, 2: 0, 3: 0}, **zero)
        self.assertEqual(fifth["pois"][3]["scheduled_recommendation_departures"], 1)
        self.assertEqual(fifth["roads"][3]["recommended_vehicles"], 1)
        self.assertEqual(sim.travelers["one"].status, "in_transit")
        sim.step(observed_road_background={1: 0, 2: 0, 3: 0}, **zero)
        self.assertEqual(sim.travelers["one"].status, "complete")

    def test_fork_keeps_mutable_event_state_independent(self):
        sim = self.simulator()
        trial = sim.fork()
        trial.step([Request("new", 1, 3, 1, (1, 2))],
                   observed_road_background={1: 0, 2: 0, 3: 0},
                   observed_poi_arrivals={3: 0}, observed_poi_departures={3: 0})
        self.assertEqual(sim.slot, 0)
        self.assertEqual(sim.travelers, {})
        self.assertEqual(sim.roads[1].count_history, [0.0])

    def test_forecast_depends_only_on_observed_requests(self):
        sim = self.simulator()
        future_a = {1: [{"request_id": "hidden-a", "forecast_poi_id": 3}]}
        future_b = {1: [{"request_id": "hidden-b", "forecast_poi_id": 999}]}
        self.assertEqual(_future_request_map(sim, 4), {})
        _set_policy_context(sim, {"policy_use_forecast": True, "requests": future_a}, 4)
        first = dict(sim.policy_forecast_by_poi)
        _set_policy_context(sim, {"policy_use_forecast": True, "requests": future_b}, 4)
        self.assertEqual(first, sim.policy_forecast_by_poi)
        sim.policy_recent_requests = (Request("seen", 1, 3, 1, (1, 2)),)
        self.assertEqual(len(_future_request_map(sim, 4)[1]), 1)
        _set_policy_context(sim, {"policy_use_forecast": True}, 4)
        self.assertEqual(sim.policy_forecast_by_poi, {3: 3.0})


if __name__ == "__main__":
    unittest.main()
