import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_four_scenarios import _apply_condition, _sample_request_window, build


class FourScenarioWindowTests(unittest.TestCase):
    def test_6000_requests_are_balanced_across_full_day(self):
        source_rows = [
            {"request_id": f"source-{slot}-{region}-{index}",
             "region_id": region, "departure_slot": slot,
             "destination_poi_id": 1 if index == 0 else 2,
             "edge_ids": [1],
             "candidates": [{"destination_poi_id": 1 if index == 0 else 2,
                             "edge_ids": [1]}]}
            for slot in range(28, 36)
            for region in range(4)
            for index in range(100)
        ]
        rows = _sample_request_window(source_rows, set(range(4)), 0, 96, 6000,
                                      9303, 28)
        expected = {slot: 63 if slot % 8 < 4 else 62 for slot in range(96)}
        self.assertEqual(Counter(row["departure_slot"] for row in rows), expected)
        scene = {
            "requests": rows,
            "roads": [{"edge_id": 1, "capacity_veh_per_slot": 10.0,
                       "speed_limit_kmh": 30.0}],
            "pois": [{"poi_id": poi_id, "category": poi_id,
                      "service_people_per_slot": 2.0, "capacity_people": 20.0}
                     for poi_id in range(1, 101)],
            "provenance": {},
        }
        result = _apply_condition(scene, "C", list(range(192)), 9303, True)
        self.assertEqual(Counter(row["departure_slot"] for row in result["requests"]),
                         expected)

    def test_full_day_fixed_requests_and_poi_concentration(self):
        source_rows = [
            {"request_id": f"source-{slot}-{region}-{index}",
             "region_id": region, "departure_slot": slot,
             "destination_poi_id": 1 if index == 0 else 2,
             "edge_ids": [1],
             "candidates": [{"destination_poi_id": 1 if index == 0 else 2,
                             "edge_ids": [1]}]}
            for slot in range(28, 36)
            for region in range(4)
            for index in range(10)
        ]
        rows = _sample_request_window(source_rows, set(range(4)), 0, 96, 3840,
                                      9303, 28)
        self.assertEqual(Counter(row["departure_slot"] for row in rows),
                         {slot: 40 for slot in range(96)})
        scene = {
            "requests": rows,
            "roads": [{"edge_id": 1, "capacity_veh_per_slot": 10.0,
                       "speed_limit_kmh": 30.0}],
            "pois": [{"poi_id": poi_id, "category": poi_id,
                      "service_people_per_slot": 2.0, "capacity_people": 20.0}
                     for poi_id in range(1, 101)],
            "provenance": {},
        }
        before = Counter(row["destination_poi_id"] for row in rows
                         if row["departure_slot"] in (2, 3, 4))
        result = _apply_condition(scene, "C", list(range(96)), 9303, True)
        self.assertEqual(Counter(row["departure_slot"] for row in result["requests"]),
                         {slot: 40 for slot in range(96)})
        after = Counter(row["destination_poi_id"] for row in result["requests"]
                        if row["departure_slot"] in (2, 3, 4))
        self.assertGreater(after[2], before[2])
        self.assertEqual(result["profile_observations"][0]["road_background"],
                         result["profile_observations"][8]["road_background"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source/test/citywide/seed_9303/scenario.json"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(json.dumps({
                "start_slot": 28, "requests": source_rows,
                "roads": scene["roads"], "pois": scene["pois"],
                "background_profile_file": "unused.sqlite3",
            }), encoding="utf-8")
            build(root / "source", root / "output", set(range(4)), 96, 96, 0,
                  {"test": 9303}, observation_tail=12,
                  fixed_requests_per_slot=True)
            for condition in "ABCD":
                path = root / f"output/scenario_{condition}/test/seed_9303/scenario.json"
                generated = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(generated["pois"]), {"A": 20, "B": 40, "C": 60, "D": 100}[condition])
                self.assertEqual(generated["provenance"]["selected_pois"], len(generated["pois"]))
                self.assertEqual(generated["steps"], 108)
                self.assertEqual(len(generated["profile_observations"]), 108)
                self.assertEqual(Counter(row["departure_slot"] for row in generated["requests"]),
                                 {slot: 1 for slot in range(96)})
                self.assertEqual(generated["profile_observations"][96]["slot"], 96)

    def test_eight_hour_window_and_observation_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            rows = [
                {"request_id": f"source-{slot}-{region}-{index}",
                 "region_id": region, "departure_slot": slot,
                 "destination_poi_id": 1, "edge_ids": [1],
                 "candidates": [{"destination_poi_id": 1, "edge_ids": [1]}]}
                for slot in range(28, 36)
                for region in range(4)
                for index in range(2)
            ]
            scene = {
                "requests": rows,
                "roads": [{"edge_id": 1, "capacity_veh_per_slot": 10.0,
                           "speed_limit_kmh": 30.0}],
                "pois": [{"poi_id": poi_id, "category": 7,
                          "service_people_per_slot": 2.0, "capacity_people": 20.0}
                         for poi_id in range(1, 101)],
                "background_profile_file": "unused.sqlite3",
            }
            seeds = {"train": 9301, "validation": 9302, "test": 9303}
            for split, seed in seeds.items():
                path = source / split / "citywide" / f"seed_{seed}" / "scenario.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(scene), encoding="utf-8")

            build(source, output, set(range(4)), 128, 32, 28, seeds,
                  observation_tail=4)

            for condition in "ABCD":
                path = output / "dataset" / "test" / f"scenario_{condition}" / "seed_9303" / "scenario.json"
                self.assertTrue(path.is_file())
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(result["pois"]), {"A": 20, "B": 40, "C": 60, "D": 100}[condition])
                self.assertEqual(result["steps"], 36)
                self.assertEqual(result["request_window_slots"], 32)
                self.assertEqual(len(result["requests"]), 128)
                self.assertEqual(len({row["request_id"] for row in result["requests"]}), 128)
                self.assertEqual(len(result["profile_observations"]), 36)
                self.assertEqual(max(row["departure_slot"] for row in result["requests"]), 59 if condition != "C" else 56)
                self.assertEqual(result["cost_normalization"]["road_reference_vehicle_min"],
                                 result["roads"][0]["capacity_veh_per_slot"] * 36 * 15)

            a = json.loads((output / "scenario_A/test/seed_9303/scenario.json").read_text())
            self.assertEqual(Counter(row["departure_slot"] for row in a["requests"]),
                             {slot: 4 for slot in range(28, 60)})
            self.assertEqual(a["profile_observations"][0]["road_background"],
                             a["profile_observations"][8]["road_background"])

            c = json.loads((output / "scenario_C/test/seed_9303/scenario.json").read_text())
            self.assertEqual({row["departure_slot"] for row in c["requests"]},
                             {28 + 8 * cycle + offset for cycle in range(4)
                              for offset in (2, 3, 4)})

            d = json.loads((output / "scenario_D/test/seed_9303/scenario.json").read_text())
            shocks = [item["slot"] for item in d["profile_observations"]
                      if item["road_background"]["1"] == 9.5]
            self.assertEqual(shocks, [32])
            self.assertEqual(d["profile_observations"][4]["poi_arrivals"]["1"], 3.0)
            self.assertNotEqual(d["profile_observations"][12]["poi_arrivals"]["1"], 3.0)


if __name__ == "__main__":
    unittest.main()
