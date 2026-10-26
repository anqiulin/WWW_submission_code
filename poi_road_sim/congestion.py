"""Capacity exceedance statistics for one simulator time step and episode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def congestion_snapshot(simulator, result: Mapping) -> dict[str, int]:
    """Count objects whose pre-service demand strictly exceeds capacity.

    POI occupancy is capped by the simulator; waiting people are included in
    demand so an overflow remains observable after admission is limited.
    """
    roads = sum(float(row["total_load_vehicles"]) > float(row["capacity_vehicles"]) + 1e-9
                for row in result["roads"].values())
    pois = sum(float(row["occupancy"]) + float(row["waiting"]) >
               simulator.pois[int(poi_id)].capacity_people + 1e-9
               for poi_id, row in result["pois"].items())
    return {"congested_road_edges": roads, "congested_pois": pois}


def congestion_episode(rows: Sequence[Mapping], slot_minutes: int = 15) -> dict:
    """Summarize first onset, total duration, and longest continuous episode."""
    summary = {}
    for object_name, count_key in (("road", "congested_road_edges"),
                                   ("poi", "congested_pois")):
        congested = [int(row["slot"]) for row in rows if int(row[count_key]) > 0]
        longest = run = 0
        previous = None
        for slot in congested:
            run = run + 1 if previous is not None and slot == previous + 1 else 1
            longest = max(longest, run)
            previous = slot
        summary[f"first_{object_name}_congestion_slot"] = congested[0] if congested else None
        summary[f"first_{object_name}_congestion_time"] = (
            f"{(congested[0] * slot_minutes) // 60:02d}:"
            f"{(congested[0] * slot_minutes) % 60:02d}" if congested else None)
        summary[f"{object_name}_congestion_duration_min"] = len(congested) * slot_minutes
        summary[f"{object_name}_longest_congestion_min"] = longest * slot_minutes
        summary[f"{object_name}_congested_object_slots"] = sum(int(row[count_key]) for row in rows)
    return summary
