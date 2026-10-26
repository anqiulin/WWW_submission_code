"""Couple predicted background demand with action-driven road and POI states."""

from __future__ import annotations

import math
import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .model import CountModel, poi_extra, road_extra
from .metrics import GlobalLoadWeights
from .background_profile import DailyBackgroundProfile


STEP_SECONDS = 900


def bpr_travel_time(
    free_time_sec: float,
    flow_vehicles: float,
    capacity_vehicles: float,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    """Return a monotone BPR travel time for one 15-minute link."""
    if free_time_sec < 0 or flow_vehicles < 0 or capacity_vehicles <= 0:
        raise ValueError("BPR inputs must be non-negative with positive capacity")
    vc_ratio = flow_vehicles / capacity_vehicles
    return free_time_sec * (1.0 + alpha * vc_ratio ** beta)


@dataclass
class RoadState:
    edge_id: int
    from_node: int
    to_node: int
    length_m: float
    speed_limit_kmh: float
    capacity_veh_per_slot: float
    count_history: list[float] = field(default_factory=lambda: [0.0])
    speed_kmh: float = 30.0
    background_vehicles: float = 0.0
    recommended_vehicles: float = 0.0
    travel_time_sec: float = 0.0
    queue_vehicles: float = 0.0
    outflow_vehicles: float = 0.0
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0


@dataclass
class POIState:
    poi_id: int
    node_id: int
    category: int
    capacity_people: float
    service_people_per_slot: float
    occupancy: float = 0.0
    waiting: float = 0.0
    mta_period: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    arrival_history: list[float] = field(default_factory=lambda: [0.0])
    departure_history: list[float] = field(default_factory=lambda: [0.0])
    arrivals: float = 0.0
    departures: float = 0.0
    admitted: float = 0.0
    wait_time_min: float = 0.0


@dataclass(frozen=True)
class Request:
    request_id: str
    origin_node_id: int
    destination_poi_id: int
    people: float
    edge_ids: tuple[int, ...]
    vehicles: float = 1.0
    final_destination_node_id: int | None = None
    egress_edge_ids: tuple[int, ...] = ()
    dwell_slots: int = 1
    category: int | None = None


@dataclass
class Traveler:
    request: Request
    status: str = "scheduled"
    route_position: int = 0
    phase: str = "access"
    next_event_slot: int = 0
    access_entry_slots: tuple[int, ...] = ()
    access_arrival_slot: int = 0
    egress_entry_slots: tuple[int, ...] = ()
    egress_finish_slot: int = 0


class CoupledSimulator:
    def __init__(
        self, roads: dict[int, RoadState], pois: dict[int, POIState],
        road_model: CountModel, arrival_model: CountModel, departure_model: CountModel,
        start_slot: int = 0,
        allow_over_capacity: bool = False,
        cost_weights: GlobalLoadWeights | None = None,
        background_profile: DailyBackgroundProfile | None = None,
        dynamics_mode: str = "reserved",
    ) -> None:
        if dynamics_mode not in ("reserved", "event_driven"):
            raise ValueError(f"unknown dynamics mode: {dynamics_mode}")
        self.roads = roads
        self.pois = pois
        self.road_model = road_model
        self.arrival_model = arrival_model
        self.departure_model = departure_model
        self.slot = start_slot
        self.allow_over_capacity = allow_over_capacity
        self.cost_weights = cost_weights or GlobalLoadWeights()
        if background_profile is not None:
            background_profile.validate_ids(roads, pois)
        self.background_profile = background_profile
        self.dynamics_mode = dynamics_mode
        self.travelers: dict[str, Traveler] = {}
        self.events: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
        self.edge_waiting: dict[int, deque[tuple[str | None, float]]] = defaultdict(deque)
        for edge_id, road in roads.items():
            if road.queue_vehicles > 0:
                self.edge_waiting[edge_id].append((None, road.queue_vehicles))
        self.poi_waiting: dict[int, deque[str]] = defaultdict(deque)
        self.poi_admission_queue: dict[int, deque[tuple[str | None, float]]] = defaultdict(deque)
        for poi_id, poi in pois.items():
            if poi.waiting > 0:
                self.poi_admission_queue[poi_id].append((None, poi.waiting))
        self.recommended_poi_waiting: dict[int, float] = defaultdict(float)
        self.provisional_edge_counts: dict[int, float] = defaultdict(float)
        self.background_occupancy = {poi_id: poi.occupancy for poi_id, poi in pois.items()}
        self.reserved: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.pending_arrivals: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.active_requests: set[str] = set()
        self.pending_departures: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def fork(self) -> "CoupledSimulator":
        """Copy the entire mutable state for a paired policy comparison."""
        trial = copy.copy(self)
        trial.roads = {edge_id: copy.copy(road) for edge_id, road in self.roads.items()}
        for road in trial.roads.values():
            road.count_history = road.count_history.copy()
        trial.pois = {poi_id: copy.copy(poi) for poi_id, poi in self.pois.items()}
        for poi in trial.pois.values():
            poi.arrival_history = poi.arrival_history.copy()
            poi.departure_history = poi.departure_history.copy()
        trial.reserved = copy.deepcopy(self.reserved)
        trial.pending_arrivals = copy.deepcopy(self.pending_arrivals)
        trial.pending_departures = copy.deepcopy(self.pending_departures)
        trial.active_requests = self.active_requests.copy()
        trial.travelers = {key: copy.copy(value) for key, value in self.travelers.items()}
        trial.events = defaultdict(list, {slot: events.copy() for slot, events in self.events.items()})
        trial.edge_waiting = defaultdict(deque, {edge: deque(queue) for edge, queue in self.edge_waiting.items()})
        trial.poi_waiting = defaultdict(deque, {poi: deque(queue) for poi, queue in self.poi_waiting.items()})
        trial.poi_admission_queue = defaultdict(
            deque, {poi: deque(queue) for poi, queue in self.poi_admission_queue.items()})
        trial.recommended_poi_waiting = defaultdict(float, self.recommended_poi_waiting)
        trial.provisional_edge_counts = defaultdict(float, self.provisional_edge_counts)
        trial.background_occupancy = self.background_occupancy.copy()
        for name in ("policy_forecast_by_poi", "policy_batch_counts",
                     "policy_in_transit_by_poi", "policy_arriving_soon_by_poi"):
            if hasattr(self, name):
                setattr(trial, name, copy.copy(getattr(self, name)))
        return trial

    def _background(self, road: RoadState, slot: int) -> float:
        if self.background_profile is not None:
            return self.background_profile.road_flow(road, slot)
        extra = road_extra(road.length_m, road.speed_limit_kmh, road.speed_kmh)
        return self.road_model.predict(road.count_history[-4:], slot, extra)

    @staticmethod
    def _travel_time(road: RoadState, vehicles_per_slot: float) -> float:
        limit = road.speed_limit_kmh if road.speed_limit_kmh > 0 else road.speed_kmh
        free_speed = max(5.0, limit)
        free_time = max(1.0, road.length_m * 3.6 / free_speed)
        return bpr_travel_time(
            free_time, max(0.0, vehicles_per_slot),
            max(1.0, road.capacity_veh_per_slot), road.bpr_alpha, road.bpr_beta,
        )

    def _schedule(self, request: Request, current_background: dict[int, float]) -> int:
        if request.request_id in self.active_requests:
            raise ValueError(f"Duplicate request ID: {request.request_id}")
        if request.people <= 0 or request.vehicles <= 0 or not request.edge_ids:
            raise ValueError("A request needs people, vehicles, and a non-empty route")
        if request.dwell_slots < 0:
            raise ValueError("dwell_slots must be non-negative")
        poi = self.pois[request.destination_poi_id]
        if request.category is not None and request.category != poi.category:
            raise ValueError(f"POI {poi.poi_id} does not match request category")
        previous_node = request.origin_node_id
        for edge_id in request.edge_ids:
            road = self.roads[edge_id]
            if road.from_node != previous_node:
                raise ValueError(f"Disconnected route at edge {edge_id}")
            previous_node = road.to_node
        if previous_node != poi.node_id:
            raise ValueError("Route does not end at the destination POI node")
        if request.egress_edge_ids:
            if request.final_destination_node_id is None:
                raise ValueError("egress route requires final_destination_node_id")
            previous_node = poi.node_id
            for edge_id in request.egress_edge_ids:
                road = self.roads[edge_id]
                if road.from_node != previous_node:
                    raise ValueError(f"Disconnected egress route at edge {edge_id}")
                previous_node = road.to_node
            if previous_node != request.final_destination_node_id:
                raise ValueError("Egress route does not end at final destination")
        elif request.final_destination_node_id is not None:
            raise ValueError("final destination requires egress_edge_ids")

        if self.dynamics_mode == "event_driven":
            access_entry_slots, access_arrival_slot = self._compressed_route_schedule(
                request.edge_ids, self.slot, minimum_slots=2)
            self.active_requests.add(request.request_id)
            self.travelers[request.request_id] = Traveler(
                request, next_event_slot=self.slot,
                access_entry_slots=access_entry_slots,
                access_arrival_slot=access_arrival_slot,
            )
            for position, event_slot in enumerate(access_entry_slots):
                self.events[event_slot].append((request.request_id, "edge_entry", position))
            self.provisional_edge_counts[request.edge_ids[0]] += request.vehicles
            # The reported arrival slot is the POI arrival, before dwell and
            # any optional egress leg.
            return access_arrival_slot

        # Prepare the whole route first, then commit reservations atomically.
        planned: list[tuple[int, int]] = []
        timestamp = self.slot * STEP_SECONDS
        tentative: dict[tuple[int, int], float] = defaultdict(float)
        for edge_id in request.edge_ids:
            road = self.roads[edge_id]
            if request.vehicles > road.capacity_veh_per_slot:
                raise ValueError(f"Request exceeds edge {edge_id} per-slot capacity")
            candidate_slot = int(timestamp // STEP_SECONDS)
            while True:
                if candidate_slot > self.slot + 96:
                    raise ValueError("No road capacity within the next 24 hours")
                background = (current_background[edge_id] if candidate_slot == self.slot
                              else self._background(road, candidate_slot))
                already = self.reserved[candidate_slot].get(edge_id, 0.0) + tentative[(candidate_slot, edge_id)]
                carried_queue = road.queue_vehicles if candidate_slot == self.slot else 0.0
                if (self.allow_over_capacity or
                        carried_queue + background + already + request.vehicles <= road.capacity_veh_per_slot + 1e-9):
                    break
                candidate_slot += 1
            timestamp = max(timestamp, candidate_slot * STEP_SECONDS)
            tentative[(candidate_slot, edge_id)] += request.vehicles
            planned.append((candidate_slot, edge_id))
            queue_ahead = carried_queue + background + already
            if self.allow_over_capacity:
                timestamp += int(queue_ahead // road.capacity_veh_per_slot) * STEP_SECONDS
            timestamp += self._travel_time(
                road, min(road.capacity_veh_per_slot, queue_ahead + request.vehicles))
        for candidate_slot, edge_id in planned:
            self.reserved[candidate_slot][edge_id] += request.vehicles
        arrival_slot = int(timestamp // STEP_SECONDS)
        self.pending_arrivals[arrival_slot][request.destination_poi_id] += request.people
        destination_arrival_slot = None
        if request.egress_edge_ids:
            # Reserve the departure leg after the POI dwell period. The POI
            # departure event is consumed by the conservation update below.
            egress_timestamp = (arrival_slot + request.dwell_slots) * STEP_SECONDS
            egress_planned: list[tuple[int, int]] = []
            tentative_egress: dict[int, float] = defaultdict(float)
            for edge_id in request.egress_edge_ids:
                road = self.roads[edge_id]
                candidate_slot = int(egress_timestamp // STEP_SECONDS)
                while True:
                    if candidate_slot > self.slot + 96:
                        raise ValueError("No egress capacity within the next 24 hours")
                    background = (current_background[edge_id] if candidate_slot == self.slot
                                  else self._background(road, candidate_slot))
                    already = self.reserved[candidate_slot].get(edge_id, 0.0) + tentative_egress[edge_id]
                    carried_queue = road.queue_vehicles if candidate_slot == self.slot else 0.0
                    if carried_queue + background + already + request.vehicles <= road.capacity_veh_per_slot + 1e-9:
                        break
                    candidate_slot += 1
                egress_planned.append((candidate_slot, edge_id))
                tentative_egress[edge_id] += request.vehicles
                egress_timestamp = max(egress_timestamp, candidate_slot * STEP_SECONDS)
                egress_timestamp += self._travel_time(
                    road, min(road.capacity_veh_per_slot, background + already + request.vehicles))
            for candidate_slot, edge_id in egress_planned:
                self.reserved[candidate_slot][edge_id] += request.vehicles
            departure_slot = egress_planned[0][0]
            self.pending_departures[departure_slot][request.destination_poi_id] += request.people
            destination_arrival_slot = int(egress_timestamp // STEP_SECONDS)
        self.active_requests.add(request.request_id)
        return arrival_slot if destination_arrival_slot is None else destination_arrival_slot

    def _compressed_route_schedule(self, edge_ids: tuple[int, ...], start_slot: int,
                                   minimum_slots: int) -> tuple[tuple[int, ...], int]:
        """Schedule an ordered route within a bounded number of time slots."""
        if not edge_ids:
            return (), start_slot
        travel = sum(self._travel_time(self.roads[edge_id], 0.0) for edge_id in edge_ids)
        complexity_delay = (3 if minimum_slots >= 2 and len(edge_ids) <= 2
                            else math.ceil(len(edge_ids) / 4))
        target_delay = max(math.ceil(travel / STEP_SECONDS), complexity_delay, minimum_slots)
        target_delay = max(minimum_slots, min(4, target_delay))
        arrival_slot = start_slot + target_delay
        if len(edge_ids) == 1:
            entries = (start_slot,)
        else:
            entries = tuple(start_slot + (index * (target_delay - 1)) // (len(edge_ids) - 1)
                            for index in range(len(edge_ids)))
        return entries, arrival_slot

    def step(
        self, requests: list[Request] | None = None,
        observed_road_background: dict[int, float] | None = None,
        observed_poi_arrivals: dict[int, float] | None = None,
        observed_poi_departures: dict[int, float] | None = None,
    ) -> dict:
        """Advance one slot. Observations override proxies when they are available."""
        if self.dynamics_mode == "event_driven":
            return self._step_event_driven(requests or [], observed_road_background,
                                           observed_poi_arrivals, observed_poi_departures)
        requests = requests or []
        for observed in (observed_road_background, observed_poi_arrivals, observed_poi_departures):
            if observed and any(value < 0 for value in observed.values()):
                raise ValueError("Observed counts must be non-negative")
        background = {
            edge_id: (observed_road_background[edge_id] if observed_road_background and edge_id in observed_road_background
                      else self._background(road, self.slot))
            for edge_id, road in self.roads.items()
        }
        arrival_slots = [self._schedule(request, background) for request in requests]
        assigned = self.reserved.pop(self.slot, {})
        road_output = {}
        for edge_id, road in self.roads.items():
            road.background_vehicles = background[edge_id]
            road.recommended_vehicles = assigned.get(edge_id, 0.0)
            entering = road.background_vehicles + road.recommended_vehicles
            total_load = road.queue_vehicles + entering
            road.outflow_vehicles = min(total_load, max(0.0, road.capacity_veh_per_slot))
            road.queue_vehicles = max(0.0, total_load - road.outflow_vehicles)
            # Vehicles left in the queue incur one slot of waiting. BPR models
            # the vehicles actually served, so excess demand is not charged a
            # quartic travel time and a queue penalty at the same time.
            road.travel_time_sec = self._travel_time(road, road.outflow_vehicles)
            road.speed_kmh = max(0.1, road.length_m * 3.6 / road.travel_time_sec)
            road.count_history.append(road.background_vehicles)
            road.count_history = road.count_history[-4:]
            road_output[edge_id] = {
                "background_vehicles": road.background_vehicles,
                "recommended_vehicles": road.recommended_vehicles,
                "total_inflow_vehicles": entering,
                "total_load_vehicles": total_load,
                "outflow_vehicles": road.outflow_vehicles,
                "queue_vehicles": road.queue_vehicles,
                "capacity_vehicles": road.capacity_veh_per_slot,
                "vc_ratio": total_load / max(1.0, road.capacity_veh_per_slot),
                "speed_kmh": road.speed_kmh,
                "travel_time_sec": road.travel_time_sec,
            }
        due = self.pending_arrivals.pop(self.slot, {})
        scheduled_departures = self.pending_departures.pop(self.slot, {})
        poi_output = {}
        for poi_id, poi in self.pois.items():
            signal = poi.mta_period[(self.slot % 96) // 24]
            extra = poi_extra(poi.occupancy, signal)
            base = (observed_poi_arrivals[poi_id] if observed_poi_arrivals and poi_id in observed_poi_arrivals
                    else self.background_profile.poi_arrivals(poi, self.slot) if self.background_profile is not None
                    else self.arrival_model.predict(poi.arrival_history, self.slot, extra, poi.category))
            scheduled = scheduled_departures.get(poi_id, 0.0)
            background_departures = (observed_poi_departures[poi_id]
                                     if observed_poi_departures and poi_id in observed_poi_departures
                                     else self.background_profile.poi_departures(
                                         poi, max(0.0, poi.occupancy - scheduled))
                                     if self.background_profile is not None
                                     else self.departure_model.predict(
                                         poi.departure_history, self.slot, extra, poi.category))
            expected_departures = background_departures + scheduled
            departed = min(poi.occupancy, expected_departures)
            total_arrivals = base + due.get(poi_id, 0.0)
            waiting = poi.waiting + total_arrivals
            available_space = max(0.0, poi.capacity_people - (poi.occupancy - departed))
            admitted = min(waiting, available_space, poi.service_people_per_slot)
            poi.occupancy = max(0.0, poi.occupancy - departed + admitted)
            poi.waiting = max(0.0, waiting - admitted)
            poi.arrivals = total_arrivals
            poi.departures = departed
            poi.admitted = admitted
            poi.wait_time_min = 15.0 * poi.waiting / max(1.0, poi.service_people_per_slot)
            poi.arrival_history.append(base)
            poi.departure_history.append(departed)
            poi.arrival_history = poi.arrival_history[-4:]
            poi.departure_history = poi.departure_history[-4:]
            poi_output[poi_id] = {
                "base_arrivals": base, "recommendation_arrivals": due.get(poi_id, 0.0),
                "scheduled_recommendation_departures": scheduled,
                "arrivals": total_arrivals, "departures": departed,
                "admitted": admitted, "occupancy": poi.occupancy,
                "waiting": poi.waiting, "wait_time_min": poi.wait_time_min,
            }
        result = {"slot": self.slot, "request_arrival_slots": arrival_slots,
                  "roads": road_output, "pois": poi_output}
        self.slot += 1
        return result

    def _step_event_driven(self, requests, observed_road_background,
                           observed_poi_arrivals, observed_poi_departures) -> dict:
        for observed in (observed_road_background, observed_poi_arrivals, observed_poi_departures):
            if observed and any(value < 0 for value in observed.values()):
                raise ValueError("Observed counts must be non-negative")
        background = {
            edge_id: (observed_road_background[edge_id]
                      if observed_road_background and edge_id in observed_road_background
                      else self._background(road, self.slot))
            for edge_id, road in self.roads.items()
        }
        arrival_slots = [self._schedule(request, background) for request in requests]
        for edge_id, base in background.items():
            if base:
                queue = self.edge_waiting[edge_id]
                if queue and queue[-1][0] is None:
                    _, previous = queue.pop()
                    queue.append((None, previous + base))
                else:
                    queue.append((None, base))
        due_arrivals: dict[int, float] = defaultdict(float)
        due_departures: dict[int, float] = defaultdict(float)
        entering_road: dict[int, float] = defaultdict(float)
        current_events = self.events.pop(self.slot, [])
        event_index = 0
        while event_index < len(current_events):
            traveler_id, kind, position = current_events[event_index]
            event_index += 1
            traveler = self.travelers[traveler_id]
            request = traveler.request
            if kind == "edge_entry":
                route = request.edge_ids if traveler.phase == "access" else request.egress_edge_ids
                self.edge_waiting[route[position]].append((traveler_id, request.vehicles))
                entering_road[route[position]] += request.vehicles
                traveler.status = "road_queue"
                traveler.route_position = position
            elif kind == "poi_arrival":
                self.poi_waiting[request.destination_poi_id].append(traveler_id)
                self.poi_admission_queue[request.destination_poi_id].append(
                    (traveler_id, request.people))
                self.recommended_poi_waiting[request.destination_poi_id] += request.people
                due_arrivals[request.destination_poi_id] += request.people
                traveler.status = "poi_queue"
            elif kind == "poi_departure":
                due_departures[request.destination_poi_id] += request.people
                traveler.phase = "egress"
                if request.egress_edge_ids:
                    traveler.egress_entry_slots, traveler.egress_finish_slot = self._compressed_route_schedule(
                        request.egress_edge_ids, self.slot, minimum_slots=1)
                    for position, event_slot in enumerate(traveler.egress_entry_slots):
                        event = (traveler_id, "edge_entry", position)
                        if event_slot == self.slot:
                            current_events.append(event)
                        else:
                            self.events[event_slot].append(event)
                    # The first egress event is consumed by the event loop
                    # above when it is scheduled for the current slot.
                    traveler.status = "road_queue"
                    traveler.route_position = 0
                else:
                    traveler.status = "complete"
            elif kind == "finish":
                traveler.status = "complete"
        self.provisional_edge_counts.clear()
        road_output = {}
        road_queue_vehicle_min = 0.0
        road_travel_delay_vehicle_min = 0.0
        peak_vc = 0.0
        for edge_id, road in self.roads.items():
            queue = self.edge_waiting[edge_id]
            base = background[edge_id]
            incoming_rec = entering_road[edge_id]
            total_load = road.queue_vehicles + base + incoming_rec
            available = max(0.0, road.capacity_veh_per_slot)
            served = 0.0
            served_recommended = 0.0
            travel_time = self._travel_time(road, min(total_load, available))
            while queue and available > 1e-9:
                traveler_id, amount = queue[0]
                if traveler_id is not None and amount > available + 1e-9:
                    break
                moved = min(amount, available)
                available -= moved
                served += moved
                if moved + 1e-9 < amount:
                    queue[0] = (traveler_id, amount - moved)
                    break
                queue.popleft()
                if traveler_id is not None:
                    served_recommended += moved
                    traveler = self.travelers[traveler_id]
                    traveler.status = "in_transit"
                    route = traveler.request.edge_ids if traveler.phase == "access" else traveler.request.egress_edge_ids
                    if traveler.route_position + 1 >= len(route):
                        next_slot = max(self.slot + 1, (
                            traveler.access_arrival_slot if traveler.phase == "access"
                            else traveler.egress_finish_slot))
                        next_kind = "poi_arrival" if traveler.phase == "access" else "finish"
                        traveler.next_event_slot = next_slot
                        self.events[next_slot].append((traveler_id, next_kind,
                                                        traveler.route_position + 1))
            road.background_vehicles = base
            road.recommended_vehicles = served_recommended
            road.queue_vehicles = max(0.0, total_load - served)
            road.outflow_vehicles = served
            road.travel_time_sec = travel_time
            road.speed_kmh = max(0.1, road.length_m * 3.6 / travel_time)
            road.count_history = (road.count_history + [base])[-4:]
            road_queue_vehicle_min += road.queue_vehicles * STEP_SECONDS / 60.0
            road_travel_delay_vehicle_min += served * max(
                0.0, travel_time - self._travel_time(road, 0.0)) / 60.0
            peak_vc = max(peak_vc, total_load / max(1.0, road.capacity_veh_per_slot))
            road_output[edge_id] = {
                "background_vehicles": base, "recommended_vehicles": served_recommended,
                "total_inflow_vehicles": base + incoming_rec,
                "total_load_vehicles": total_load, "outflow_vehicles": served,
                "queue_vehicles": road.queue_vehicles,
                "capacity_vehicles": road.capacity_veh_per_slot,
                "vc_ratio": total_load / max(1.0, road.capacity_veh_per_slot),
                "speed_kmh": road.speed_kmh, "travel_time_sec": travel_time,
            }
        poi_output = {}
        poi_wait_person_min = 0.0
        peak_poi_utilization = 0.0
        for poi_id, poi in self.pois.items():
            signal = poi.mta_period[(self.slot % 96) // 24]
            extra = poi_extra(poi.occupancy, signal)
            base = (observed_poi_arrivals[poi_id]
                    if observed_poi_arrivals and poi_id in observed_poi_arrivals
                    else self.background_profile.poi_arrivals(poi, self.slot)
                    if self.background_profile is not None
                    else self.arrival_model.predict(poi.arrival_history, self.slot, extra, poi.category))
            background_departures = (observed_poi_departures[poi_id]
                                     if observed_poi_departures and poi_id in observed_poi_departures
                                     else self.background_profile.poi_departures(poi, self.background_occupancy[poi_id])
                                     if self.background_profile is not None
                                     else self.departure_model.predict(poi.departure_history, self.slot, extra, poi.category))
            background_departures = min(self.background_occupancy[poi_id], background_departures)
            self.background_occupancy[poi_id] -= background_departures
            queue = self.poi_admission_queue[poi_id]
            if base:
                if queue and queue[-1][0] is None:
                    _, previous = queue.pop()
                    queue.append((None, previous + base))
                else:
                    queue.append((None, base))
            available = min(poi.service_people_per_slot,
                            max(0.0, poi.capacity_people - poi.occupancy +
                                background_departures + due_departures[poi_id]))
            admitted_background = 0.0
            admitted_rec = 0.0
            while queue and available > 1e-9:
                traveler_id, amount = queue[0]
                if traveler_id is not None and amount > available + 1e-9:
                    break
                moved = min(amount, available)
                available -= moved
                if moved + 1e-9 < amount:
                    queue[0] = (traveler_id, amount - moved)
                    admitted_background += moved
                    break
                queue.popleft()
                if traveler_id is None:
                    admitted_background += moved
                    continue
                self.poi_waiting[poi_id].popleft()
                traveler = self.travelers[traveler_id]
                admitted_rec += amount
                self.recommended_poi_waiting[poi_id] -= amount
                traveler.status = "visiting"
                next_slot = self.slot + max(1, traveler.request.dwell_slots)
                traveler.next_event_slot = next_slot
                self.events[next_slot].append((traveler_id, "poi_departure", 0))
            self.background_occupancy[poi_id] += admitted_background
            poi.occupancy = max(0.0, poi.occupancy - background_departures -
                                due_departures[poi_id] + admitted_background + admitted_rec)
            poi.waiting = max(0.0, poi.waiting + base + due_arrivals[poi_id] -
                              admitted_background - admitted_rec)
            poi.arrivals = base + due_arrivals[poi_id]
            poi.departures = background_departures + due_departures[poi_id]
            poi.admitted = admitted_background + admitted_rec
            poi.wait_time_min = 15.0 * poi.waiting / max(1.0, poi.service_people_per_slot)
            poi_wait_person_min += poi.waiting * STEP_SECONDS / 60.0
            peak_poi_utilization = max(peak_poi_utilization,
                                       poi.occupancy / max(1.0, poi.capacity_people))
            poi.arrival_history = (poi.arrival_history + [base])[-4:]
            poi.departure_history = (poi.departure_history + [background_departures])[-4:]
            poi_output[poi_id] = {
                "base_arrivals": base, "recommendation_arrivals": due_arrivals[poi_id],
                "scheduled_recommendation_departures": due_departures[poi_id],
                "arrivals": poi.arrivals, "departures": poi.departures,
                "admitted": poi.admitted, "occupancy": poi.occupancy,
                "waiting": poi.waiting, "wait_time_min": poi.wait_time_min,
            }
        result = {"slot": self.slot, "request_arrival_slots": arrival_slots,
                  "roads": road_output, "pois": poi_output,
                  "aggregate": {"road_queue_vehicle_min": road_queue_vehicle_min,
                                "road_travel_delay_vehicle_min": road_travel_delay_vehicle_min,
                                "poi_wait_person_min": poi_wait_person_min,
                                "peak_vc": peak_vc,
                                "peak_poi_utilization": peak_poi_utilization}}
        self.slot += 1
        return result
