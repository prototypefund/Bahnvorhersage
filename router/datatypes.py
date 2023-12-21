from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

from gtfs.routes import Routes
from gtfs.stop_times import StopTimes

from router.constants import EXTRA_TIME, MINIMAL_DISTANCE_DIFFERENCE


@dataclass(frozen=True, eq=True)
class MockStopTimes:
    departure_time: datetime


@dataclass(eq=True)
class Leg:
    trip_id: int
    departure_ts: datetime
    arrival_ts: datetime
    duration: timedelta = field(init=False)
    is_regional: bool
    dist_traveled: float

    def __post_init__(self):
        self.duration = self.arrival_ts - self.departure_ts

    def to_string(self, db: "GTFSInterface") -> str:
        trip: Trip = db.get_trip(self.trip_id)
        if self.is_regional:
            return f'🭃{self.duration.total_seconds() // 60:n} {trip.route.route_short_name}🭎'
        else:
            return f'🭊🭂{self.duration.total_seconds() // 60:n} {trip.route.route_short_name}🭍🬿'


@dataclass
class Trip:
    trip_id: int
    route: Routes
    stop_sequence: List[StopTimes]

    def stop_sequence_from(
        self, stop_id: int, include_start: bool = False
    ) -> List[StopTimes]:
        for i, stop_time in enumerate(self.stop_sequence):
            if stop_time.stop_id == stop_id:
                if include_start:
                    return self.stop_sequence[i:]
                else:
                    return self.stop_sequence[i + 1 :]
        raise KeyError(f"Stop {stop_id} not in trip {self.trip_id}")

    def stop_time_at(self, stop_id: int) -> StopTimes:
        for stop_time in self.stop_sequence:
            if stop_time.stop_id == stop_id:
                return stop_time
        raise KeyError(f"Stop {stop_id} not in trip {self.trip_id}")
    
    def dist_traveled(self, from_stop_id: int, to_stop_id: int) -> float:
        from_distance = None
        to_distance = None
        for stop_time in self.stop_sequence:
            if stop_time.stop_id == from_stop_id:
                from_distance = stop_time.shape_dist_traveled
            if stop_time.stop_id == to_stop_id:
                to_distance = stop_time.shape_dist_traveled
                break
        if from_distance is None or to_distance is None:
            raise KeyError(
                f"Stop {from_stop_id} or {to_stop_id} not in trip {self.trip_id}"
            )
        return to_distance - from_distance


@dataclass
class JourneyIdentifier:
    tree_node_id: int
    departure_time: datetime
    arrival_time: datetime
    duration: timedelta = field(init=False)
    transfers: int
    is_regional: bool
    dist_traveled: float

    def __post_init__(self):
        self.duration = self.arrival_time - self.departure_time

    def fully_overlapped(self, other: "JourneyIdentifier"):
        x1 = (self.departure_time,)
        x2 = (self.arrival_time,)
        y1 = (other.departure_time,)
        y2 = (other.arrival_time,)
        return (x1 <= y1 and x2 >= y2) or (y1 <= x1 and y2 >= x2)

    def is_pareto_dominated_by(self, other: "JourneyIdentifier") -> bool:
        # TODO: Find criteria for pareto dominance
        #   - which criteria must be better?
        #   - which criteria might be equal or worse?
        #   - which criteria might be added?
        #   - should connections with same n of transfers have same duration? Or is only the faster usefull?
        if self.is_regional > other.is_regional:
            return False
        if not self.fully_overlapped(other):
            return False
        else:
            # A journey is dominated, if it is not better in any aspect,
            # but worse in at least one aspect
            worse_count = sum(
                [
                    self.duration > EXTRA_TIME(other.duration),
                    self.transfers > other.transfers,
                    self.dist_traveled > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
                ]
            )
            if worse_count:
                equal_count = sum(
                    [
                        EXTRA_TIME(other.duration) >= self.duration >= other.duration,
                        self.transfers == other.transfers,
                        abs(self.dist_traveled - other.dist_traveled) < MINIMAL_DISTANCE_DIFFERENCE,
                    ]
                )
                if equal_count + worse_count == 3:
                    return True

            return False
            # # self is worse in all aspects -> self is dominated by other
            # return (
            #     # self.arrival_time >= other.arrival_time and
            #     self.duration >= EXTRA_TIME(other.duration)
            #     and self.transfers >= other.transfers
            #     # and self.is_regional >= other.is_regional
            # )

    def __str__(self) -> str:
        return f'dp: {self.departure_time.strftime("%H:%M")}, ar: {self.arrival_time.strftime("%H:%M")}, t: {self.transfers}, d: {self.duration.total_seconds() // 60:n}, l: {self.dist_traveled // 1000:n}km'


@dataclass
class Stop:
    stop_id: int
    name: str
    heuristic: timedelta
    pareto_connections: List[JourneyIdentifier] = field(init=False)

    def __post_init__(self):
        self.pareto_connections = []
