from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from gtfs.stop_times import StopTimes
from gtfs.calendar_dates import CalendarDates
from gtfs.trips import Trips
from gtfs.routes import Routes
from database.engine import sessionfactory
from helpers.StationPhillip import StationPhillip
import queue
from sortedcontainers import SortedKeyList
import sqlalchemy
from itertools import pairwise

TRAIN_SPEED_MS = 70  # ~ 250 km/h
MAX_TIME_DRIVING_AWAY = timedelta(
    seconds=30_000 / TRAIN_SPEED_MS
)  # driving away for 30 km
MINIMAL_TRANSFER_TIME = timedelta(seconds=0)
MINIMAL_TIME_FOR_NEXT_SEARCH = timedelta(seconds=60)
N_MINIMAL_ROUTES_TO_DESTINATION = 2


"""
General algorithm description:
- Add start to connection tree
- Add new node to queue
- While queue not empty:
    - Get next trips from node
    - For each trip:
    - Add node after trip to queue
        - For each stop in trip:
            - If a trip to stop makes sense:
                - Add stop to connection tree (child of node)
                - Add new node to queue
                - If stop is destination:
                    - Do some special magic
"""
# TODO: minimum heuristic including stopover stops


def compute_heuristic(
    stations: StationPhillip, stop_id: int, destination_id: int
) -> timedelta:
    distance_m = stations.geographic_distance_by_eva(
        eva_1=stop_id, eva_2=destination_id
    )
    return timedelta(seconds=distance_m / TRAIN_SPEED_MS)


@dataclass(frozen=True, eq=True)
class MockStopTimes:
    departure_time: datetime


@dataclass(eq=True)
class Leg:
    trip_id: int
    departure_ts: datetime
    arrival_ts: datetime
    duration: timedelta = field(init=False)

    def __post_init__(self):
        self.duration = self.arrival_ts - self.departure_ts


@dataclass
class Trip:
    trip_id: int
    route: Routes
    stop_sequence: List[StopTimes]

    def stop_sequence_from(self, stop_id: int, include_start: bool = False) -> List[StopTimes]:
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


@dataclass(order=True)
class QueueItem:
    node_id: int = field(compare=False)
    heuristic: timedelta = field(compare=False)
    timestamp: datetime = field(compare=False)
    priority: datetime = field(init=False)

    def __post_init__(self):
        self.priority = self.timestamp + self.heuristic


def interval_totally_contained(x1, x2, y1, y2) -> bool:
    return (x1 <= y1 and x2 >= y2) or (y1 <= x1 and y2 >= x2)


@dataclass
class JourneyIdentifier:
    tree_node_id: int
    departure_time: datetime
    arrival_time: datetime
    duration: timedelta = field(init=False)
    transfers: int
    is_regional: bool

    def __post_init__(self):
        self.duration = self.arrival_time - self.departure_time

    def is_pareto_dominated_by(self, other: "JourneyIdentifier") -> bool:
        if not interval_totally_contained(
            self.departure_time,
            self.arrival_time,
            other.departure_time,
            other.arrival_time,
        ):
            return False
        else:
            # self is worse in all aspects -> self is dominated by other
            return (
                self.arrival_time >= other.arrival_time
                and self.duration >= other.duration
                and self.transfers >= other.transfers
                and self.is_regional >= other.is_regional
            )

    def __str__(self) -> str:
        return f'dp: {self.departure_time.strftime("%H:%M")}, ar: {self.arrival_time.strftime("%H:%M")}, t: {self.transfers}, d: {self.duration.total_seconds() // 60:n}'


@dataclass
class Stop:
    stop_id: int
    name: str
    heuristic: timedelta
    pareto_connections: List[JourneyIdentifier] = field(init=False)

    def __post_init__(self):
        self.pareto_connections = []


def find_pareto_optimal_set(
    journeys: List[JourneyIdentifier],
) -> List[JourneyIdentifier]:
    is_dominated = [False] * len(journeys)
    for i, journey in enumerate(journeys):
        for j, other in enumerate(journeys):
            if i != j and not is_dominated[i]:
                is_dominated[i] = journey.is_pareto_dominated_by(other)
    return [
        journey for journey, dominated in zip(journeys, is_dominated) if not dominated
    ]


class GTFSInterface:
    def __init__(self, destination_id: int, stations: StationPhillip) -> None:
        self.destination_id = destination_id
        self.stations = stations

        self.trips_by_stop: Dict[int, SortedKeyList] = {}
        self.trips: Dict[int, Trip] = {}
        self.stops: Dict[int, Stop] = {}
        self.session = sessionfactory()[1]()

    def _next_stop_times_from_cache(
        self, stop_id: int, timestamp: datetime
    ) -> List[StopTimes]:
        if stop_id in self.trips_by_stop:
            index = self.trips_by_stop[stop_id].bisect_left(MockStopTimes(timestamp))
            min_ts = self.trips_by_stop[stop_id][index].departure_time
            if min_ts < timestamp:
                index += 1
                min_ts = self.trips_by_stop[stop_id][index].departure_time
                print('min_ts < timestamp')

            stop_times = []
            for stop_time in self.trips_by_stop[stop_id][index:]:
                if stop_time.departure_time == min_ts:
                    stop_times.append(stop_time)
                else:
                    break
            return stop_times
        raise KeyError(f"No next stop time for {stop_id}, {timestamp} in cache")

    def _next_stop_times_from_db(
        self, stop_id: int, timestamp: datetime
    ) -> List[StopTimes]:
        stmt = (
            sqlalchemy.select(StopTimes)
            .join(Trips, StopTimes.trip_id == Trips.trip_id)
            .join(CalendarDates, Trips.service_id == CalendarDates.service_id)
            .where(
                StopTimes.stop_id == stop_id,
                StopTimes.departure_time != None,
                StopTimes.departure_time >= timestamp,
                CalendarDates.date == timestamp.date(),
            )
            .order_by(StopTimes.departure_time)
        )

        stop_times = self.session.scalars(stmt).all()
        self.trips_by_stop[stop_id] = SortedKeyList(
            stop_times, key=lambda x: x.departure_time
        )

        return self._next_stop_times_from_cache(stop_id, timestamp)

    def next_trips(self, stop_id: int, timestamp: datetime) -> List[int]:
        try:
            stop_times = self._next_stop_times_from_cache(stop_id, timestamp)
        except KeyError:
            stop_times = self._next_stop_times_from_db(stop_id, timestamp)

        return [stop_time.trip_id for stop_time in stop_times]

    def _get_trip_from_db(self, trip_id: int) -> Trip:
        stmt = (
            sqlalchemy.select(StopTimes)
            .where(
                StopTimes.trip_id == trip_id,
            )
            .order_by(StopTimes.stop_sequence)
        )
        trip = self.session.scalars(stmt).all()

        route_stmt = sqlalchemy.select(Routes).where(
            Trips.route_id == Routes.route_id,
            Trips.trip_id == trip_id,
        )
        route = self.session.scalars(route_stmt).one()

        self.trips[trip_id] = Trip(trip_id=trip_id, route=route, stop_sequence=trip)

        return self.trips[trip_id]

    def get_trip(self, trip_id: int) -> Trip:
        if trip_id in self.trips:
            return self.trips[trip_id]
        else:
            return self._get_trip_from_db(trip_id)

    def get_stop_sequence(
        self, trip_id: int, start_stop_id: int, include_start: bool = False
    ) -> List[StopTimes]:
        if trip_id in self.trips:
            return self.trips[trip_id].stop_sequence_from(
                start_stop_id, include_start=include_start
            )
        else:
            return self._get_trip_from_db(trip_id).stop_sequence_from(
                start_stop_id, include_start=include_start
            )

    def get_stop_time(self, trip_id: int, stop_id: int) -> StopTimes:
        if trip_id in self.trips:
            return self.trips[trip_id].stop_time_at(stop_id)
        else:
            return self._get_trip_from_db(trip_id).stop_time_at(stop_id)

    def init_stop(self, stop_id: int) -> Stop:
        stop = Stop(
            stop_id=stop_id,
            name=self.stations.get_name(stop_id),
            heuristic=compute_heuristic(
                stations=self.stations,
                stop_id=stop_id,
                destination_id=self.destination_id,
            ),
        )
        self.stops[stop_id] = stop
        return stop

    def get_stop(self, stop_id: int) -> Stop:
        if stop_id in self.stops:
            return self.stops[stop_id]
        else:
            return self.init_stop(stop_id)

    def add_pareto_connection(self, stop_id: int, journey: JourneyIdentifier) -> None:
        stop = self.get_stop(stop_id)
        stop.pareto_connections.append(journey)
        self.stops[stop_id] = stop


class ConnectionTree:
    def __init__(self, root_stop: Stop) -> None:
        self.downwards: Dict[int, List[int]] = {}
        self.upwards: Dict[int, int] = {}
        self.node_id_to_stop_id: Dict[int, int] = {}
        self.legs: Dict[Tuple[int, int], Leg] = {}
        self.size = 1
        self.new_node_id = 0
        self.root_stop = root_stop

        self.downwards[self.new_node_id] = []
        self.node_id_to_stop_id[self.new_node_id] = root_stop.stop_id
        self.new_node_id += 1

    def __len__(self):
        return self.size

    def get_stop_id(self, node_id: int) -> int:
        return self.node_id_to_stop_id[node_id]

    def add_leg(self, from_node_id: int, to_stop: Stop, leg: Leg) -> int:
        if from_node_id not in self.downwards:
            self.downwards[from_node_id] = []
        self.downwards[from_node_id].append(self.new_node_id)
        self.upwards[self.new_node_id] = from_node_id
        self.node_id_to_stop_id[self.new_node_id] = to_stop.stop_id
        self.legs[from_node_id, self.new_node_id] = leg
        self.size += 1
        self.new_node_id += 1
        return self.new_node_id - 1

    def remove_node(self, node_id: int) -> None:
        if node_id in self.node_id_to_stop_id:  # Only remove node if it exists
            connected_to = self.upwards[node_id]
            self.downwards[connected_to].remove(node_id)
            del self.upwards[node_id]
            del self.node_id_to_stop_id[node_id]
            del self.legs[connected_to, node_id]
            self.size -= 1
            if node_id in self.downwards:
                tmp_copy = list(self.downwards[node_id])
                for child in tmp_copy:
                    self.remove_node(child)
                del self.downwards[node_id]

    def get_journey(self, node_id: int):
        yield node_id
        if node_id in self.upwards:
            parent = self.upwards[node_id]
            yield from self.get_journey(parent)

    def get_departure_ts(self, node_id: int) -> datetime:
        v, u = list(self.get_journey(node_id))[-2:]
        return self.legs[u, v].departure_ts

    def get_duration(self, node_id: int) -> timedelta:
        departure_ts = self.get_departure_ts(node_id)
        arrival_ts = self.legs[self.upwards[node_id], node_id]
        return arrival_ts - departure_ts

    def get_transfers(self, node_id: int) -> int:
        return len(list(self.get_journey(node_id))) - 1

    def get_minimum_heuristic(self, node_id: int, db: GTFSInterface) -> timedelta:
        return min(
            [
                db.get_stop((self.node_id_to_stop_id[stop_id])).heuristic
                for stop_id in self.get_journey(node_id)
            ]
        )

    def is_trip_id_in_route(self, node_id: int, trip_id: int) -> bool:
        return trip_id in [
            self.legs[u, v] for v, u in pairwise(self.get_journey(node_id))
        ]
    
    def is_regional(self, node_id: int, db: GTFSInterface) -> bool:
        return all([
            db.get_trip(self.legs[u, v].trip_id).route.is_regional()
            for v, u in pairwise(self.get_journey(node_id))
        ])

    def is_stop_id_in_route(self, node_id: int, stop_id: int) -> bool:
        return stop_id in [self.get_stop_id(n) for n in self.get_journey(node_id)]

    def truncate_to_trips(self, destination_stop_id: int):
        destination_node_ids = set(
            [
                node_id
                for node_id in self.node_id_to_stop_id
                if self.get_stop_id(node_id) == destination_stop_id
            ]
        )
        node_ids_on_route = set()
        for node_id in destination_node_ids:
            node_ids_on_route.update(self.get_journey(node_id))
        node_ids_to_remove = set(self.node_id_to_stop_id.keys()) - node_ids_on_route
        node_ids_to_remove -= destination_node_ids
        for node_id in node_ids_to_remove:
            self.remove_node(node_id)

    def journey_to_str(
        self, journey_destination_node_id: int, db: GTFSInterface
    ) -> str:
        journey = list(reversed(list(self.get_journey(journey_destination_node_id))))
        identifier = JourneyIdentifier(
            tree_node_id=journey_destination_node_id,
            departure_time=self.get_departure_ts(journey_destination_node_id),
            arrival_time=self.legs[
                self.upwards[journey_destination_node_id], journey_destination_node_id
            ].arrival_ts,
            transfers=len(journey) - 2,
            is_regional=True,
        )

        journey_str = str(identifier) + ': '
        for u, v in pairwise(journey):
            journey_str += f'{db.get_stop(self.get_stop_id(u)).name} -> '
            trip = db.get_trip(self.legs[u, v].trip_id)
            if trip.route.is_regional():
                journey_str += f'🭃{self.legs[u, v].duration.total_seconds() // 60:n} {trip.route.route_short_name}🭎 -> '
            else:
                journey_str += f'🭊🭂{self.legs[u, v].duration.total_seconds() // 60:n} {trip.route.route_short_name}🭍🬿 -> '
        journey_str += f'{db.get_stop(self.get_stop_id(journey[-1])).name}'
        return journey_str

    def to_str(self, db: GTFSInterface) -> str:
        end_node_ids = [
            node_id
            for node_id in self.node_id_to_stop_id
            if node_id not in self.downwards or not self.downwards[node_id]
        ]
        tree_str = ''
        for node_id in end_node_ids:
            tree_str += self.journey_to_str(journey_destination_node_id=node_id, db=db)
            tree_str += '\n'
        return tree_str


class PriorityQueue:
    def __init__(self) -> None:
        self.size = 0
        self._queue = queue.PriorityQueue()

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        item = self._get()
        return item

    def put(self, item: QueueItem):
        self._queue.put(item)
        self.size += 1

    def _get(self) -> QueueItem:
        self.size -= 1
        return self._queue.get()


class RouterTwo:
    def __init__(self, start, destination, timestamp) -> None:
        self.start_id = start
        self.destination_id = destination
        self.departure_time = timestamp

        self.queue = PriorityQueue()
        self.db = GTFSInterface(
            destination_id=self.destination_id, stations=StationPhillip()
        )
        self.con_tree = ConnectionTree(root_stop=self.db.get_stop(self.start_id))
        self.routes_to_destination = 0
        self.arrival_time = None

        self.queue.put(
            QueueItem(
                node_id=0,
                heuristic=compute_heuristic(
                    stations=self.db.stations,
                    stop_id=self.start_id,
                    destination_id=self.destination_id,
                ),
                timestamp=self.departure_time,
            )
        )

    def router_stop_condition(self, stop_id: int | None, timestamp: datetime) -> bool:
        if stop_id == self.destination_id:
            print('Found route to destination')
            self.routes_to_destination += 1
            if self.routes_to_destination == N_MINIMAL_ROUTES_TO_DESTINATION:
                self.arrival_time = timestamp
        if self.arrival_time is not None and timestamp > self.arrival_time:
            return True
        else:
            return False

    def filter_useful_stops(
        self, stops: List[StopTimes], start_node_id: int
    ) -> List[StopTimes]:
        min_heuristic = self.con_tree.get_minimum_heuristic(start_node_id, db=self.db)
        useful_stops = []

        for stop in stops:
            if (
                self.db.get_stop(stop.stop_id).heuristic
                >= min_heuristic + MAX_TIME_DRIVING_AWAY
            ):
                break
            elif self.con_tree.is_stop_id_in_route(start_node_id, stop.stop_id):
                break
            elif stop.stop_id == self.destination_id:
                useful_stops.append(stop)
                break
            else:
                useful_stops.append(stop)

        return useful_stops

    def do_routing(self):
        for q_item in self.queue:
            if self.router_stop_condition(
                stop_id=None,
                timestamp=q_item.timestamp,
            ):
                return

            next_trips = self.db.next_trips(
                stop_id=self.con_tree.get_stop_id(q_item.node_id),
                timestamp=q_item.timestamp,
            )

            # A queue item needs to be added, after next_trips are departed to search for other alternatives
            # As all next trips are departing at the same time, we can just take the first one
            departing_train = self.db.get_stop_time(
                next_trips[0], self.con_tree.get_stop_id(q_item.node_id)
            )
            self.queue.put(
                QueueItem(
                    node_id=q_item.node_id,
                    heuristic=q_item.heuristic,
                    timestamp=departing_train.departure_time
                    + MINIMAL_TIME_FOR_NEXT_SEARCH,
                )
            )

            for trip_id in next_trips:
                if self.con_tree.is_trip_id_in_route(q_item.node_id, trip_id):
                    continue
                trip = self.db.get_trip(trip_id)
                stop_sequence = trip.stop_sequence_from(
                    stop_id=self.con_tree.get_stop_id(q_item.node_id)
                )
                stop_sequence = self.filter_useful_stops(
                    stop_sequence, start_node_id=q_item.node_id
                )
                for stop_time in stop_sequence:
                    stop = self.db.get_stop(stop_time.stop_id)
                    this_node_id = self.con_tree.add_leg(
                        from_node_id=q_item.node_id,
                        to_stop=stop,
                        leg=Leg(
                            trip_id=trip_id,
                            departure_ts=departing_train.departure_time,
                            arrival_ts=stop_time.arrival_time,
                        ),
                    )
                    this_journey = JourneyIdentifier(
                        tree_node_id=this_node_id,
                        departure_time=self.con_tree.get_departure_ts(this_node_id),
                        arrival_time=stop_time.arrival_time,
                        transfers=self.con_tree.get_transfers(node_id=q_item.node_id),
                        is_regional=self.con_tree.is_regional(node_id=q_item.node_id, db=self.db),
                    )
                    if any(
                        [
                            this_journey.is_pareto_dominated_by(other_journey)
                            for other_journey in stop.pareto_connections
                        ]
                    ):
                        self.con_tree.remove_node(this_node_id)
                    else:
                        print(
                            f'Adding leg from {self.db.stations.get_name(eva=self.con_tree.get_stop_id(q_item.node_id))} to {self.db.stations.get_name(eva=stop.stop_id)} at {departing_train.departure_time}'
                        )
                        self.db.add_pareto_connection(stop_time.stop_id, this_journey)
                        if stop.stop_id != self.destination_id:
                            self.queue.put(
                                QueueItem(
                                    node_id=this_node_id,
                                    heuristic=stop.heuristic,
                                    timestamp=stop_time.arrival_time,
                                )
                            )
                        self.router_stop_condition(
                            stop_id=stop.stop_id, timestamp=stop_time.arrival_time
                        )
                print(f"Tree size: {len(self.con_tree)}")


if __name__ == '__main__':
    stations = StationPhillip()

    router = RouterTwo(
        start=stations.get_eva(name='Augsburg Hbf'),
        destination=stations.get_eva(name='Plochingen'),
        timestamp=datetime(2022, 9, 1, 12, 0, 0),
    )
    router.do_routing()
    router.con_tree.truncate_to_trips(destination_stop_id=router.destination_id)
    print(router.con_tree.to_str(db=router.db))
