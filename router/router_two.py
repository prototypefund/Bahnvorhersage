from datetime import datetime, time, timedelta
from itertools import pairwise
from typing import Dict, List, Tuple, Set

from gtfs.stop_times import StopTimes
from helpers.StationPhillip import StationPhillip
from router.constants import (
    MAX_TIME_DRIVING_AWAY,
    MINIMAL_TIME_FOR_NEXT_SEARCH,
    MINIMAL_TRANSFER_TIME,
    N_MINIMAL_ROUTES_TO_DESTINATION,
    EXTRA_TIME,
    MAX_TRANSFERS,
)
from router.datatypes import JourneyIdentifier, Leg, Stop, Trip
from router.exceptions import TimeNotFetchedError
from router.gtfs_db_interface import GTFSInterface
from router.heuristic import compute_heuristic
from router.priority_queue import PriorityQueue, QueueItem
from helpers.profiler import profile

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


# TODO: alternativensuche
# TODO: suche aus dem Zug heraus
# TODO: generell gtfs mergen
# TODO: delfi gtfs mergen
# TODO: tagübergänge regeln
# TODO: transfers
# TODO: connection to string starts with wrong station


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

    def get_journey_identifier(self, node_id: int) -> JourneyIdentifier:
        journey = list(self.get_journey(node_id))
        return JourneyIdentifier(
            tree_node_id=node_id,
            departure_time=self.get_departure_ts(journey),
            arrival_time=self.get_arrival_ts(journey),
            transfers=self.get_transfers(journey),
            is_regional=self.is_regional(journey),
            dist_traveled=self.sum_dist_traveled(journey),
        )

    def get_departure_ts(self, journey: List[int]) -> datetime:
        return self.legs[journey[-1], journey[-2]].departure_ts
        # v, u = list(self.get_journey(node_id))[-2:]
        # return self.legs[u, v].departure_ts

    def get_arrival_ts(self, journey: List[int]) -> datetime:
        return self.legs[journey[1], journey[0]].arrival_ts
        # v, u = islice(self.get_journey(node_id), 2)
        # return self.legs[u, v].arrival_ts

    def get_transfers(self, journey: List[int]) -> int:
        return len(journey) - 1
        # return len(list(self.get_journey(node_id))) - 1

    def is_regional(self, journey: List[int]) -> bool:
        return all([self.legs[u, v].is_regional for v, u in pairwise(journey)])

    def sum_dist_traveled(self, journey: List[int]) -> float:
        return sum([self.legs[u, v].dist_traveled for v, u in pairwise(journey)])

    def get_minimum_heuristic(self, node_id: int, db: GTFSInterface) -> timedelta:
        return min(
            [
                db.get_stop((self.node_id_to_stop_id[stop_id])).heuristic
                for stop_id in self.get_journey(node_id)
            ]
        )

    def is_trip_id_in_route(self, node_id: int, trip_id: int) -> bool:
        return trip_id in [
            self.legs[u, v].trip_id for v, u in pairwise(self.get_journey(node_id))
        ]

    def is_line_in_route(self, node_id: int, trip: Trip, db: GTFSInterface) -> bool:
        return trip.route.route_short_name in [
            db.get_trip(self.legs[u, v].trip_id).route.route_short_name
            for v, u in pairwise(self.get_journey(node_id))
        ]

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
        journey = list(self.get_journey(journey_destination_node_id))
        identifier = JourneyIdentifier(
            tree_node_id=journey_destination_node_id,
            departure_time=self.get_departure_ts(journey=journey),
            arrival_time=self.legs[
                self.upwards[journey_destination_node_id], journey_destination_node_id
            ].arrival_ts,
            transfers=self.get_transfers(journey),
            is_regional=True,
            dist_traveled=self.sum_dist_traveled(journey=journey),
        )
        journey = list(reversed(journey))

        journey_str = str(identifier) + ': '
        journey_str += f'{db.get_stop(self.get_stop_id(journey[0])).name} -> '

        for i in range(len(journey) - 2):
            journey_str += self.legs[journey[i], journey[i + 1]].to_string(db)
            ar_ts = self.legs[journey[i], journey[i + 1]].arrival_ts
            dp_ts = self.legs[journey[i + 1], journey[i + 2]].departure_ts
            transfer_time = dp_ts - ar_ts
            journey_str += f' -> {transfer_time.total_seconds() // 60:n} {db.get_stop(self.get_stop_id(journey[i + 1])).name} -> '

        journey_str += self.legs[journey[-2], journey[-1]].to_string(db)
        journey_str += f' -> {db.get_stop(self.get_stop_id(journey[-1])).name}'
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


class RouterTwo:
    def __init__(self, start: int, destination: int, timestamp: datetime) -> None:
        self.start_id = start
        self.destination_id = destination
        self.departure_time = timestamp

        self.queue = PriorityQueue()
        self.db = GTFSInterface(
            destination_id=self.destination_id, stations=StationPhillip()
        )
        self.con_tree = ConnectionTree(root_stop=self.db.get_stop(self.start_id))
        self.routes_to_destination = 0
        self.latest_arrival_time = self.departure_time

        self.db.prefetch_for_date(self.departure_time.date())

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
            if self.routes_to_destination <= N_MINIMAL_ROUTES_TO_DESTINATION:
                self.latest_arrival_time = max(timestamp, self.latest_arrival_time)
        if (
            self.routes_to_destination > N_MINIMAL_ROUTES_TO_DESTINATION
            and timestamp > self.latest_arrival_time
        ):
            return True
        else:
            return False

    def filter_useful_stops(
        self, stops: List[StopTimes], start_node_id: int
    ) -> List[StopTimes]:
        min_heuristic = self.con_tree.get_minimum_heuristic(start_node_id, db=self.db)
        useful_stops = []

        for stop in stops:
            stop_identifier = (stop.arrival_time, stop.stop_id)

            current_heuristic = self.db.get_stop(stop.stop_id).heuristic
            if current_heuristic < min_heuristic:
                min_heuristic = current_heuristic

            if current_heuristic >= min_heuristic + MAX_TIME_DRIVING_AWAY:
                break
            elif self.con_tree.is_stop_id_in_route(start_node_id, stop.stop_id):
                break
            elif stop_identifier in self.destinations_seen:
                continue
            elif stop.stop_id == self.destination_id:
                self.destinations_seen.add(stop_identifier)
                useful_stops.append(stop)
                break
            else:
                self.destinations_seen.add(stop_identifier)
                useful_stops.append(stop)

        return useful_stops

    def is_node_pareto(self, node_id: int, journey: JourneyIdentifier) -> bool:
        stop = self.db.get_stop(self.con_tree.node_id_to_stop_id[node_id])
        if any(
            [
                journey.is_pareto_dominated_by(other_journey)
                for other_journey in stop.pareto_connections
            ]
        ):
            return False

        destination = self.db.get_stop(self.destination_id)
        if any(
            [
                journey.is_pareto_dominated_by(other_journey)
                for other_journey in destination.pareto_connections
            ]
        ):
            return False

        return True

    def append_or_replace_pareto_connection(
        self, stop_id: int, journey: JourneyIdentifier
    ) -> None:
        stop = self.db.get_stop(stop_id)

        dominated_journeys = list(
            filter(lambda x: x.is_pareto_dominated_by(journey), stop.pareto_connections)
        )
        for dominated_journey in dominated_journeys:
            self.con_tree.remove_node(dominated_journey.tree_node_id)
            stop.pareto_connections.remove(dominated_journey)

        stop.pareto_connections.append(journey)
        self.db.stops[stop_id] = stop

    def do_routing(self):
        processed_q_items = 0
        unique_trip_id = set()
        for q_item in self.queue:
            processed_q_items += 1
            if self.router_stop_condition(
                stop_id=None,
                timestamp=q_item.timestamp,
            ):
                print('processed q items:', processed_q_items)
                print('n processed trips:', len(unique_trip_id))
                return
            if q_item.node_id not in self.con_tree.node_id_to_stop_id:
                continue
            if (
                self.con_tree.get_transfers(list(self.con_tree.get_journey(q_item.node_id)))
                >= MAX_TRANSFERS
            ):
                continue

            from_stop_id = self.con_tree.get_stop_id(q_item.node_id)

            try:
                next_trips = self.db.next_trips(
                    stop_id=from_stop_id,
                    timestamp=q_item.timestamp,
                )
            except TimeNotFetchedError:
                self.queue.put(
                    QueueItem(
                        node_id=q_item.node_id,
                        heuristic=q_item.heuristic,
                        timestamp=datetime.combine(q_item.timestamp.date(), time.max),
                    )
                )
                continue

            # A queue item needs to be added, after next_trips are departed to search for other alternatives
            # As all next trips are departing at the same time, we can just take the first one
            departure_time = self.db.get_stop_time(
                next_trips[0], from_stop_id
            ).departure_time
            self.queue.put(
                QueueItem(
                    node_id=q_item.node_id,
                    heuristic=q_item.heuristic,
                    timestamp=departure_time + MINIMAL_TIME_FOR_NEXT_SEARCH,
                )
            )

            # Routes might have branches in them (joining or splitting trains)
            # if two or more joined trains travel together on the same route, only
            # one of them should be considered for routing
            self.destinations_seen: Set[Tuple[int, int]] = set()

            for trip_id in next_trips:
                unique_trip_id.add(trip_id)
                trip = self.db.get_trip(trip_id)
                print(trip.route.route_short_name)

                stop_sequence = trip.stop_sequence_from(stop_id=from_stop_id)
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
                            departure_ts=departure_time,
                            arrival_ts=stop_time.arrival_time,
                            is_regional=trip.route.is_regional(),
                            dist_traveled=trip.dist_traveled(
                                from_stop_id=from_stop_id,
                                to_stop_id=stop_time.stop_id,
                            ),
                        ),
                    )
                    this_journey = self.con_tree.get_journey_identifier(this_node_id)
                    if self.is_node_pareto(this_node_id, journey=this_journey):
                        print(
                            f'Adding leg from {self.db.stations.get_name(eva=from_stop_id)} to {self.db.stations.get_name(eva=stop.stop_id)} at {departure_time}'
                        )
                        self.append_or_replace_pareto_connection(
                            stop_time.stop_id, this_journey
                        )
                        if stop.stop_id != self.destination_id:
                            self.queue.put(
                                QueueItem(
                                    node_id=this_node_id,
                                    heuristic=stop.heuristic,
                                    timestamp=stop_time.arrival_time
                                    + MINIMAL_TRANSFER_TIME,
                                )
                            )
                        self.router_stop_condition(
                            stop_id=stop.stop_id, timestamp=stop_time.arrival_time
                        )
                    else:
                        self.con_tree.remove_node(this_node_id)
                print(f"Tree size: {len(self.con_tree)}")


if __name__ == '__main__':
    stations = StationPhillip()

    router_two = RouterTwo(
        start=stations.get_eva(name='Augsburg Hbf'),
        destination=stations.get_eva(name='Tübingen Hbf'),
        timestamp=datetime(2023, 5, 1, 13, 0, 0),
    )
    router_two.do_routing()

    # import cloudpickle
    # cloudpickle.dump(router_two.db.trips, open('trips_1_9_22.cpickle', 'wb'))
    # cloudpickle.dump(router_two.db.trips_by_stop, open('trips_by_stop_1_9_22.cpickle', 'wb'))
    # eva_plo = router_two.db.stations.get_eva(name='Plochingen')
    # router_two.con_tree.truncate_to_trips(destination_stop_id=eva_plo)
    router_two.con_tree.truncate_to_trips(destination_stop_id=router_two.destination_id)
    print(router_two.con_tree.to_str(db=router_two.db))
