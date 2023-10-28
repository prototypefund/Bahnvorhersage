import networkx as nx
from datetime import datetime, timedelta
import queue
from dataclasses import dataclass, field, replace
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from gtfs.calendar_dates import CalendarDates
import sqlalchemy
from database.engine import sessionfactory
import enum
from typing import Dict, Any, List, Set, Tuple
import sqlalchemy.orm
from pyvis.network import Network
from rtd_crawler.hash64 import xxhash64
from helpers.pairwise import pairwise
from helpers.StationPhillip import StationPhillip
from sortedcontainers import SortedKeyList

# Note: Every timestamp must be in UTC.


class AlreadyInGraphError(Exception):
    pass


class SegmentType(enum.Enum):
    TRIP = 1
    TRANSFER = 2


class AddedStatus(enum.Enum):
    ADDED = 1
    ALREADY_IN_GRAPH = 2


@dataclass
class Segment:
    segment_type: SegmentType
    duration: timedelta
    trip_id: int | None


@dataclass(order=True)
class Node:
    timestamp: datetime = field(compare=False)
    stop_id: int = field(compare=False)
    route_destination_id: int = field(compare=False)
    heuristic: timedelta = field(init=False, repr=False, compare=False)
    stop_name: str = field(init=False, compare=False)
    priority: datetime = field(init=False, repr=False)
    TRAIN_SPEED_MS = 70  # m/s (250 km/h)

    def __post_init__(self):
        self.stop_name = stations.get_name(eva=self.stop_id)
        distance_m = stations.geographic_distance_by_eva(
            eva_1=self.stop_id, eva_2=self.route_destination_id
        )
        self.heuristic = timedelta(seconds=distance_m / self.TRAIN_SPEED_MS)
        self.priority = self.timestamp + self.heuristic

    @staticmethod
    def from_stop_time(
        stop_time: StopTimes, route_destination_id: int, arrival: bool = False
    ):
        if arrival:
            return Node(
                timestamp=stop_time.arrival_time,
                stop_id=stop_time.stop_id,
                route_destination_id=route_destination_id,
            )
        else:
            return Node(
                timestamp=stop_time.departure_time,
                stop_id=stop_time.stop_id,
                route_destination_id=route_destination_id,
            )

    def hash(self) -> int:
        return xxhash64(self.timestamp.isoformat() + str(self.stop_id))


class UniquePriorityQueue:
    def __init__(self, destination) -> None:
        self.destination = destination
        self.routes_to_destination = 0
        self.stopping_time = None
        self.size = 0
        self._queue = queue.PriorityQueue()
        self._set = set()

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        item = self._get()
        if self.stopping_time is not None and item.priority > self.stopping_time:
            raise StopIteration
        return item

    def put(self, item: Node, added_status: AddedStatus):
        if item.hash() not in self._set:
            if item.stop_id == self.destination:
                self.routes_to_destination += 1
                if self.routes_to_destination == 3:
                    self.stopping_time = item.timestamp
            else:
                if added_status == AddedStatus.ADDED:
                    self._queue.put(item)
                    self._set.add(item.hash())
                    self.size += 1
                    return False

    def _get(self) -> Node:
        self.size -= 1
        return self._queue.get()


def iter_routes(routes: nx.DiGraph, start_node_id: int = 0):
    yield start_node_id
    for node in nx.bfs_successors(routes, start_node_id):
        yield from node[1]


def iter_routes_predecessors(
    routes: nx.DiGraph,
    start_node_id: int,
    stopping_stop_id: int = None,
    visited: Set[int] = None,
):
    if visited is not None and start_node_id in visited:
        return

    yield start_node_id

    if visited is None:
        visited = set()
    visited.add(start_node_id)

    if (
        stopping_stop_id is None
        or routes.nodes[start_node_id]['data'].stop_id != stopping_stop_id
    ):
        for node_id in routes.predecessors(start_node_id):
            yield from iter_routes_predecessors(
                routes, node_id, stopping_stop_id, visited
            )


def get_stop(trip_id: int, stop_sequence: int, session: sqlalchemy.orm.Session):
    stmt = (
        sqlalchemy.select(StopTimes)
        .where(StopTimes.trip_id == trip_id)
        .where(StopTimes.stop_sequence == stop_sequence)
    )
    return session.scalars(stmt).one()


def node_id_of_stop(routes: nx.DiGraph, stop_id: int) -> int:
    if stop_id == routes.nodes[0]['data'].stop_id:
        return 0
    for node in routes.nodes:
        if routes.nodes[node]['data'].stop_id == stop_id:
            # The node might have a predecessor with the same stop_id,
            # but we want the first node with this stop_id, so skip this
            if any(
                routes.nodes[p]['data'].stop_id == stop_id
                for p in routes.predecessors(node)
            ):
                continue
            else:
                return node
    raise ValueError(f'Stop {stop_id} not found in routes')


def graph_of_stop(routes: nx.DiGraph, stop_id: int) -> nx.DiGraph:
    stop_graph = nx.DiGraph()
    try:
        node_id = node_id_of_stop(routes, stop_id)
    except ValueError:
        return stop_graph

    stop_graph.add_node(
        len(stop_graph.nodes), node_id=node_id, data=routes.nodes[node_id]['data']
    )

    while True:
        successors = list(routes.successors(node_id))
        for s in successors:
            if routes.nodes[s]['data'].stop_id == stop_id:
                stop_graph.add_node(
                    len(stop_graph.nodes), node_id=s, data=routes.nodes[s]['data']
                )
                node_id = s
                break
        else:
            break

    for u, v in pairwise(range(len(stop_graph.nodes))):
        segment = Segment(
            segment_type=SegmentType.TRIP,
            duration=stop_graph.nodes[v]['data'].timestamp
            - stop_graph.nodes[u]['data'].timestamp,
            trip_id=None,
        )
        stop_graph.add_edge(u, v, data=segment)

    return stop_graph


def node_id_in_graph(node: Node, graph: nx.DiGraph) -> int:
    node_hash = node.hash()
    for n in iter_routes(graph):
        if graph.nodes[n]['data'].hash() == node_hash:
            return n
    return None


def display_routes(routes: nx.DiGraph, filename: str = 'nx.html'):
    net = Network(
        directed=True,
        select_menu=True,
        filter_menu=True,
    )
    net.toggle_physics(False)
    net.from_nx(routes)
    for network_node in net.nodes:
        network_node['label'] = network_node["data"].stop_name
        network_node['color'] = '#' + hex(xxhash64(network_node["data"].stop_name))[-6:]
        network_node['title'] = network_node["data"].timestamp.isoformat()
        y, x = stations.get_location(network_node["data"].stop_id)
        network_node['x'] = x * 1_000
        network_node['y'] = -y * 1_000
        del network_node["data"]
    for network_edge in net.edges:
        network_edge['title'] = f'{network_edge["data"].duration.total_seconds() // 60}, {network_edge["data"].trip_id}'
        network_edge['color'] = (
            'black' if network_edge["data"].segment_type == SegmentType.TRIP else 'blue'
        )
        del network_edge["data"]
    net.save_graph(filename)


class Router:
    def __init__(self, start, destination, timestamp, session) -> None:
        self.start_id = start
        self.destination_id = destination
        self.departure_ts = timestamp

        self.session = session

        self.stations_timetable: Dict[int, Any] = {}
        self.visited_trips: Dict[int, int] = {}

        self.routes = nx.DiGraph()
        # self.stations = StationPhillip(prefer_cache=True)
        self.routes_to_destination = 0
        self.stopping_time = None

        self._queue = queue.PriorityQueue()
        self._nodes_in_queue = set()
        self.queue_size = 0

    def get_routes(self):
        pass

    def get_departures_from_db(
        self, station: int, timestamp: datetime
    ) -> List[StopTimes]:
        stmt = (
            sqlalchemy.select(StopTimes)
            .join(Trips, StopTimes.trip_id == Trips.trip_id)
            .join(CalendarDates, Trips.service_id == CalendarDates.service_id)
            .where(
                StopTimes.stop_id == station,
                StopTimes.departure_time != None,
                StopTimes.departure_time >= timestamp,
                CalendarDates.date == timestamp.date(),
            )
            .order_by(StopTimes.departure_time)
        )

        return self.session.scalars(stmt).all()

    def get_departures(self, station: int, timestamp: datetime) -> List[StopTimes]:
        if station in self.stations_timetable:
            departures = [
                d for d in self.stations_timetable[station].irange_key(timestamp)
            ]
            if len(departures) > 0:
                return departures
        if not station in self.stations_timetable:
            departures = self.get_departures_from_db(station, timestamp)
            self.stations_timetable[station] = SortedKeyList(
                departures, key=lambda x: x.departure_time
            )
            return [d for d in self.stations_timetable[station].irange_key(timestamp)]

    def get_next_departures(self, station: int, timestamp: datetime) -> list[StopTimes]:
        departures = self.get_departures(station, timestamp)
        min_departure = min(departures, key=lambda x: x.departure_time)
        return [
            x for x in departures if x.departure_time == min_departure.departure_time
        ]

    def get_trip(self, trip_id: int, min_stop_sequence: int):
        if (
            trip_id in self.visited_trips
            and self.visited_trips[trip_id] <= min_stop_sequence
        ):
            return []
        else:
            self.visited_trips[trip_id] = min_stop_sequence
        stmt = (
            sqlalchemy.select(StopTimes)
            .where(
                StopTimes.trip_id == trip_id,
                StopTimes.stop_sequence >= min_stop_sequence,
            )
            .order_by(StopTimes.stop_sequence)
        )
        return self.session.scalars(stmt).all()

    def put_in_queue(self, node: Node):
        if node.hash() not in self._nodes_in_queue:
            if node.stop_id == self.destination_id:
                self.routes_to_destination += 1
                if self.routes_to_destination == 3:
                    self.stopping_time = node.timestamp
            else:
                self._queue.put(node)
                self._nodes_in_queue.add(node.hash())
                self.queue_size += 1

    def iter_queue(self):
        while True:
            item = self._queue.get()
            self.queue_size -= 1
            if self.stopping_time is not None and item.priority > self.stopping_time:
                break
            yield item

    def get_nodes_to_connect_to(self, node: Node) -> Tuple[int, int]:
        stop_graph = graph_of_stop(self.routes, node.stop_id)

        # The stop is not yet in the graph
        if len(stop_graph.nodes) == 0:
            return None, None
        # The node is before the first node of the stop
        elif stop_graph.nodes[0]['data'].timestamp > node.timestamp:
            return None, stop_graph.nodes[0]['node_id']
        # The node is after the last node of the stop
        elif (
            stop_graph.nodes[len(stop_graph.nodes) - 1]['data'].timestamp
            < node.timestamp
        ):
            return stop_graph.nodes[len(stop_graph.nodes) - 1]['node_id'], None

        for edge in nx.bfs_edges(stop_graph, 0):
            # Node is between two nodes in graph
            if (
                stop_graph.nodes[edge[0]]['data'].timestamp
                < node.timestamp
                < stop_graph.nodes[edge[1]]['data'].timestamp
            ):
                return (
                    stop_graph.nodes[edge[0]]['node_id'],
                    stop_graph.nodes[edge[1]]['node_id'],
                )

        raise ValueError(f'Node {node} could not find a place in the graph')

    def insert_node(self, node: Node) -> (int, AddedStatus):
        in_graph = node_id_in_graph(node, self.routes)
        if in_graph is not None:
            return in_graph, AddedStatus.ALREADY_IN_GRAPH

        predecessor_id, successor_id = self.get_nodes_to_connect_to(node)

        node_id = len(self.routes.nodes)
        self.routes.add_node(node_id, data=node)

        if predecessor_id is not None:
            segment = Segment(
                segment_type=SegmentType.TRANSFER,
                duration=node.timestamp
                - self.routes.nodes[predecessor_id]['data'].timestamp,
                trip_id=None,
            )
            self.routes.add_edge(predecessor_id, node_id, data=segment)
        if successor_id is not None:
            segment = Segment(
                segment_type=SegmentType.TRANSFER,
                duration=self.routes.nodes[successor_id]['data'].timestamp
                - node.timestamp,
                trip_id=None,
            )
            self.routes.add_edge(node_id, successor_id, data=segment)
        if predecessor_id is not None and successor_id is not None:
            self.routes.remove_edge(predecessor_id, successor_id)

        return node_id, AddedStatus.ADDED

    def already_passed(self, start_node_id: int, stop_ids: Set[int]) -> Set[int]:
        passed_stops = set()
        for node_id in iter_routes_predecessors(self.routes, start_node_id):
            if self.routes.nodes[node_id]['data'].stop_id in stop_ids:
                passed_stops.add(self.routes.nodes[node_id]['data'].stop_id)
        return passed_stops

    def add_segment_to_routes(self, start: Node, destination: Node, trip_id: int):
        print(
            f'{start.timestamp}: found segment from {start.stop_name} to {destination.stop_name}'
        )

        start_id, start_inserted = self.insert_node(start)
        destination_id, destination_inserted = self.insert_node(destination)

        if not self.routes.has_edge(start_id, destination_id):
            segment = Segment(
                segment_type=SegmentType.TRIP,
                duration=destination.timestamp - start.timestamp,
                trip_id=trip_id,
            )
            self.routes.add_edge(start_id, destination_id, data=segment)

        if start_inserted == AddedStatus.ADDED:
            self.put_in_queue(
                replace(start, timestamp=start.timestamp + timedelta(seconds=1))
            )
        if destination_inserted == AddedStatus.ADDED:
            self.put_in_queue(destination)

    def do_routing(self):
        first_node = Node(
            timestamp=self.departure_ts,
            stop_id=self.start_id,
            route_destination_id=self.destination_id,
        )
        self.put_in_queue(first_node)
        self.routes.add_node(0, data=first_node)

        for node in self.iter_queue():
            print(f'{node.timestamp}: Processing {node.stop_name}')
            next_trips = self.get_next_departures(
                station=node.stop_id, timestamp=node.timestamp
            )
            for next_trip in next_trips:
                trip = self.get_trip(
                    next_trip.trip_id,
                    min_stop_sequence=next_trip.stop_sequence,
                )
                if len(trip) == 0:
                    continue

                stop_ids = {s.stop_id for s in trip}
                predecessor_id = node_id_in_graph(
                    Node.from_stop_time(trip[0], self.destination_id), self.routes
                )
                if predecessor_id is None:
                    predecessor_id, _ = self.get_nodes_to_connect_to(
                        Node.from_stop_time(trip[0], self.destination_id)
                    )
                if predecessor_id is not None:
                    passed_stops = self.already_passed(predecessor_id, stop_ids)
                else:
                    passed_stops = set()

                for d, a in pairwise(trip):
                    departure = Node.from_stop_time(
                        d, self.destination_id, arrival=False
                    )
                    arrival = Node.from_stop_time(a, self.destination_id, arrival=True)

                    if arrival.stop_id in passed_stops:
                        print('found loop')
                        break

                    self.add_segment_to_routes(departure, arrival, trip_id=d.trip_id)

                    if arrival.stop_id == self.destination_id:
                        break

                display_routes(self.routes, 'tmp.html')
                print(f'Queue size: {self.queue_size}')
                print(
                    f'Graph size: Nodes: {len(self.routes.nodes)} Edges: {len(self.routes.edges)}'
                )

    def trim_routes(self):
        graph_of_destination = graph_of_stop(self.routes, self.destination_id)
        last_node = graph_of_destination.nodes[len(graph_of_destination.nodes) - 1]
        ids_to_keep = set()

        for node_id in iter_routes_predecessors(
            self.routes, last_node['node_id'], stopping_stop_id=self.start_id
        ):
            ids_to_keep.add(node_id)

        for node_id in list(self.routes.nodes):
            if node_id not in ids_to_keep:
                self.routes.remove_node(node_id)


if __name__ == "__main__":
    stations = StationPhillip()

    engine, Session = sessionfactory()

    with Session() as session:
        router = Router(
            start=stations.get_eva('Augsburg Hbf'),
            destination=stations.get_eva('Ulm Hbf'),
            timestamp=datetime(2022, 9, 1, 12, 0, 0),
            session=session,
        )
        router.do_routing()
        display_routes(router.routes, 'tmp.html')
        # import pickle

        # pickle.dump(router.routes, open('routes_biberach.pickle', 'wb'))
        # router.routes = pickle.load(open('routes_biberach.pickle', 'rb'))
        router.trim_routes()
        display_routes(router.routes, 'tmp_trimmed.html')
    print('Done')

    # routes = get_routes(
    #     start=stations.get_eva('Augsburg Hbf'),
    #     destination=stations.get_eva('Ulm Hbf'),
    #     departure_ts=datetime(2022, 9, 1, 12, 0, 0),
    # )

    # routes = get_routes_using_full_trips(
    #     start=stations.get_eva('Augsburg Hbf'),
    #     destination=stations.get_eva('Ulm Hbf'),
    #     departure_ts=datetime(2022, 9, 1, 12, 0, 0),
    # )

    # display_routes(routes, 'tmp.html')
    # print('Done')
