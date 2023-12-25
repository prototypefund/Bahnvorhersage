from datetime import date, datetime
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
import sqlalchemy
from typing import List, Tuple, Dict
from collections import namedtuple
from itertools import pairwise
from router.constants import (
    MINIMAL_DISTANCE_DIFFERENCE,
    TRAIN_SPEED_MS,
    NO_TRIP_ID,
    MAX_SECONDS_DRIVING_AWAY,
    MAX_EXPECTED_DELAY_SECONDS,
)
from database.engine import sessionfactory
from sqlalchemy.orm import Session as SessionType
from helpers.StationPhillip import StationPhillip
from tqdm import tqdm
from helpers.profiler import profile


# TODO:
# - search all alternatives
# - tagesübergänge regeln
# - clean up code
# - make data format to for api
# - sort out splitting and merging trains
# - numba speedup
# - make api
# - make frontend


Connection = namedtuple(
    'Connection',
    [
        'departure_time',
        'arrival_time',
        'departure_stop_id',
        'arrival_stop_id',
        'trip_id',
        'is_regional',
        'dist_traveled',
    ],
)

JourneyIdentifier = namedtuple(
    'JourneyIdentifier',
    [
        'departure_time',
        'arrival_time',
        'transfers',
        'dist_traveled',
        'is_regional',
        'current_trip_id',
        'min_heuristic',
        'j_ident_id',
        'last_j_ident_id',
        'last_stop_id',
        'last_departure_time',
    ],
)

AlternativeJourneyIdentifier = namedtuple(
    'AlternativeJourneyIdentifier',
    [
        'arrival_time',
        'departure_time',
        'transfers',
        'dist_traveled',
        'is_regional',
        'transfer_time_from_delayed_trip',
        'from_failed_transfer_stop_id',
        'current_trip_id',
        'min_heuristic',
        'j_ident_id',
        'last_j_ident_id',
        'last_stop_id',
        'last_departure_time',
    ],
)


def is_alternative_pareto_dominated(
    journey: AlternativeJourneyIdentifier, other: AlternativeJourneyIdentifier
) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    worse_count = sum(
        [
            journey.arrival_time > other.arrival_time,
            journey.transfers > other.transfers,
            journey.dist_traveled > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
            journey.is_regional < other.is_regional,
            journey.transfer_time_from_delayed_trip
            < other.transfer_time_from_delayed_trip,
            journey.from_failed_transfer_stop_id < other.from_failed_transfer_stop_id,
        ]
    )
    if worse_count:
        equal_count = sum(
            [
                journey.arrival_time == other.arrival_time,
                journey.transfers == other.transfers,
                abs(journey.dist_traveled - other.dist_traveled)
                < MINIMAL_DISTANCE_DIFFERENCE,
                journey.is_regional == other.is_regional,
                journey.transfer_time_from_delayed_trip
                == other.transfer_time_from_delayed_trip,
                journey.from_failed_transfer_stop_id
                == other.from_failed_transfer_stop_id,
            ]
        )

        if equal_count + worse_count == 6:
            return True
    return False


def is_pareto_dominated(journey: JourneyIdentifier, other: JourneyIdentifier) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    worse_count = sum(
        [
            journey.departure_time < other.departure_time,
            journey.arrival_time > other.arrival_time,
            journey.transfers > other.transfers,
            journey.dist_traveled > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
            journey.is_regional < other.is_regional,
        ]
    )
    if worse_count:
        equal_count = sum(
            [
                journey.departure_time == other.departure_time,
                journey.arrival_time == other.arrival_time,
                journey.transfers == other.transfers,
                abs(journey.dist_traveled - other.dist_traveled)
                < MINIMAL_DISTANCE_DIFFERENCE,
                journey.is_regional == other.is_regional,
            ]
        )

        if equal_count + worse_count == 5:
            return True
    return False


def _get_stop_times(service_date: date, session: SessionType) -> List[StopTimes]:
    stmt = (
        sqlalchemy.select(StopTimes)
        .join(Trips, StopTimes.trip_id == Trips.trip_id)
        .join(CalendarDates, Trips.service_id == CalendarDates.service_id)
        .where(
            CalendarDates.date == service_date,
        )
        .order_by(StopTimes.trip_id, StopTimes.stop_sequence)
    )

    stop_times = session.scalars(stmt).all()

    return stop_times


def _get_routes_from_db(
    trip_ids: List[int], session: SessionType
) -> List[Tuple[Routes, int]]:
    stmt = (
        sqlalchemy.select(Routes, Trips.trip_id)
        .join(Trips, Routes.route_id == Trips.route_id)
        .where(Trips.trip_id.in_(trip_ids))
    )

    return session.execute(stmt).all()


def get_routes(trip_ids: List[int], session: SessionType) -> Dict[int, Routes]:
    routes = _get_routes_from_db(trip_ids, session)

    routes = {trip_id: route for route, trip_id in routes}

    return routes


def get_connections(service_date: date, session: SessionType) -> List[Connection]:
    stop_times = _get_stop_times(service_date, session)

    trip_ids = set(stop_time.trip_id for stop_time in stop_times)
    routes = get_routes(trip_ids, session)

    connections = [
        Connection(
            departure_time=int(dp.departure_time.timestamp()),
            arrival_time=int(ar.arrival_time.timestamp()),
            departure_stop_id=dp.stop_id,
            arrival_stop_id=ar.stop_id,
            trip_id=dp.trip_id,
            is_regional=int(routes[dp.trip_id].is_regional()),
            dist_traveled=int(ar.shape_dist_traveled - dp.shape_dist_traveled),
        )
        for dp, ar in pairwise(stop_times)
        if dp.trip_id == ar.trip_id
    ]

    connections = sorted(
        connections,
        key=lambda conn: (conn.departure_time,),
    )

    return connections


def compute_heuristic(
    stations: StationPhillip, stop_id: int, destination_id: int
) -> int:
    distance_m = stations.geographic_distance_by_eva(
        eva_1=stop_id, eva_2=destination_id
    )
    return int(distance_m // TRAIN_SPEED_MS)


@profile()
def csa(
    connections: List[Connection],
    stops: Dict[int, List[JourneyIdentifier]],
    heuristics: Dict[int, int],
):
    j_ident_id = 1  # 0 is reserved for the starting point
    for connection in connections:
        for journey in stops[connection.departure_stop_id]:
            is_same_trip = connection.trip_id == journey.current_trip_id
            if connection.departure_time > journey.arrival_time or is_same_trip:
                if (
                    heuristics[connection.arrival_stop_id] - MAX_SECONDS_DRIVING_AWAY
                ) > journey.min_heuristic:
                    continue

                new_journey = JourneyIdentifier(
                    arrival_time=connection.arrival_time,
                    departure_time=connection.departure_time
                    if journey.current_trip_id == NO_TRIP_ID
                    else journey.departure_time,
                    dist_traveled=journey.dist_traveled + connection.dist_traveled,
                    is_regional=min(journey.is_regional, connection.is_regional),
                    current_trip_id=connection.trip_id,
                    transfers=journey.transfers
                    if is_same_trip
                    else journey.transfers + 1,
                    min_heuristic=min(
                        journey.min_heuristic, heuristics[connection.arrival_stop_id]
                    ),
                    j_ident_id=j_ident_id,
                    last_j_ident_id=journey.j_ident_id,
                    last_stop_id=connection.departure_stop_id,
                    last_departure_time=connection.departure_time,
                )
                j_ident_id += 1
                # Using reversed speeds up the algo by a lot, as journeys are kind of
                # sorted by departure time. It is more likely for a journey to be dominated
                # by a journey that departs later.
                for other_journey in reversed(stops[connection.arrival_stop_id]):
                    if is_pareto_dominated(new_journey, other_journey):
                        break
                else:
                    dest_journeys = stops[connection.arrival_stop_id]
                    dest_journeys = [
                        j
                        for j in dest_journeys
                        if not is_pareto_dominated(j, new_journey)
                    ]
                    dest_journeys.append(new_journey)
                    stops[connection.arrival_stop_id] = dest_journeys
    return stops


def csa_alternative_connections(
    connections: List[Connection],
    stops: Dict[int, List[AlternativeJourneyIdentifier]],
    heuristics: Dict[int, int],
    delayed_trip_id: int,
    min_delay: int,
    max_delay: int,
    failed_transfer_stop_id: int,
):
    j_ident_id = 1  # 0 reserved for start
    for connection in connections:
        for journey in stops[connection.departure_stop_id]:
            is_same_trip = connection.trip_id == journey.current_trip_id
            is_delayed = journey.current_trip_id == delayed_trip_id
            enough_transfer_time = (
                connection.departure_time > (journey.arrival_time + min_delay)
                if is_delayed
                else connection.departure_time > journey.arrival_time
            )
            if is_same_trip or enough_transfer_time:
                if (
                    heuristics[connection.arrival_stop_id] - MAX_SECONDS_DRIVING_AWAY
                ) > journey.min_heuristic:
                    continue

                from_failed_transfer_stop_id = max(
                    journey.from_failed_transfer_stop_id,
                    connection.departure_stop_id == failed_transfer_stop_id,
                )

                transfer_time_from_delayed_trip = (
                    min(
                        connection.departure_time - journey.arrival_time,
                        max_delay,
                    )
                    if is_delayed and from_failed_transfer_stop_id
                    else journey.transfer_time_from_delayed_trip
                )

                new_journey = AlternativeJourneyIdentifier(
                    arrival_time=connection.arrival_time,
                    departure_time=connection.departure_time
                    if journey.current_trip_id == NO_TRIP_ID
                    else journey.departure_time,
                    current_trip_id=connection.trip_id,
                    transfers=journey.transfers
                    if is_same_trip
                    else journey.transfers + 1,
                    dist_traveled=journey.dist_traveled + connection.dist_traveled,
                    is_regional=min(journey.is_regional, connection.is_regional),
                    transfer_time_from_delayed_trip=transfer_time_from_delayed_trip,
                    from_failed_transfer_stop_id=from_failed_transfer_stop_id,
                    min_heuristic=min(
                        journey.min_heuristic, heuristics[connection.arrival_stop_id]
                    ),
                    j_ident_id=j_ident_id,
                    last_j_ident_id=journey.j_ident_id,
                    last_stop_id=connection.departure_stop_id,
                    last_departure_time=connection.departure_time,
                )

                j_ident_id += 1
                # Using reversed speeds up the algo by a lot, as journeys are kind of
                # sorted by departure time. It is more likely for a journey to be dominated
                # by a journey that departs later.
                for other_journey in reversed(stops[connection.arrival_stop_id]):
                    if is_alternative_pareto_dominated(new_journey, other_journey):
                        break
                else:
                    dest_journeys = stops[connection.arrival_stop_id]
                    dest_journeys = [
                        j
                        for j in dest_journeys
                        if not is_alternative_pareto_dominated(j, new_journey)
                    ]
                    dest_journeys.append(new_journey)
                    stops[connection.arrival_stop_id] = dest_journeys
    return stops


def get_journey_identifier(
    stops: Dict[int, List[JourneyIdentifier]], stop_id: int, j_ident_id: int
):
    for journey in stops[stop_id]:
        if journey.j_ident_id == j_ident_id:
            return journey


def extract_journey(
    stops: Dict[int, List[JourneyIdentifier]],
    journey: JourneyIdentifier,
):
    yield journey
    while True:
        journey = get_journey_identifier(
            stops, journey.last_stop_id, journey.last_j_ident_id
        )
        if journey.current_trip_id != NO_TRIP_ID:
            yield journey
        else:
            break


def extract_journeys(
    stops: Dict[int, List[JourneyIdentifier]],
    destination_stop_id: int,
) -> List[List[JourneyIdentifier]]:
    journeys = []
    for journey in stops[destination_stop_id]:
        journeys.append(list(reversed(list(extract_journey(stops, journey)))))
    return journeys


def human_readable_journey_identifier(journey_identifier: JourneyIdentifier):
    departure_ts = datetime.fromtimestamp(journey_identifier.departure_time)
    arrival_ts = datetime.fromtimestamp(journey_identifier.arrival_time)
    duration_seconds = (
        journey_identifier.arrival_time - journey_identifier.departure_time
    )
    transfers = journey_identifier.transfers
    dist_traveled = journey_identifier.dist_traveled

    last_departure_ts = datetime.fromtimestamp(journey_identifier.last_departure_time)

    journey_str = f'{departure_ts.strftime("%H:%M")} - {arrival_ts.strftime("%H:%M")} ({duration_seconds//60:n}min, {transfers}x, last_dp:{last_departure_ts.strftime("%H:%M")}, {dist_traveled // 1000:n}km)'
    return journey_str


def journey_to_str(
    journey: List[JourneyIdentifier], destination_stop_id: int, stations: StationPhillip
):
    departure_ts = datetime.fromtimestamp(journey[0].departure_time)
    arrival_ts = datetime.fromtimestamp(journey[-1].arrival_time)
    duration_seconds = journey[-1].arrival_time - journey[0].departure_time
    transfers = journey[-1].transfers
    dist_traveled = journey[-1].dist_traveled
    is_regional = journey[-1].is_regional

    journey_str = f'{departure_ts.strftime("%H:%M")} - {arrival_ts.strftime("%H:%M")} ({duration_seconds//60:n}min, {transfers}x, {dist_traveled // 1000:n}km, regio:{"y" if is_regional else "n"}): '

    last_departure_stop_id = journey[0].last_stop_id
    last_departure_ts = journey[0].last_departure_time

    for j1, j2 in pairwise(journey):
        if j1.current_trip_id != j2.current_trip_id:
            stop_name = stations.get_name(last_departure_stop_id)
            duration = (j1.arrival_time - last_departure_ts) // 60
            journey_str += f'{stop_name} -> {datetime.fromtimestamp(last_departure_ts).strftime("%H:%M")} 🭃{duration:n}min {j1.current_trip_id}🭎 {datetime.fromtimestamp(j1.arrival_time).strftime("%H:%M")} -> '

            last_departure_stop_id = j2.last_stop_id
            last_departure_ts = j2.last_departure_time

    stop_name = stations.get_name(last_departure_stop_id)
    duration = (journey[-1].arrival_time - last_departure_ts) // 60
    journey_str += f'{stop_name} -> {datetime.fromtimestamp(last_departure_ts).strftime("%H:%M")} 🭃{duration:n}min {j1.current_trip_id}🭎 {arrival_ts.strftime("%H:%M")} -> '

    journey_str += stations.get_name(destination_stop_id)
    return journey_str


def print_journeys(
    journeys: List[List[JourneyIdentifier]],
    destination_stop_id: int,
    stations: StationPhillip,
):
    for journey in journeys:
        print(journey_to_str(journey, destination_stop_id, stations))


Transfer = namedtuple(
    'Transfer',
    [
        'arrival_time',
        'departure_time',
        'stop_id',
        'is_regional',
        'arriving_trip_id',
        'previous_transfer_stop_id',
        'previous_transfer_departure_time',
        'previous_transfer_arrival_time',
    ],
)


def find_alternative_connections(
    journey: List[JourneyIdentifier],
    connections: List[Connection],
    stations: StationPhillip,
    heuristics: Dict[int, int],
    destination_stop_id: int,
):
    transfers: List[Transfer] = []

    last_transfer_station = journey[0].last_stop_id
    last_transfer_departure_time = journey[0].last_departure_time
    last_transfer_arrival_time = journey[0].last_departure_time - 1
    last_transfer_is_regional = 1

    for j1, j2 in pairwise(journey):
        if j1.current_trip_id != j2.current_trip_id:
            transfers.append(
                Transfer(
                    arrival_time=j1.arrival_time,
                    departure_time=j2.last_departure_time,
                    stop_id=j2.last_stop_id,
                    is_regional=last_transfer_is_regional,
                    arriving_trip_id=j1.current_trip_id,
                    previous_transfer_stop_id=last_transfer_station,
                    previous_transfer_departure_time=last_transfer_departure_time,
                    previous_transfer_arrival_time=last_transfer_arrival_time,
                )
            )

            last_transfer_station = j2.last_stop_id
            last_transfer_departure_time = j2.last_departure_time
            last_transfer_arrival_time = j1.arrival_time
            last_transfer_is_regional = j1.is_regional

    alternatives = []

    for transfer in transfers:
        transfer_time_missed = transfer.departure_time - transfer.arrival_time
        if transfer_time_missed > MAX_EXPECTED_DELAY_SECONDS:
            continue

        stops = {stop_id: [] for stop_id in stations.evas}
        stops[transfer.previous_transfer_stop_id].append(
            AlternativeJourneyIdentifier(
                arrival_time=transfer.previous_transfer_arrival_time,
                departure_time=transfer.previous_transfer_arrival_time,
                current_trip_id=NO_TRIP_ID,
                transfers=-1,
                dist_traveled=0,
                is_regional=1,
                transfer_time_from_delayed_trip=0,
                from_failed_transfer_stop_id=0,
                min_heuristic=heuristics[transfer.stop_id],
                j_ident_id=0,
                last_j_ident_id=0,
                last_stop_id=transfer.stop_id,
                last_departure_time=transfer.previous_transfer_arrival_time,
            )
        )

        stops = csa_alternative_connections(
            connections=connections,
            stops=stops,
            heuristics=heuristics,
            delayed_trip_id=transfer.arriving_trip_id,
            min_delay=transfer_time_missed,
            max_delay=MAX_EXPECTED_DELAY_SECONDS,
            failed_transfer_stop_id=transfer.stop_id,
        )

        journeys = extract_journeys(stops, destination_stop_id)
        print('Alternatives')
        print_journeys(journeys, destination_stop_id, stations)


def do_routing(
    start: str,
    destination: str,
    departure_ts: datetime,
    session: SessionType,
    stations: StationPhillip,
):
    start_stop_id = stations.get_eva(name=start)
    destination_stop_id = stations.get_eva(name=destination)

    heuristics = {
        stop_id: compute_heuristic(stations, stop_id, destination_stop_id)
        for stop_id in stations.evas
    }
    stops = {stop_id: [] for stop_id in stations.evas}
    stops[start_stop_id].append(
        JourneyIdentifier(
            departure_time=int(departure_ts.timestamp()),
            arrival_time=int(departure_ts.timestamp()),
            current_trip_id=NO_TRIP_ID,
            transfers=-1,
            min_heuristic=heuristics[start_stop_id],
            dist_traveled=0,
            is_regional=1,
            j_ident_id=0,
            last_j_ident_id=0,
            last_stop_id=start_stop_id,
            last_departure_time=int(departure_ts.timestamp()),
        )
    )

    service_date = departure_ts.date()
    import pickle

    # connections = get_connections(service_date, session=session)
    # pickle.dump(connections, open('test_connections.pickle', 'wb'))
    connections = pickle.load(open('test_connections.pickle', 'rb'))

    stops = csa(connections, stops, heuristics)

    journeys = extract_journeys(stops, destination_stop_id)
    print_journeys(journeys, destination_stop_id, stations)

    find_alternative_connections(
        journey=journeys[0],
        connections=connections,
        stations=stations,
        heuristics=heuristics,
        destination_stop_id=destination_stop_id,
    )


def main():
    engine, Session = sessionfactory()

    stations = StationPhillip(prefer_cache=True)

    with Session() as session:
        do_routing(
            start='Tübingen Hbf',
            destination='Biberach(Riß)',
            departure_ts=datetime(2023, 5, 1, 13, 0, 0),
            session=session,
            stations=stations,
        )


if __name__ == '__main__':
    main()
