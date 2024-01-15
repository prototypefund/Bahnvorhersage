from datetime import date, datetime, UTC
from itertools import pairwise
from typing import Dict, List, Tuple

import sqlalchemy
from sqlalchemy.orm import Session as SessionType

from database.engine import sessionfactory
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from helpers.profiler import profile
from helpers.StationPhillip import StationPhillip
from router.constants import (
    MAX_EXPECTED_DELAY_SECONDS,
    MAX_SECONDS_DRIVING_AWAY,
    MINIMAL_DISTANCE_DIFFERENCE,
    NO_TRIP_ID,
    NO_STOP_ID,
    TRAIN_SPEED_MS,
)
from router.datatypes import (
    AlternativeReachability,
    Connection,
    Reachability,
    Transfer,
)
from router.printing import print_journeys
from bisect import bisect_left

# TODO:
# - search all alternatives
# - tagesübergänge regeln
# - clean up code
# - make data format to for api
# - sort out splitting and merging trains
# - numba speedup
# - make api
# - make frontend


def utc_ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def is_alternative_pareto_dominated(
    reachability: AlternativeReachability, other: AlternativeReachability
) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    worse_count = sum(
        [
            reachability.ar_ts > other.ar_ts,
            reachability.transfers > other.transfers,
            # reachability.dist_traveled
            # > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
            reachability.is_regio < other.is_regio,
            reachability.transfer_time_from_delayed_trip
            < other.transfer_time_from_delayed_trip,
            reachability.from_failed_transfer_stop_id
            < other.from_failed_transfer_stop_id,
        ]
    )
    if worse_count:
        equal_count = sum(
            [
                reachability.ar_ts == other.ar_ts,
                reachability.transfers == other.transfers,
                # abs(reachability.dist_traveled - other.dist_traveled)
                # < MINIMAL_DISTANCE_DIFFERENCE,
                reachability.is_regio == other.is_regio,
                reachability.transfer_time_from_delayed_trip
                == other.transfer_time_from_delayed_trip,
                reachability.from_failed_transfer_stop_id
                == other.from_failed_transfer_stop_id,
            ]
        )

        if equal_count + worse_count == 5:
            return True
    return False


def is_pareto_dominated(reachability: Reachability, other: Reachability) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    worse_count = sum(
        [
            reachability.dp_ts < other.dp_ts,
            reachability.ar_ts > other.ar_ts,
            reachability.transfers > other.transfers,
            reachability.dist_traveled
            > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
            reachability.is_regio < other.is_regio,
        ]
    )
    if worse_count:
        equal_count = sum(
            [
                reachability.dp_ts == other.dp_ts,
                reachability.ar_ts == other.ar_ts,
                reachability.transfers == other.transfers,
                abs(reachability.dist_traveled - other.dist_traveled)
                < MINIMAL_DISTANCE_DIFFERENCE,
                reachability.is_regio == other.is_regio,
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
            dp_ts=int(dp.departure_time.timestamp()),
            ar_ts=int(ar.arrival_time.timestamp()),
            dp_stop_id=dp.stop_id,
            ar_stop_id=ar.stop_id,
            trip_id=dp.trip_id,
            is_regio=int(routes[dp.trip_id].is_regional()),
            dist_traveled=int(ar.shape_dist_traveled - dp.shape_dist_traveled),
        )
        for dp, ar in pairwise(stop_times)
        if dp.trip_id == ar.trip_id
    ]

    connections = sorted(
        connections,
        key=lambda conn: (conn.dp_ts,),
    )

    return connections


def compute_heuristic(
    stations: StationPhillip, stop_id: int, destination_id: int
) -> int:
    distance_m = stations.geographic_distance_by_eva(
        eva_1=stop_id, eva_2=destination_id
    )
    return int(distance_m // TRAIN_SPEED_MS)


def csa(
    connections: List[Connection],
    stops: Dict[int, List[Reachability]],
    heuristics: Dict[int, int],
):
    r_ident_id = 1  # 0 is reserved for the starting point
    for connection in connections:
        for previous in stops[connection.dp_stop_id]:
            is_same_trip = connection.trip_id == previous.current_trip_id
            if connection.dp_ts > previous.ar_ts or is_same_trip:
                if (
                    heuristics[connection.ar_stop_id] - MAX_SECONDS_DRIVING_AWAY
                ) > previous.min_heuristic:
                    continue

                reachability = Reachability(
                    ar_ts=connection.ar_ts,
                    dp_ts=connection.dp_ts
                    if previous.current_trip_id == NO_TRIP_ID
                    else previous.dp_ts,
                    dist_traveled=previous.dist_traveled + connection.dist_traveled,
                    is_regio=min(previous.is_regio, connection.is_regio),
                    current_trip_id=connection.trip_id,
                    transfers=previous.transfers
                    if is_same_trip
                    else previous.transfers + 1,
                    min_heuristic=min(
                        previous.min_heuristic, heuristics[connection.ar_stop_id]
                    ),
                    r_ident_id=r_ident_id,
                    last_r_ident_id=previous.r_ident_id,
                    last_stop_id=connection.dp_stop_id,
                    last_dp_ts=connection.dp_ts,
                )
                r_ident_id += 1
                # Using reversed speeds up the algo by a lot, as reachabilities are kind of
                # sorted by departure time. It is more likely for a reachability to be dominated
                # by a reachability that departs later.
                for other in reversed(stops[connection.ar_stop_id]):
                    if is_pareto_dominated(reachability, other):
                        break
                else:
                    pareto_set = stops[connection.ar_stop_id]
                    pareto_set = [
                        j
                        for j in pareto_set
                        if not is_pareto_dominated(j, reachability)
                    ]
                    pareto_set.append(reachability)
                    stops[connection.ar_stop_id] = pareto_set
    return stops


def csa_alternative_connections(
    connections: List[Connection],
    stops: Dict[int, List[AlternativeReachability]],
    heuristics: Dict[int, int],
    delayed_trip_id: int,
    min_delay: int,
    max_delay: int,
    failed_transfer_stop_id: int,
):
    r_ident_id = 10  # 0 - 9 reserved for start
    for connection in connections:
        for previous in stops[connection.dp_stop_id]:
            is_same_trip = connection.trip_id == previous.current_trip_id
            is_delayed = previous.current_trip_id == delayed_trip_id
            enough_transfer_time = (
                connection.dp_ts > (previous.ar_ts + min_delay)
                if is_delayed
                else connection.dp_ts > previous.ar_ts
            )
            if is_same_trip or enough_transfer_time:
                if (
                    heuristics[connection.ar_stop_id] - MAX_SECONDS_DRIVING_AWAY
                ) > previous.min_heuristic:
                    continue

                transfer_time_from_delayed_trip = (
                    min(
                        connection.dp_ts - previous.ar_ts,
                        max_delay,
                    )
                    if is_delayed and previous.from_failed_transfer_stop_id
                    else previous.transfer_time_from_delayed_trip
                )

                reachability = AlternativeReachability(
                    ar_ts=connection.ar_ts,
                    dp_ts=connection.dp_ts
                    if previous.current_trip_id == NO_TRIP_ID
                    else previous.dp_ts,
                    current_trip_id=connection.trip_id,
                    transfers=previous.transfers
                    if is_same_trip or previous.current_trip_id == NO_TRIP_ID
                    else previous.transfers + 1,
                    dist_traveled=previous.dist_traveled + connection.dist_traveled,
                    is_regio=min(previous.is_regio, connection.is_regio),
                    transfer_time_from_delayed_trip=transfer_time_from_delayed_trip,
                    from_failed_transfer_stop_id=previous.from_failed_transfer_stop_id,
                    min_heuristic=min(
                        previous.min_heuristic, heuristics[connection.ar_stop_id]
                    ),
                    r_ident_id=r_ident_id,
                    last_r_ident_id=previous.r_ident_id,
                    last_stop_id=connection.dp_stop_id,
                    last_dp_ts=connection.dp_ts,
                )

                r_ident_id += 1
                # Using reversed speeds up the algo by a lot, as reachabilities are kind of
                # sorted by departure time. It is more likely for a reachability to be dominated
                # by a reachability that departs later.
                for other in reversed(stops[connection.ar_stop_id]):
                    if is_alternative_pareto_dominated(reachability, other):
                        break
                else:
                    pareto_set = stops[connection.ar_stop_id]
                    pareto_set = [
                        j
                        for j in pareto_set
                        if not is_alternative_pareto_dominated(j, reachability)
                    ]
                    pareto_set.append(reachability)
                    stops[connection.ar_stop_id] = pareto_set
    return stops


def extract_reachability_chain(
    stops: Dict[int, List[Reachability]],
    destination: Reachability,
) -> List[Reachability]:
    journey = [destination]
    previous = destination
    while True:
        for current in stops[previous.last_stop_id]:
            if current.r_ident_id == previous.last_r_ident_id:
                previous = current
                break
        if previous.last_stop_id != NO_STOP_ID:
            journey.append(previous)
        else:
            break
    return list(reversed(journey))


def extract_reachability_chains(
    stops: Dict[int, List[Reachability]],
    destination_stop_id: int,
) -> List[List[Reachability]]:
    reachability_chains = []
    for reachability in stops[destination_stop_id]:
        reachability_chains.append(extract_reachability_chain(stops, reachability))
    return reachability_chains


def match_connection_to_reachability(
    connections: List[Connection],
    reachability: Reachability,
):
    start_index = bisect_left(
        connections, reachability.last_dp_ts, key=lambda c: c.dp_ts
    )
    for connection in connections[start_index:]:
        if connection.trip_id == reachability.current_trip_id:
            return connection


def extract_journeys(
    stops: Dict[int, List[Reachability]],
    destination_stop_id: int,
    connections: List[Connection],
):
    reachability_chains = extract_reachability_chains(stops, destination_stop_id)

    journeys = []
    for reachability_chain in reachability_chains:
        journey = []
        for reachability in reachability_chain:
            journey.append(match_connection_to_reachability(connections, reachability))
        journeys.append(journey)

    return journeys


def clean_alternatives(journey: List[Connection], alternatives: List[List[Connection]]):
    for i, alternative in enumerate(alternatives):
        for start_index in range(len(alternative)):
            if alternative[start_index] not in journey:
                break
        alternatives[i] = alternative[start_index:]

    return alternatives


def journey_simplification(journey: List[Connection]):
    simplified_journey = []

    dp_ts = journey[0].dp_ts
    dp_stop_id = journey[0].dp_stop_id
    dist_traveled = 0

    for c1, c2 in pairwise(journey):
        dist_traveled += c1.dist_traveled
        if c1.trip_id == c2.trip_id:
            pass
        else:
            simplified_journey.append(
                Connection(
                    dp_ts=dp_ts,
                    ar_ts=c1.ar_ts,
                    dp_stop_id=dp_stop_id,
                    ar_stop_id=c1.ar_stop_id,
                    trip_id=c1.trip_id,
                    is_regio=c1.is_regio,
                    dist_traveled=dist_traveled,
                )
            )
            dp_ts = c2.dp_ts
            dp_stop_id = c2.dp_stop_id
            dist_traveled = 0

    simplified_journey.append(
        Connection(
            dp_ts=dp_ts,
            ar_ts=journey[-1].ar_ts,
            dp_stop_id=dp_stop_id,
            ar_stop_id=journey[-1].ar_stop_id,
            trip_id=journey[-1].trip_id,
            is_regio=journey[-1].is_regio,
            dist_traveled=dist_traveled + journey[-1].dist_traveled,
        )
    )

    return simplified_journey


def journey_to_fptf(journey: List[Connection]):
    fptf_journey = {
        'type': 'journey',
        'legs': [],
    }

    stopovers = []

    dp_ts = journey[0].dp_ts
    dp_stop_id = journey[0].dp_stop_id
    dist_traveled = 0

    for c1, c2 in pairwise(journey):
        dist_traveled += c1.dist_traveled
        if c1.trip_id == c2.trip_id:
            stopovers.append(
                {
                    'type': 'stopover',
                    'stop': c1.ar_stop_id,
                    'arrival': utc_ts_to_iso(c1.ar_ts),
                    'departure': utc_ts_to_iso(c2.dp_ts),
                    'distTraveled': dist_traveled,
                }
            )

        else:
            fptf_journey['legs'].append(
                {
                    'origin': dp_stop_id,
                    'destination': c1.ar_stop_id,
                    'departure': utc_ts_to_iso(dp_ts),
                    'arrival': utc_ts_to_iso(c1.ar_ts),
                    'stopovers': stopovers,
                    'mode': 'train',
                    'public': True,
                    'distTraveled': dist_traveled,
                    'isRegio': bool(c1.is_regio),
                }
            )
            dp_ts = c2.dp_ts
            dp_stop_id = c2.dp_stop_id
            dist_traveled = 0
            stopovers = []
    
    fptf_journey['legs'].append(
        {
            'origin': dp_stop_id,
            'destination': journey[-1].ar_stop_id,
            'departure': utc_ts_to_iso(dp_ts),
            'arrival': utc_ts_to_iso(journey[-1].ar_ts),
            'stopovers': stopovers,
            'mode': 'train',
            'public': True,
            'distTraveled': dist_traveled + journey[-1].dist_traveled,
            'isRegio': bool(journey[-1].is_regio),
        }
    )

    return fptf_journey


def find_alternative_connections(
    journey: List[Connection],
    connections: List[Connection],
    stations: StationPhillip,
    heuristics: Dict[int, int],
    destination_stop_id: int,
):
    transfers: List[Transfer] = []

    is_regio = 1
    last_transfer_station = journey[0].dp_stop_id
    n_transfers = 0
    dist_traveled = 0

    for c1, c2 in pairwise(journey):
        dist_traveled += c1.dist_traveled
        if c1.trip_id != c2.trip_id:
            is_regio = min(is_regio, c1.is_regio)
            transfers.append(
                Transfer(
                    ar_ts=c1.ar_ts,
                    dp_ts=c2.dp_ts,
                    stop_id=c2.dp_stop_id,
                    is_regio=is_regio,
                    ar_trip_id=c1.trip_id,
                    previous_transfer_stop_id=last_transfer_station,
                    transfers=n_transfers,
                    dist_traveled=dist_traveled
                )
            )
            last_transfer_station = c2.dp_stop_id
            n_transfers += 1

    alternatives = []

    for i, transfer in enumerate(transfers):
        transfer_time_missed = transfer.dp_ts - transfer.ar_ts
        if transfer_time_missed > MAX_EXPECTED_DELAY_SECONDS:
            continue

        stops = {stop_id: [] for stop_id in stations.evas}
        stops[transfer.previous_transfer_stop_id].append(
            AlternativeReachability(
                ar_ts=transfers[i - 1].ar_ts if i > 0 else journey[0].dp_ts - 1,
                dp_ts=transfers[i - 1].ar_ts if i > 0 else journey[0].dp_ts,
                current_trip_id=transfers[i - 1].ar_trip_id if i > 0 else NO_TRIP_ID,
                transfers=transfers[i - 1].transfers if i > 0 else 0,
                dist_traveled=transfers[i - 1].dist_traveled if i > 0 else 0,
                is_regio=transfer.is_regio, # transfers[i - 1].is_regio if i > 0 else 1,
                transfer_time_from_delayed_trip=0,
                from_failed_transfer_stop_id=0,
                min_heuristic=heuristics[transfer.previous_transfer_stop_id],
                r_ident_id=0,
                last_r_ident_id=0,
                last_stop_id=NO_STOP_ID,
                last_dp_ts=0,
            )
        )

        stops[transfer.stop_id].append(
            AlternativeReachability(
                ar_ts=transfer.ar_ts,
                dp_ts=transfer.dp_ts,
                current_trip_id=transfer.ar_trip_id,
                transfers=transfer.transfers,
                dist_traveled=transfer.dist_traveled,
                is_regio=transfer.is_regio, # transfers[i - 1].is_regio if i > 0 else 1,
                transfer_time_from_delayed_trip=0,
                from_failed_transfer_stop_id=1,
                min_heuristic=heuristics[transfer.stop_id],
                r_ident_id=1,
                last_r_ident_id=1,
                last_stop_id=NO_STOP_ID,
                last_dp_ts=0,
            )
        )

        stops = csa_alternative_connections(
            connections=connections,
            stops=stops,
            heuristics=heuristics,
            delayed_trip_id=transfer.ar_trip_id,
            min_delay=transfer_time_missed,
            max_delay=MAX_EXPECTED_DELAY_SECONDS,
            failed_transfer_stop_id=transfer.stop_id,
        )

        journeys = extract_journeys(
            stops=stops,
            destination_stop_id=destination_stop_id,
            connections=connections,
        )
        alternatives.extend(journeys)

    return alternatives


def do_routing(
    start: str,
    destination: str,
    dp_ts: datetime,
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
        Reachability(
            dp_ts=int(dp_ts.timestamp()),
            ar_ts=int(dp_ts.timestamp()),
            current_trip_id=NO_TRIP_ID,
            transfers=-1,
            min_heuristic=heuristics[start_stop_id],
            dist_traveled=0,
            is_regio=1,
            r_ident_id=0,
            last_r_ident_id=0,
            last_stop_id=NO_STOP_ID,
            last_dp_ts=int(dp_ts.timestamp()),
        )
    )

    service_date = dp_ts.date()
    import pickle

    # connections = get_connections(service_date, session=session)
    # pickle.dump(connections, open('test_connections.pickle', 'wb'))
    connections = pickle.load(open('test_connections.pickle', 'rb'))

    stops = csa(connections, stops, heuristics)

    journeys = extract_journeys(stops, destination_stop_id, connections)

    alternatives = [
        find_alternative_connections(
            journey=journey,
            connections=connections,
            stations=stations,
            heuristics=heuristics,
            destination_stop_id=destination_stop_id,
        )
        for journey in journeys
    ]

    alternatives = [
        clean_alternatives(journey=journey, alternatives=alternatives_for_journey)
        for journey, alternatives_for_journey in zip(journeys, alternatives)
    ]

    return journeys, alternatives


def main():
    engine, Session = sessionfactory()

    stations = StationPhillip(prefer_cache=True)

    with Session() as session:
        journeys, alternatives = do_routing(
            start='Augsburg Hbf',
            destination='Tübingen Hbf',
            dp_ts=datetime(2023, 5, 1, 13, 0, 0),
            session=session,
            stations=stations,
        )

        journey_and_alternatives = []

        for journey, alternatives_for_journey in zip(journeys, alternatives):
            # journey = journey_simplification(journey)
            journey_fptf = journey_to_fptf(journey)
            alternatives_fptf = [journey_to_fptf(j) for j in alternatives_for_journey]

            journey_and_alternatives.append(
                {
                    'journey': journey_fptf,
                    'alternatives': alternatives_fptf,
                }
            )

            print('Journey:')
            print('--------')
            print_journeys(
                journeys=[journey],
                stations=stations,
            )

            print('Alternatives:')
            print('-------------')
            print_journeys(
                journeys=alternatives_for_journey,
                stations=stations,
            )
            print('\n\n')

        import json

        json.dump(
            journey_and_alternatives,
            open('test.json', 'w'),
            indent=4,
        )


if __name__ == '__main__':
    main()
