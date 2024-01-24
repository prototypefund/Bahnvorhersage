from datetime import date, datetime, UTC
from itertools import pairwise
from typing import Dict, List, Tuple

import sqlalchemy
from sqlalchemy.orm import Session as SessionType
from tqdm import tqdm

from database.engine import sessionfactory
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from helpers.profiler import profile
from helpers.StationPhillip import StationPhillip
from router.constants import (
    MAX_EXPECTED_DELAY_SECONDS,
    MAX_METERS_DRIVING_AWAY,
    MINIMAL_DISTANCE_DIFFERENCE,
    NO_TRIP_ID,
    NO_STOP_ID,
    MINIMUM_TRANSFER_TIME,
)
from router.datatypes import (
    AlternativeReachability,
    Connection,
    Reachability,
    Transfer,
)
from router.printing import print_journeys
from bisect import bisect_left
from router.journey_reconstruction import (
    extract_journeys,
    journey_to_fptf,
    remove_duplicate_journeys,
    clean_alternatives,
)

# TODO:
# - tagesübergänge regeln
# - clean up code
# - make data format to for api
# - sort out splitting and merging trains
# - numba speedup
# - make api


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
            reachability.dist_traveled
            > (other.dist_traveled + MINIMAL_DISTANCE_DIFFERENCE),
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
                abs(reachability.dist_traveled - other.dist_traveled)
                < MINIMAL_DISTANCE_DIFFERENCE,
                reachability.is_regio == other.is_regio,
                reachability.transfer_time_from_delayed_trip
                == other.transfer_time_from_delayed_trip,
                reachability.from_failed_transfer_stop_id
                == other.from_failed_transfer_stop_id,
            ]
        )

        if equal_count + worse_count == 6:
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
    return int(distance_m) # TRAIN_SPEED_MS


def csa(
    connections: List[Connection],
    stops: Dict[int, List[Reachability]],
    heuristics: Dict[int, int],
):
    r_ident_id = 1  # 0 is reserved for the starting point
    for connection in connections:
        for previous in stops[connection.dp_stop_id]:
            is_same_trip = connection.trip_id == previous.current_trip_id
            if (
                connection.dp_ts >= previous.ar_ts + MINIMUM_TRANSFER_TIME
                or is_same_trip
            ):
                if (
                    heuristics[connection.ar_stop_id] - MAX_METERS_DRIVING_AWAY
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


def create_alternative_reachability(
    connection: Connection,
    previous: AlternativeReachability,
    transfer_time_from_delayed_trip: int,
    min_heuristic: int,
    r_ident_id: int,
):
    return AlternativeReachability(
        ar_ts=connection.ar_ts,
        dp_ts=connection.dp_ts
        if previous.current_trip_id == NO_TRIP_ID
        else previous.dp_ts,
        current_trip_id=connection.trip_id,
        transfers=previous.transfers
        if previous.current_trip_id == NO_TRIP_ID
        else previous.transfers + 1,
        dist_traveled=previous.dist_traveled + connection.dist_traveled,
        is_regio=min(previous.is_regio, connection.is_regio),
        transfer_time_from_delayed_trip=transfer_time_from_delayed_trip,
        from_failed_transfer_stop_id=previous.from_failed_transfer_stop_id,
        min_heuristic=min_heuristic,
        r_ident_id=r_ident_id,
        last_r_ident_id=previous.r_ident_id,
        last_stop_id=connection.dp_stop_id,
        last_dp_ts=connection.dp_ts,
    )


def add_connection_to_trip_reachability(
    reachability: AlternativeReachability, connection: Connection, heuristic: int
):
    return AlternativeReachability(
        ar_ts=connection.ar_ts,
        dp_ts=reachability.dp_ts,
        current_trip_id=reachability.current_trip_id,
        transfers=reachability.transfers,
        dist_traveled=reachability.dist_traveled + connection.dist_traveled,
        is_regio=reachability.is_regio,
        transfer_time_from_delayed_trip=reachability.transfer_time_from_delayed_trip,
        from_failed_transfer_stop_id=reachability.from_failed_transfer_stop_id,
        min_heuristic=min(heuristic, reachability.min_heuristic),
        r_ident_id=reachability.r_ident_id,
        last_r_ident_id=reachability.last_r_ident_id,
        last_stop_id=reachability.last_stop_id,
        last_dp_ts=reachability.last_dp_ts,
    )


def reachability_from_trip_reachability(
    reachability: AlternativeReachability,
    r_ident_id: int,
):
    return AlternativeReachability(
        ar_ts=reachability.ar_ts,
        dp_ts=reachability.dp_ts,
        current_trip_id=reachability.current_trip_id,
        transfers=reachability.transfers,
        dist_traveled=reachability.dist_traveled,
        is_regio=reachability.is_regio,
        transfer_time_from_delayed_trip=reachability.transfer_time_from_delayed_trip,
        from_failed_transfer_stop_id=reachability.from_failed_transfer_stop_id,
        min_heuristic=reachability.min_heuristic,
        r_ident_id=r_ident_id,
        last_r_ident_id=reachability.last_r_ident_id,
        last_stop_id=reachability.last_stop_id,
        last_dp_ts=reachability.last_dp_ts,
    )


def add_reachability_to_pareto(
    reachability: AlternativeReachability,
    pareto_set: List[AlternativeReachability],
):
    # Using reversed speeds up the algo by a lot, as reachabilities are kind of
    # sorted by departure time. It is more likely for a reachability to be dominated
    # by a reachability that departs later.
    for other in reversed(pareto_set):
        if is_alternative_pareto_dominated(reachability, other):
            break
    else:
        pareto_set = [
            p
            for p in pareto_set
            if not is_alternative_pareto_dominated(p, reachability)
        ]
        pareto_set.append(reachability)
    return pareto_set


# Difference to normal csa:
# - Don't track dp_ts
# - Track transfer time from delayed trip
# - Track if the reachability is from a failed transfer stop
# - Different stopping criteria

def csa_alternative_connections(
    connections: List[Connection],
    stops: Dict[int, List[AlternativeReachability]],
    trips: Dict[int, List[AlternativeReachability]],
    heuristics: Dict[int, int],
    delayed_trip_id: int,
    min_delay: int,
    destination_stop_id: int,
):
    latest_ar_ts_at_destination = connections[-1].ar_ts
    r_ident_id = 10  # 0 - 9 reserved for start

    for connection in connections:
        if connection.dp_ts > latest_ar_ts_at_destination:
            break
        new_reachabilities: List[AlternativeReachability] = []

        if connection.trip_id in trips:
            # Update trip reachabilities with additional arrival time and distance traveled
            trips[connection.trip_id] = [
                add_connection_to_trip_reachability(
                    trip, connection, heuristics[connection.ar_stop_id]
                )
                for trip in trips[connection.trip_id]
                if not (heuristics[connection.ar_stop_id] - MAX_METERS_DRIVING_AWAY)
                > trip.min_heuristic
            ]

            for trip_reachability in trips[connection.trip_id]:
                new_reachabilities.append(
                    reachability_from_trip_reachability(
                        trip_reachability,
                        r_ident_id,
                    )
                )
                r_ident_id += 1

        for previous in stops[connection.dp_stop_id]:
            is_same_trip = connection.trip_id == previous.current_trip_id
            # The previous trip is delayed, so we can only take the connection if there is
            # enough transfer time (at least min_delay)
            is_delayed = previous.current_trip_id == delayed_trip_id
            enough_transfer_time = (
                connection.dp_ts >= (previous.ar_ts + min_delay + MINIMUM_TRANSFER_TIME)
                if is_delayed
                else connection.dp_ts >= previous.ar_ts + MINIMUM_TRANSFER_TIME
            )
            # Connection is reachable if there is enough transfer time. If it is the same trip,
            # it was already handled in the trip reachability update.
            if enough_transfer_time and not is_same_trip:
                if (
                    heuristics[connection.ar_stop_id] - MAX_METERS_DRIVING_AWAY
                ) > previous.min_heuristic:
                    continue

                transfer_time_from_delayed_trip = (
                    min(
                        connection.dp_ts - previous.ar_ts,
                        MAX_EXPECTED_DELAY_SECONDS,
                    )
                    if is_delayed and previous.from_failed_transfer_stop_id
                    else previous.transfer_time_from_delayed_trip
                )

                reachability = create_alternative_reachability(
                    connection=connection,
                    previous=previous,
                    transfer_time_from_delayed_trip=transfer_time_from_delayed_trip,
                    min_heuristic=min(
                        previous.min_heuristic, heuristics[connection.ar_stop_id]
                    ),
                    r_ident_id=r_ident_id,
                )

                new_reachabilities.append(reachability)

                r_ident_id += 1

                trips[connection.trip_id] = add_reachability_to_pareto(
                    reachability, trips.get(connection.trip_id, [])
                )

        for reachability in new_reachabilities:
            stops[connection.ar_stop_id] = add_reachability_to_pareto(
                reachability, stops[connection.ar_stop_id]
            )

        if connection.ar_stop_id == destination_stop_id and len(new_reachabilities):
            if any(
                [
                    r.from_failed_transfer_stop_id
                    and r.transfer_time_from_delayed_trip == MAX_EXPECTED_DELAY_SECONDS
                    for r in stops[connection.ar_stop_id]
                ]
            ):
                latest_ar_ts_at_destination = max(
                    r.ar_ts for r in stops[connection.ar_stop_id]
                )

    return stops


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
                    dist_traveled=dist_traveled,
                )
            )
            last_transfer_station = c2.dp_stop_id
            n_transfers += 1

    alternatives = []

    for i, transfer in enumerate(transfers):
        transfer_time_missed = transfer.dp_ts - transfer.ar_ts
        if transfer_time_missed > MAX_EXPECTED_DELAY_SECONDS:
            continue

        ar_ts = transfers[i - 1].ar_ts if i > 0 else journey[0].dp_ts - 1

        stops = {stop_id: [] for stop_id in stations.evas}
        stops[transfer.previous_transfer_stop_id].append(
            AlternativeReachability(
                ar_ts=ar_ts,
                dp_ts=transfers[i - 1].ar_ts if i > 0 else journey[0].dp_ts,
                current_trip_id=transfers[i - 1].ar_trip_id if i > 0 else NO_TRIP_ID,
                transfers=transfers[i - 1].transfers if i > 0 else 0,
                dist_traveled=transfers[i - 1].dist_traveled if i > 0 else 0,
                is_regio=transfer.is_regio,  # transfers[i - 1].is_regio if i > 0 else 1,
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
                is_regio=transfer.is_regio,  # transfers[i - 1].is_regio if i > 0 else 1,
                transfer_time_from_delayed_trip=0,
                from_failed_transfer_stop_id=1,
                min_heuristic=heuristics[transfer.stop_id],
                r_ident_id=1,
                last_r_ident_id=1,
                last_stop_id=NO_STOP_ID,
                last_dp_ts=0,
            )
        )

        trips: Dict[int, List[AlternativeReachability]] = dict()

        start_index = bisect_left(connections, ar_ts, key=lambda c: c.dp_ts)

        stops = csa_alternative_connections(
            connections=connections[start_index:],
            stops=stops,
            trips=trips,
            heuristics=heuristics,
            delayed_trip_id=transfer.ar_trip_id,
            min_delay=transfer_time_missed,
            destination_stop_id=destination_stop_id,
        )

        journeys = extract_journeys(
            stops=stops,
            destination_stop_id=destination_stop_id,
            connections=connections,
        )
        alternatives.extend(journeys)

    return alternatives


@profile()
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
    journeys = remove_duplicate_journeys(journeys)

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

    alternatives = [
        remove_duplicate_journeys(alternatives_for_journey)
        for alternatives_for_journey in alternatives
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

        trip_ids = set()
        for journey in journeys:
            for connection in journey:
                trip_ids.add(connection.trip_id)
        for alternatives_for_journey in alternatives:
            for alternative in alternatives_for_journey:
                for connection in alternative:
                    trip_ids.add(connection.trip_id)

        routes = get_routes(trip_ids, session)
        journey_and_alternatives = []

        for journey, alternatives_for_journey in zip(journeys, alternatives):
            # journey = journey_simplification(journey)
            journey_fptf = journey_to_fptf(journey, routes=routes)
            alternatives_fptf = [
                journey_to_fptf(j, routes=routes) for j in alternatives_for_journey
            ]

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
