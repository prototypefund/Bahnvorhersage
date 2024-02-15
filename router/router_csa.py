from bisect import bisect_left
from datetime import date, datetime
from itertools import pairwise
from typing import Dict, List, Tuple

import sqlalchemy
from sqlalchemy.orm import Session as SessionType

from database.engine import sessionfactory
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes
from gtfs.stops import Stops, StopSteffen
from gtfs.trips import Trips
from helpers.profiler import profile
from helpers.StationPhillip import StationPhillip
from router.constants import (
    MAX_EXPECTED_DELAY_SECONDS,
    MAX_METERS_DRIVING_AWAY,
    MINIMAL_DISTANCE_DIFFERENCE,
    MINIMUM_TRANSFER_TIME,
    N_ROUTES_TO_FIND,
    NO_DELAYED_TRIP_ID,
    NO_STOP_ID,
    NO_TRIP_ID,
)
from router.datatypes import Connection, Reachability, Transfer
from router.exceptions import NoRouteFound, NoTimetableFound
from router.journey_reconstruction import (
    FPTFJourney,
    FPTFJourneyAndAlternatives,
    clean_alternatives,
    extract_journeys,
    remove_duplicate_journeys,
)
from router.printing import print_journeys

# TODO:
# - tagesübergänge regeln
# - clean up code
# - sort out splitting and merging trains
# - numba speedup


def is_alternative_pareto_dominated(
    reachability: Reachability, other: Reachability
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


def get_connections(
    service_date: date, session: SessionType, stop_steffen: StopSteffen
) -> List[Connection]:
    stop_times = _get_stop_times(service_date, session)

    trip_ids = set(stop_time.trip_id for stop_time in stop_times)
    routes = get_routes(trip_ids, session)

    connections = [
        Connection(
            dp_ts=int(dp.departure_time.timestamp()),
            ar_ts=int(ar.arrival_time.timestamp()),
            dp_stop_id=stop_steffen.get_stop(dp.stop_id).parent_station,
            ar_stop_id=stop_steffen.get_stop(ar.stop_id).parent_station,
            trip_id=dp.trip_id,
            is_regio=int(routes[dp.trip_id].is_regional()),
            dist_traveled=int(ar.shape_dist_traveled - dp.shape_dist_traveled),
            dp_platform_id=dp.stop_id,
            ar_platform_id=ar.stop_id,
        )
        for dp, ar in pairwise(stop_times)
        if dp.trip_id == ar.trip_id
    ]

    if not connections:
        raise NoTimetableFound(f'Could not find any timetable for {service_date}. Please try another date.')

    connections = sorted(
        connections,
        key=lambda conn: (conn.dp_ts,),
    )

    return connections


def create_alternative_reachability(
    connection: Connection,
    previous: Reachability,
    transfer_time_from_delayed_trip: int,
    min_heuristic: int,
    r_ident_id: int,
):
    return Reachability(
        ar_ts=connection.ar_ts,
        dp_ts=(
            connection.dp_ts
            if previous.current_trip_id == NO_TRIP_ID
            else previous.dp_ts
        ),
        current_trip_id=connection.trip_id,
        transfers=(
            previous.transfers
            if previous.current_trip_id == NO_TRIP_ID
            else previous.transfers + 1
        ),
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
    reachability: Reachability, connection: Connection, heuristic: int
):
    return Reachability(
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
    reachability: Reachability,
    r_ident_id: int,
):
    return Reachability(
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
    reachability: Reachability,
    pareto_set: List[Reachability],
    is_alternative: bool,
):
    if is_alternative:
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
    else:
        for other in reversed(pareto_set):
            if is_pareto_dominated(reachability, other):
                break
        else:
            pareto_set = [
                p for p in pareto_set if not is_pareto_dominated(p, reachability)
            ]
            pareto_set.append(reachability)
        return pareto_set


def csa(
    connections: List[Connection],
    stops: Dict[int, List[Reachability]],
    trips: Dict[int, List[Reachability]],
    heuristics: Dict[int, int],
    delayed_trip_id: int,
    min_delay: int,
    destination_stop_id: int,
    search_alternatives: bool,
):
    latest_ar_ts_at_destination = connections[-1].ar_ts + MINIMUM_TRANSFER_TIME
    r_ident_id = 10  # 0 - 9 reserved for start

    for connection in connections:
        # Early stopping criteria
        if connection.dp_ts > latest_ar_ts_at_destination:
            break

        new_reachabilities: List[Reachability] = []

        # Was the trip reached already?
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
                    reachability,
                    trips.get(connection.trip_id, []),
                    is_alternative=search_alternatives,
                )

        for reachability in new_reachabilities:
            stops[connection.ar_stop_id] = add_reachability_to_pareto(
                reachability,
                stops[connection.ar_stop_id],
                is_alternative=search_alternatives,
            )

        if connection.ar_stop_id == destination_stop_id and len(new_reachabilities):
            if search_alternatives:
                # If a route to the destination was found that has as much transfer time
                # from the delayed trip as the maximum expected delay, we can stop searching
                # for alternatives after the latest arrival time at the destination.
                if any(
                    [
                        r.from_failed_transfer_stop_id
                        and r.transfer_time_from_delayed_trip
                        == MAX_EXPECTED_DELAY_SECONDS
                        for r in stops[destination_stop_id]
                    ]
                ):
                    latest_ar_ts_at_destination = max(
                        r.ar_ts for r in stops[destination_stop_id]
                    )
            else:
                if len(stops[destination_stop_id]) >= N_ROUTES_TO_FIND:
                    latest_ar_ts_at_destination = list(
                        sorted(r.ar_ts for r in stops[destination_stop_id])
                    )[N_ROUTES_TO_FIND - 1]

    return stops


class RouterCSA:
    def __init__(self):
        self.stop_steffen = StopSteffen()

    def do_routing(
        self,
        origin: str,
        destination: str,
        dp_ts: datetime,
        session: SessionType,
    ) -> List[FPTFJourneyAndAlternatives]:
        origin_stop_id = self.stop_steffen.names_to_ids[origin][0]
        destination_stop_id = self.stop_steffen.names_to_ids[destination][0]

        heuristics = {
            stop.stop_id: int(
                self.stop_steffen.get_distance(stop.stop_id, destination_stop_id)
            )
            for stop in self.stop_steffen.stations()
        }
        stops = {stop.stop_id: [] for stop in self.stop_steffen.stations()}
        stops[origin_stop_id].append(
            Reachability(
                dp_ts=int(dp_ts.timestamp()),
                ar_ts=int(dp_ts.timestamp()),
                transfers=0,
                dist_traveled=0,
                is_regio=1,
                transfer_time_from_delayed_trip=0,
                from_failed_transfer_stop_id=0,
                current_trip_id=NO_TRIP_ID,
                min_heuristic=heuristics[origin_stop_id],
                r_ident_id=0,
                last_r_ident_id=0,
                last_stop_id=NO_STOP_ID,
                last_dp_ts=int(dp_ts.timestamp()),
            )
        )

        service_date = dp_ts.date()
        # import pickle

        connections = get_connections(
            service_date, session=session, stop_steffen=self.stop_steffen
        )
        # pickle.dump(connections, open('test_connections.pickle', 'wb'))
        # connections = pickle.load(open('test_connections.pickle', 'rb'))

        trips: Dict[int, List[Reachability]] = dict()

        stops = csa(
            connections=connections,
            stops=stops,
            trips=trips,
            heuristics=heuristics,
            delayed_trip_id=NO_DELAYED_TRIP_ID,
            min_delay=0,
            destination_stop_id=destination_stop_id,
            search_alternatives=False,
        )

        journeys = extract_journeys(stops, destination_stop_id, connections)

        if len(journeys) == 0:
            raise NoRouteFound('No route found')

        journeys = remove_duplicate_journeys(journeys)

        alternatives = [
            self.find_alternative_connections(
                journey=journey,
                connections=connections,
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

        # Format for API
        trip_ids = set()
        for journey in journeys:
            for connection in journey:
                trip_ids.add(connection.trip_id)
        for alternatives_for_journey in alternatives:
            for alternative in alternatives_for_journey:
                for connection in alternative:
                    trip_ids.add(connection.trip_id)

        routes = get_routes(trip_ids, session)
        journey_and_alternatives: List[FPTFJourneyAndAlternatives] = []

        for journey, alternatives_for_journey in zip(journeys, alternatives):
            journey_and_alternatives.append(
                FPTFJourneyAndAlternatives(
                    journey=FPTFJourney.from_journey(
                        journey, routes=routes, stop_steffen=self.stop_steffen
                    ),
                    alternatives=[
                        FPTFJourney.from_journey(
                            alternative, routes=routes, stop_steffen=self.stop_steffen
                        )
                        for alternative in alternatives_for_journey
                    ],
                )
            )

        return journey_and_alternatives

    def find_alternative_connections(
        self,
        journey: List[Connection],
        connections: List[Connection],
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

            stops = {stop.stop_id: [] for stop in self.stop_steffen.stations()}
            stops[transfer.previous_transfer_stop_id].append(
                Reachability(
                    ar_ts=ar_ts,
                    dp_ts=transfers[i - 1].ar_ts if i > 0 else journey[0].dp_ts,
                    current_trip_id=(
                        transfers[i - 1].ar_trip_id if i > 0 else NO_TRIP_ID
                    ),
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
                Reachability(
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

            trips: Dict[int, List[Reachability]] = dict()

            start_index = bisect_left(connections, ar_ts, key=lambda c: c.dp_ts)

            stops = csa(
                connections=connections[start_index:],
                stops=stops,
                trips=trips,
                heuristics=heuristics,
                delayed_trip_id=transfer.ar_trip_id,
                min_delay=transfer_time_missed,
                destination_stop_id=destination_stop_id,
                search_alternatives=True,
            )

            journeys = extract_journeys(
                stops=stops,
                destination_stop_id=destination_stop_id,
                connections=connections,
            )
            alternatives.extend(journeys)

        return alternatives


def main():
    engine, Session = sessionfactory()

    with Session() as session:
        router = RouterCSA()
        router.do_routing(
            origin='Augsburg Hbf',
            destination='Tübingen Hbf',
            dp_ts=datetime(2023, 5, 1, 13, 0, 0),
            session=session,
        )


if __name__ == '__main__':
    main()
