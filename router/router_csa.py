from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Dict, List, Tuple

import sqlalchemy
from sqlalchemy.orm import Session as SessionType

from database.engine import sessionfactory
from gtfs.connections import Connections as DBConnections
from gtfs.routes import Routes
from gtfs.stops import StopSteffen
from gtfs.transfers import Transfer, get_transfers
from gtfs.trips import Trips
from router.constants import (
    ADDITIONAL_SEARCH_WINDOW_HOURS,
    MAX_EXPECTED_DELAY_SECONDS,
    MAX_METERS_DRIVING_AWAY,
    MAX_SEARCH_WINDOW_HOURS,
    MINIMUM_TRANSFER_TIME,
    N_ROUTES_TO_FIND,
    NO_DELAYED_TRIP_ID,
    NO_STOP_ID,
    NO_TRIP_ID,
    STANDART_SEARCH_WINDOW_HOURS,
    WALK_FROM_ORIGIN_TRIP_ID,
    WALKING_TRIP_ID,
)
from router.datatypes import Changeover, Connection, Reachability
from router.exceptions import NoRouteFound, NoTimetableFound
from router.journey_reconstruction import (
    FPTFJourney,
    FPTFJourneyAndAlternatives,
    clean_alternatives,
    extract_journeys,
    remove_duplicate_journeys,
)
from router.pareto import relaxed_alternative_pareto_dominated, relaxed_pareto_dominated

# TODO:
# - sort out splitting and merging trains


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


def create_reachability(
    connection: Connection,
    previous: Reachability,
    transfer_time_from_delayed_trip: int,
    min_heuristic: int,
    r_ident_id: int,
):
    if previous.current_trip_id == NO_TRIP_ID:
        dp_ts = connection.dp_ts
    elif previous.current_trip_id == WALK_FROM_ORIGIN_TRIP_ID:
        # If the previous trip was a walk from the origin, it is possible to depart
        # later at the origin, so that the connection is just reachable.
        walk_duration = previous.ar_ts - previous.dp_ts
        dp_ts = connection.dp_ts - walk_duration - MINIMUM_TRANSFER_TIME
    else:
        dp_ts = previous.dp_ts
    return Reachability(
        ar_ts=connection.ar_ts,
        dp_ts=dp_ts,
        current_trip_id=connection.trip_id,
        changeovers=(
            previous.changeovers
            if previous.current_trip_id == NO_TRIP_ID
            or previous.current_trip_id == WALK_FROM_ORIGIN_TRIP_ID
            else previous.changeovers + 1
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
        walk_from_delayed_trip=False,
        last_changeover_duration=connection.dp_ts - previous.ar_ts,
    )


def add_transfer_to_reachability(
    reachability: Reachability,
    transfer: Transfer,
    from_delayed,
    heuristic: int,
    r_ident_id: int,
):
    return Reachability(
        ar_ts=reachability.ar_ts + transfer.duration,
        dp_ts=reachability.dp_ts,
        current_trip_id=(
            WALK_FROM_ORIGIN_TRIP_ID
            if reachability.current_trip_id == NO_TRIP_ID
            else WALKING_TRIP_ID
        ),
        changeovers=reachability.changeovers,
        dist_traveled=reachability.dist_traveled + transfer.distance,
        is_regio=reachability.is_regio,
        transfer_time_from_delayed_trip=reachability.transfer_time_from_delayed_trip,  # TODO
        from_failed_transfer_stop_id=reachability.from_failed_transfer_stop_id,
        min_heuristic=min(heuristic, reachability.min_heuristic),
        r_ident_id=r_ident_id,
        last_r_ident_id=reachability.r_ident_id,
        last_stop_id=transfer.from_stop,
        last_dp_ts=reachability.ar_ts,
        walk_from_delayed_trip=from_delayed,
        last_changeover_duration=reachability.last_changeover_duration,
    )


def add_connection_to_trip_reachability(
    reachability: Reachability, connection: Connection, heuristic: int
):
    return Reachability(
        ar_ts=connection.ar_ts,
        dp_ts=reachability.dp_ts,
        current_trip_id=reachability.current_trip_id,
        changeovers=reachability.changeovers,
        dist_traveled=reachability.dist_traveled + connection.dist_traveled,
        is_regio=reachability.is_regio,
        transfer_time_from_delayed_trip=reachability.transfer_time_from_delayed_trip,
        from_failed_transfer_stop_id=reachability.from_failed_transfer_stop_id,
        min_heuristic=min(heuristic, reachability.min_heuristic),
        r_ident_id=reachability.r_ident_id,
        last_r_ident_id=reachability.last_r_ident_id,
        last_stop_id=reachability.last_stop_id,
        last_dp_ts=reachability.last_dp_ts,
        walk_from_delayed_trip=False,
        last_changeover_duration=reachability.last_changeover_duration,
    )


def reachability_from_trip_reachability(
    reachability: Reachability,
    r_ident_id: int,
):
    return Reachability(
        ar_ts=reachability.ar_ts,
        dp_ts=reachability.dp_ts,
        current_trip_id=reachability.current_trip_id,
        changeovers=reachability.changeovers,
        dist_traveled=reachability.dist_traveled,
        is_regio=reachability.is_regio,
        transfer_time_from_delayed_trip=reachability.transfer_time_from_delayed_trip,
        from_failed_transfer_stop_id=reachability.from_failed_transfer_stop_id,
        min_heuristic=reachability.min_heuristic,
        r_ident_id=r_ident_id,
        last_r_ident_id=reachability.last_r_ident_id,
        last_stop_id=reachability.last_stop_id,
        last_dp_ts=reachability.last_dp_ts,
        walk_from_delayed_trip=False,
        last_changeover_duration=reachability.last_changeover_duration,
    )


def add_reachability_to_pareto(
    reachability: Reachability,
    pareto_set: List[Reachability],
    is_alternative: bool,
):
    was_appended = False
    if is_alternative:
        # Using reversed speeds up the algo by a lot, as reachabilities are kind of
        # sorted by departure time. It is more likely for a reachability to be dominated
        # by a reachability that departs later.
        for other in reversed(pareto_set):
            if relaxed_alternative_pareto_dominated(reachability, other):
                break
        else:
            pareto_set = [
                p
                for p in pareto_set
                if not relaxed_alternative_pareto_dominated(p, reachability)
            ]
            pareto_set.append(reachability)
            was_appended = True
    else:
        was_dominated = False
        for other in reversed(pareto_set):
            if relaxed_pareto_dominated(reachability, other):
                was_dominated = True
                break
            if other.dp_ts < reachability.dp_ts:
                break
        if not was_dominated:
            pareto_set = [
                p
                for p in pareto_set
                if p.ar_ts < reachability.ar_ts
                or not relaxed_pareto_dominated(p, reachability)
            ]
            pareto_set.append(reachability)
            pareto_set = sorted(pareto_set, key=lambda r: r.dp_ts)
            was_appended = True
    return pareto_set, was_appended


def csa(
    connections: List[Connection],
    stops: Dict[int, List[Reachability]],
    trips: Dict[int, List[Reachability]],
    transfers: Dict[int, List[Transfer]],
    heuristics: Dict[int, int],
    delayed_trip_id: int,
    min_delay: int,
    destination_stop_id: int,
    search_alternatives: bool,
    r_ident_id: int,
    early_stopping_ts: int,
):
    for connection in connections:
        # Early stopping criteria
        if connection.dp_ts > early_stopping_ts:
            return stops, True, early_stopping_ts, r_ident_id

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
                    if (is_delayed and previous.from_failed_transfer_stop_id)
                    or previous.walk_from_delayed_trip
                    else previous.transfer_time_from_delayed_trip
                )

                reachability = create_reachability(
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

                trips[connection.trip_id], _ = add_reachability_to_pareto(
                    reachability,
                    trips.get(connection.trip_id, []),
                    is_alternative=search_alternatives,
                )

        for reachability in new_reachabilities:
            stops[connection.ar_stop_id], was_added = add_reachability_to_pareto(
                reachability,
                stops[connection.ar_stop_id],
                is_alternative=search_alternatives,
            )
            # Relax walking segments here if reachability was added
            if was_added and connection.ar_stop_id in transfers:
                for transfer in transfers[connection.ar_stop_id]:
                    walk = add_transfer_to_reachability(
                        reachability=reachability,
                        transfer=transfer,
                        from_delayed=int(
                            delayed_trip_id == reachability.current_trip_id
                            and reachability.from_failed_transfer_stop_id
                        ),
                        heuristic=heuristics[transfer.to_stop],
                        r_ident_id=r_ident_id,
                    )
                    r_ident_id += 1
                    stops[transfer.to_stop], _ = add_reachability_to_pareto(
                        walk,
                        stops[transfer.to_stop],
                        is_alternative=search_alternatives,
                    )

        if connection.ar_stop_id == destination_stop_id and len(new_reachabilities):
            # Generate stopping condition for early stopping
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
                    early_stopping_ts = max(r.ar_ts for r in stops[destination_stop_id])
            else:
                if len(stops[destination_stop_id]) >= N_ROUTES_TO_FIND:
                    early_stopping_ts = list(
                        sorted(r.ar_ts for r in stops[destination_stop_id])
                    )[N_ROUTES_TO_FIND - 1]

    return stops, False, early_stopping_ts, r_ident_id


@dataclass
class RoutingParams:
    origin: str
    destination: str
    origin_stop_id: int
    destination_stop_id: int
    dp_ts: datetime
    session: SessionType
    n_hours_to_future: int
    heuristics: Dict[int, int]
    connections: List[Connection]


class RouterCSA:
    def __init__(self):
        self.stop_steffen = StopSteffen()
        self.transfers = get_transfers()
        self.params: RoutingParams = None

    def run_csa(
        self,
        stops: Dict[int, List[Reachability]],
        search_alternatives: bool,
        r_ident_id: int = 10,  # 0-9 are reserved for special cases
        delayed_trip_id: int = NO_DELAYED_TRIP_ID,
        min_delay: int = 0,
    ) -> Dict[int, List[Reachability]]:
        trips: Dict[int, List[Reachability]] = dict()
        new_connections = self.params.connections

        while True:
            early_stopping_ts = new_connections[-1].ar_ts + MINIMUM_TRANSFER_TIME
            stops, routing_finished, early_stopping_ts, r_ident_id = csa(
                connections=new_connections,
                stops=stops,
                trips=trips,
                transfers=self.transfers,
                heuristics=self.params.heuristics,
                delayed_trip_id=delayed_trip_id,
                min_delay=min_delay,
                destination_stop_id=self.params.destination_stop_id,
                search_alternatives=search_alternatives,
                early_stopping_ts=early_stopping_ts,
                r_ident_id=r_ident_id,
            )
            if (
                routing_finished
                or self.params.n_hours_to_future >= MAX_SEARCH_WINDOW_HOURS
            ):
                break
            else:
                new_connections = DBConnections.get_for_routing(
                    session=self.params.session,
                    from_ts=self.params.dp_ts
                    + timedelta(hours=self.params.n_hours_to_future),
                    to_ts=self.params.dp_ts
                    + timedelta(
                        hours=self.params.n_hours_to_future
                        + ADDITIONAL_SEARCH_WINDOW_HOURS
                    ),
                )
                if len(new_connections) == 0:
                    break
                self.params.n_hours_to_future += ADDITIONAL_SEARCH_WINDOW_HOURS
                self.params.connections.extend(new_connections)

        # Filter out reachabilities at the destination, that arrive after early_stopping_ts,
        # as these might not represent an optimal journey
        stops[self.params.destination_stop_id] = [
            r
            for r in stops[self.params.destination_stop_id]
            if r.ar_ts <= early_stopping_ts
        ]
        return stops

    def do_routing(
        self,
        origin: str,
        destination: str,
        dp_ts: datetime,
        session: SessionType,
    ) -> List[FPTFJourneyAndAlternatives]:
        destination_stop_id = self.stop_steffen.names_to_ids[destination][0]
        self.params = RoutingParams(
            origin=origin,
            destination=destination,
            origin_stop_id=self.stop_steffen.names_to_ids[origin][0],
            destination_stop_id=destination_stop_id,
            dp_ts=dp_ts,
            session=session,
            n_hours_to_future=STANDART_SEARCH_WINDOW_HOURS,
            heuristics={
                stop.stop_id: int(
                    self.stop_steffen.get_distance(stop.stop_id, destination_stop_id)
                )
                for stop in self.stop_steffen.stations()
            },
            connections=DBConnections.get_for_routing(
                session=session,
                from_ts=dp_ts,
                to_ts=dp_ts + timedelta(hours=STANDART_SEARCH_WINDOW_HOURS),
            ),
        )

        if len(self.params.connections) == 0:
            raise NoTimetableFound('No timetable found for given date and time')

        stops = {stop.stop_id: [] for stop in self.stop_steffen.stations()}
        origin_reachability = Reachability(
            dp_ts=int(dp_ts.timestamp()),
            ar_ts=int(dp_ts.timestamp()),
            changeovers=0,
            dist_traveled=0,
            is_regio=1,
            transfer_time_from_delayed_trip=0,
            from_failed_transfer_stop_id=0,
            current_trip_id=NO_TRIP_ID,
            min_heuristic=self.params.heuristics[self.params.origin_stop_id],
            r_ident_id=0,
            last_r_ident_id=0,
            last_stop_id=NO_STOP_ID,
            last_dp_ts=int(dp_ts.timestamp()),
            walk_from_delayed_trip=False,
            last_changeover_duration=0,
        )
        stops[self.params.origin_stop_id].append(origin_reachability)
        # Relax walking segments here from origin
        r_ident_id = 10
        if self.params.origin_stop_id in self.transfers:
            for transfer in self.transfers[self.params.origin_stop_id]:
                walk = add_transfer_to_reachability(
                    reachability=origin_reachability,
                    transfer=transfer,
                    from_delayed=0,
                    heuristic=self.params.heuristics[transfer.to_stop],
                    r_ident_id=r_ident_id,
                )
                r_ident_id += 1
                stops[transfer.to_stop], _ = add_reachability_to_pareto(
                    walk,
                    stops[transfer.to_stop],
                    is_alternative=False,
                )

        stops = self.run_csa(stops, search_alternatives=False, r_ident_id=r_ident_id)

        journeys = extract_journeys(
            stops,
            self.params.destination_stop_id,
            self.params.connections,
            transfers=self.transfers,
        )

        if len(journeys) == 0:
            raise NoRouteFound('No route found')

        journeys = remove_duplicate_journeys(journeys)

        # # Print for debugging
        # trip_ids = set()
        # for journey in journeys:
        #     for connection in journey:
        #         trip_ids.add(connection.trip_id)
        # routes = get_routes(trip_ids, session)
        # print('Journeys:')
        # print_journeys(journeys, self.stop_steffen, routes=routes)
        # return

        alternatives = [
            self.find_alternative_connections(
                journey=journey,
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

        journeys_and_alternatives = self.to_fptf(journeys, alternatives)

        self.params = None
        return journeys_and_alternatives

    def find_alternative_connections(
        self,
        journey: List[Connection],
    ):
        changeovers: List[Changeover] = []

        is_regio = 1
        last_transfer_station = journey[0].dp_stop_id
        n_changeovers = 0
        dist_traveled = 0

        start_index = 1 if journey[0].trip_id == WALKING_TRIP_ID else 0
        end_index = (
            len(journey) - 1 if journey[-1].trip_id == WALKING_TRIP_ID else len(journey)
        )
        for c1, c2 in pairwise(journey[start_index:end_index]):
            dist_traveled += c1.dist_traveled
            if c1.trip_id != c2.trip_id:
                is_regio = min(is_regio, c1.is_regio)
                changeovers.append(
                    Changeover(
                        ar_ts=c1.ar_ts,
                        dp_ts=c2.dp_ts,
                        stop_id=c2.dp_stop_id,
                        is_regio=is_regio,
                        ar_trip_id=c1.trip_id,
                        previous_transfer_stop_id=last_transfer_station,
                        changeovers=n_changeovers,
                        dist_traveled=dist_traveled,
                    )
                )
                last_transfer_station = c2.dp_stop_id
                n_changeovers += 1

        alternatives = []

        for i, transfer in enumerate(changeovers):
            transfer_time_missed = transfer.dp_ts - transfer.ar_ts
            if transfer_time_missed > MAX_EXPECTED_DELAY_SECONDS:
                continue

            ar_ts = changeovers[i - 1].ar_ts if i > 0 else journey[0].dp_ts - 1

            stops = {stop.stop_id: [] for stop in self.stop_steffen.stations()}
            stops[transfer.previous_transfer_stop_id].append(
                Reachability(
                    ar_ts=ar_ts,
                    dp_ts=changeovers[i - 1].ar_ts if i > 0 else journey[0].dp_ts,
                    current_trip_id=(
                        changeovers[i - 1].ar_trip_id if i > 0 else NO_TRIP_ID
                    ),
                    changeovers=changeovers[i - 1].changeovers if i > 0 else 0,
                    dist_traveled=changeovers[i - 1].dist_traveled if i > 0 else 0,
                    is_regio=transfer.is_regio,
                    transfer_time_from_delayed_trip=0,
                    from_failed_transfer_stop_id=0,
                    min_heuristic=self.params.heuristics[
                        transfer.previous_transfer_stop_id
                    ],
                    r_ident_id=0,
                    last_r_ident_id=0,
                    last_stop_id=NO_STOP_ID,
                    last_dp_ts=0,
                    walk_from_delayed_trip=False,
                    last_changeover_duration=0,
                )
            )

            stops[transfer.stop_id].append(
                Reachability(
                    ar_ts=transfer.ar_ts,
                    dp_ts=transfer.dp_ts,
                    current_trip_id=transfer.ar_trip_id,
                    changeovers=transfer.changeovers,
                    dist_traveled=transfer.dist_traveled,
                    is_regio=transfer.is_regio,
                    transfer_time_from_delayed_trip=0,
                    from_failed_transfer_stop_id=1,
                    min_heuristic=self.params.heuristics[transfer.stop_id],
                    r_ident_id=1,
                    last_r_ident_id=1,
                    last_stop_id=NO_STOP_ID,
                    last_dp_ts=0,
                    walk_from_delayed_trip=False,
                    last_changeover_duration=0,
                )
            )

            stops = self.run_csa(
                stops,
                search_alternatives=True,
                delayed_trip_id=transfer.ar_trip_id,
                min_delay=transfer_time_missed,
            )

            journeys = extract_journeys(
                stops=stops,
                destination_stop_id=self.params.destination_stop_id,
                connections=self.params.connections,
                transfers=self.transfers,
            )
            alternatives.extend(journeys)

        return alternatives

    def to_fptf(
        self,
        journeys: List[List[Connection]],
        alternatives: List[List[List[Connection]]],
    ) -> List[FPTFJourneyAndAlternatives]:
        trip_ids = set()
        for journey in journeys:
            for connection in journey:
                trip_ids.add(connection.trip_id)
        for alternatives_for_journey in alternatives:
            for alternative in alternatives_for_journey:
                for connection in alternative:
                    trip_ids.add(connection.trip_id)

        routes = get_routes(trip_ids, self.params.session)

        # for journey, alternatives_for_journey in zip(journeys, alternatives):
        #     print('Journey:')
        #     print_journeys([journey], self.stop_steffen, routes=routes)

        #     print('Alternatives:')
        #     print_journeys(alternatives_for_journey, self.stop_steffen, routes=routes)

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


def main():
    engine, Session = sessionfactory()

    with Session() as session:
        router = RouterCSA()
        router.do_routing(
            origin='Berlin Hbf',
            destination='Augsburg Hbf',
            dp_ts=datetime(2024, 2, 27, 13, 0, 0),
            session=session,
        )


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    main()
