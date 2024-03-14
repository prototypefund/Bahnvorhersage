from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise

from gtfs.routes import Routes, RouteType
from gtfs.stops import StopSteffen
from gtfs.transfers import Transfer
from router.constants import (
    MINIMUM_TRANSFER_TIME,
    NO_STOP_ID,
    WALK_FROM_ORIGIN_TRIP_ID,
    WALKING_TRIP_ID,
)
from router.datatypes import Connection, Reachability


def utc_ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def extract_reachability_chain(
    stops: dict[int, list[Reachability]],
    destination: Reachability,
) -> list[Reachability]:
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
    stops: dict[int, list[Reachability]],
    destination_stop_id: int,
) -> list[list[Reachability]]:
    reachability_chains = []
    for reachability in stops[destination_stop_id]:
        reachability_chains.append(extract_reachability_chain(stops, reachability))
    return reachability_chains


def match_connection_to_reachability(
    connections: list[Connection],
    reachability: Reachability,
):
    start_index = bisect_left(
        connections, reachability.last_dp_ts, key=lambda c: c.dp_ts
    )
    for connection in connections[start_index:]:
        if connection.trip_id == reachability.current_trip_id:
            return connection


def get_next_connection(
    connections: list[Connection],
    connection: Connection,
):
    start_index = bisect_left(connections, connection.ar_ts, key=lambda c: c.dp_ts)
    for c in connections[start_index:]:
        if c.trip_id == connection.trip_id:
            return c


def match_transfer_to_reachability(
    transfers: dict[int, list[Transfer]],
    reachability: Reachability,
    to_stop_id: int,
    is_from_origin: bool = False,
    next_dp_ts: int = None,
):
    for transfer in transfers[reachability.last_stop_id]:
        if transfer.to_stop == to_stop_id:
            if is_from_origin:
                return Connection(
                    dp_ts=next_dp_ts - transfer.duration - MINIMUM_TRANSFER_TIME,
                    ar_ts=next_dp_ts - MINIMUM_TRANSFER_TIME,
                    dp_stop_id=reachability.last_stop_id,
                    ar_stop_id=to_stop_id,
                    trip_id=WALKING_TRIP_ID,
                    is_regio=True,
                    dist_traveled=transfer.distance,
                    dp_platform_id=reachability.last_stop_id,
                    ar_platform_id=to_stop_id,
                )
            else:
                return Connection(
                    dp_ts=reachability.last_dp_ts,
                    ar_ts=reachability.ar_ts,
                    dp_stop_id=reachability.last_stop_id,
                    ar_stop_id=to_stop_id,
                    trip_id=WALKING_TRIP_ID,
                    is_regio=True,
                    dist_traveled=transfer.distance,
                    dp_platform_id=reachability.last_stop_id,
                    ar_platform_id=to_stop_id,
                )


def extract_journeys(
    stops: dict[int, list[Reachability]],
    destination_stop_id: int,
    connections: list[Connection],
    transfers: dict[int, list[Transfer]],
) -> list[list[Connection]]:
    reachability_chains = extract_reachability_chains(stops, destination_stop_id)

    journeys = []
    for reachability_chain in reachability_chains:
        sparse_journey: list[Connection] = []
        for i, reachability in enumerate(reachability_chain):
            if reachability.current_trip_id == WALKING_TRIP_ID:
                sparse_journey.append(
                    match_transfer_to_reachability(
                        transfers=transfers,
                        reachability=reachability,
                        to_stop_id=(
                            destination_stop_id
                            if i == len(reachability_chain) - 1
                            else reachability_chain[i + 1].last_stop_id
                        ),
                    )
                )
            elif reachability.current_trip_id == WALK_FROM_ORIGIN_TRIP_ID:
                sparse_journey.append(  # TODO
                    match_transfer_to_reachability(
                        transfers=transfers,
                        reachability=reachability,
                        to_stop_id=(
                            destination_stop_id
                            if i == len(reachability_chain) - 1
                            else reachability_chain[i + 1].last_stop_id
                        ),
                        is_from_origin=True,
                        next_dp_ts=(
                            reachability.ar_ts + MINIMUM_TRANSFER_TIME
                            if i == len(reachability_chain) - 1
                            else reachability_chain[i + 1].last_dp_ts
                        ),
                    )
                )
            else:
                sparse_journey.append(
                    match_connection_to_reachability(connections, reachability)
                )

        journey: list[Connection] = []
        if len(sparse_journey) == 1:
            journey.append(sparse_journey[0])
        else:
            for c1, c2 in pairwise(sparse_journey):
                journey.append(c1)
                while journey[-1].ar_stop_id != c2.dp_stop_id:
                    journey.append(get_next_connection(connections, journey[-1]))

            journey.append(c2)
        while journey[-1].ar_stop_id != destination_stop_id:
            journey.append(get_next_connection(connections, journey[-1]))

        journeys.append(journey)

    return journeys


def connections_equal(c1: Connection, c2: Connection):
    return (
        c1.dp_ts == c2.dp_ts
        and c1.ar_ts == c2.ar_ts
        and c1.dp_stop_id == c2.dp_stop_id
        and c1.ar_stop_id == c2.ar_stop_id
        # and c1.trip_id == c2.trip_id
        and c1.is_regio == c2.is_regio
        and c1.dist_traveled == c2.dist_traveled
    )


def clean_alternatives(journey: list[Connection], alternatives: list[list[Connection]]):
    for i, alternative in enumerate(alternatives):
        for start_index in range(len(alternative)):
            if alternative[start_index] not in journey:
                break
        alternatives[i] = alternative[start_index:]

    return alternatives


def journey_simplification(journey: list[Connection]):
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


def remove_duplicate_journeys(journeys: list[list[Connection]]):
    journeys = sorted(journeys, key=lambda j: j[0].dp_ts)
    unique_journeys: list[list[Connection]] = []
    for journey in journeys:
        for unique_journey in reversed(unique_journeys):
            if journey[0].dp_ts > unique_journey[0].dp_ts:
                unique_journeys.append(journey)
                break
            elif len(journey) == len(unique_journey):
                if all(
                    [
                        connections_equal(c1, c2)
                        for c1, c2 in zip(journey, unique_journey)
                    ]
                ):
                    break
            unique_journeys.append(journey)
            break
        else:
            unique_journeys.append(journey)

    return unique_journeys


@dataclass
class FPTFStop:
    id: int
    name: str
    type: str = 'stop'


@dataclass
class FPTFLine:
    id: int
    name: str
    operator: str
    isRegio: bool
    productName: str
    mode: str
    type: str = 'line'

    @staticmethod
    def from_route(route: Routes) -> 'FPTFLine':
        return FPTFLine(
            id=route.route_id,
            name=route.route_short_name,
            operator=route.agency_id,
            isRegio=route.is_regional(),
            productName=route.route_short_name.split(' ')[0],
            mode='bus' if route.route_type == RouteType.BUS else 'train',
        )

    @staticmethod
    def walking() -> 'FPTFLine':
        return FPTFLine(
            id=0,
            name='walking',
            operator='walking',
            isRegio=True,
            productName='walking',
            mode='walking',
        )


@dataclass
class FPTFStopover:
    stop: FPTFStop
    arrival: str
    arrivalPlatform: str
    departure: str
    departurePlatform: str
    distance: int
    type: str = 'stopover'


@dataclass
class FPTFLeg:
    origin: FPTFStop
    destination: FPTFStop
    departure: str
    departurePlatform: str
    arrival: str
    arrivalPlatform: str
    stopovers: list[FPTFStopover]
    distance: int
    line: FPTFLine | None
    walking: bool = False
    mode: str = 'train'
    public: bool = True


@dataclass
class FPTFJourney:
    legs: list[FPTFLeg]
    type: str = 'journey'

    @staticmethod
    def from_journey(
        journey: list[Connection], routes: dict[int, Routes], stop_steffen: StopSteffen
    ) -> 'FPTFJourney':
        legs: list[FPTFLeg] = []
        stopovers: list[FPTFStopover] = []

        dp_ts = journey[0].dp_ts
        dp_stop_id = journey[0].dp_platform_id
        dist_traveled = 0

        for c1, c2 in pairwise(journey):
            dist_traveled += c1.dist_traveled
            if c1.trip_id == c2.trip_id:
                current_stop = stop_steffen.get_stop(stop_id=c1.ar_platform_id)
                stopovers.append(
                    FPTFStopover(
                        stop=FPTFStop(
                            id=current_stop.stop_name, name=current_stop.stop_name
                        ),
                        arrival=utc_ts_to_iso(c1.ar_ts),
                        arrivalPlatform=current_stop.platform_code,
                        departure=utc_ts_to_iso(c2.dp_ts),
                        departurePlatform=current_stop.platform_code,
                        distance=dist_traveled,
                    )
                )
            elif c1.trip_id == WALKING_TRIP_ID:
                ar_stop = stop_steffen.get_stop(stop_id=c1.ar_stop_id)
                dp_stop = stop_steffen.get_stop(stop_id=c1.dp_stop_id)
                legs.append(
                    FPTFLeg(
                        origin=FPTFStop(id=dp_stop.stop_name, name=dp_stop.stop_name),
                        destination=FPTFStop(
                            id=ar_stop.stop_name, name=ar_stop.stop_name
                        ),
                        departure=utc_ts_to_iso(c1.dp_ts),
                        departurePlatform=None,
                        arrival=utc_ts_to_iso(c1.ar_ts),
                        arrivalPlatform=None,
                        stopovers=[],
                        distance=c1.dist_traveled,
                        walking=True,
                        mode='walking',
                        line=FPTFLine.walking(),
                    )
                )

                dp_ts = c2.dp_ts
                dp_stop_id = c2.dp_platform_id
                dist_traveled = 0
                stopovers: list[FPTFStopover] = []
            else:
                line = FPTFLine.from_route(routes[c1.trip_id])
                dp_stop = stop_steffen.get_stop(stop_id=dp_stop_id)
                ar_stop = stop_steffen.get_stop(stop_id=c1.ar_platform_id)
                legs.append(
                    FPTFLeg(
                        origin=FPTFStop(id=dp_stop.stop_name, name=dp_stop.stop_name),
                        destination=FPTFStop(
                            id=ar_stop.stop_name, name=ar_stop.stop_name
                        ),
                        departure=utc_ts_to_iso(dp_ts),
                        departurePlatform=dp_stop.platform_code,
                        arrival=utc_ts_to_iso(c1.ar_ts),
                        arrivalPlatform=ar_stop.platform_code,
                        stopovers=stopovers,
                        distance=dist_traveled,
                        line=line,
                    )
                )

                dp_ts = c2.dp_ts
                dp_stop_id = c2.dp_platform_id
                dist_traveled = 0
                stopovers: list[FPTFStopover] = []

        if journey[-1].trip_id == WALKING_TRIP_ID:
            ar_stop = stop_steffen.get_stop(stop_id=journey[-1].ar_stop_id)
            dp_stop = stop_steffen.get_stop(stop_id=journey[-1].dp_stop_id)
            legs.append(
                FPTFLeg(
                    origin=FPTFStop(id=dp_stop.stop_name, name=dp_stop.stop_name),
                    destination=FPTFStop(id=ar_stop.stop_name, name=ar_stop.stop_name),
                    departure=utc_ts_to_iso(journey[-1].dp_ts),
                    departurePlatform=dp_stop.platform_code,
                    arrival=utc_ts_to_iso(journey[-1].ar_ts),
                    arrivalPlatform=ar_stop.platform_code,
                    stopovers=[],
                    distance=journey[-1].dist_traveled,
                    walking=True,
                    mode='walking',
                    line=FPTFLine.walking(),
                )
            )
        else:
            line = FPTFLine.from_route(routes[journey[-1].trip_id])
            dp_stop = stop_steffen.get_stop(stop_id=dp_stop_id)
            ar_stop = stop_steffen.get_stop(
                stop_id=journey[-1].ar_platform_id,
            )
            legs.append(
                FPTFLeg(
                    origin=FPTFStop(id=dp_stop.stop_name, name=dp_stop.stop_name),
                    destination=FPTFStop(id=ar_stop.stop_name, name=ar_stop.stop_name),
                    departure=utc_ts_to_iso(dp_ts),
                    departurePlatform=dp_stop.platform_code,
                    arrival=utc_ts_to_iso(journey[-1].ar_ts),
                    arrivalPlatform=ar_stop.platform_code,
                    stopovers=stopovers,
                    distance=dist_traveled + journey[-1].dist_traveled,
                    line=line,
                )
            )

        return FPTFJourney(legs=legs)


@dataclass
class FPTFJourneyAndAlternatives:
    journey: FPTFJourney
    alternatives: list[FPTFJourney]
