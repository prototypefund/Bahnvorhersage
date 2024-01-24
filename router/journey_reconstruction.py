from typing import List, Dict
from bisect import bisect_left
from itertools import pairwise

from router.datatypes import Reachability, Connection
from router.constants import NO_STOP_ID
from gtfs.routes import Routes

from datetime import datetime, UTC

def utc_ts_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()

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


def get_next_connection(
    connections: List[Connection],
    connection: Connection,
):
    start_index = bisect_left(connections, connection.ar_ts, key=lambda c: c.dp_ts)
    for c in connections[start_index:]:
        if c.trip_id == connection.trip_id:
            return c


def extract_journeys(
    stops: Dict[int, List[Reachability]],
    destination_stop_id: int,
    connections: List[Connection],
) -> List[List[Connection]]:
    reachability_chains = extract_reachability_chains(stops, destination_stop_id)

    journeys = []
    for reachability_chain in reachability_chains:
        sparse_journey: List[Connection] = []
        for reachability in reachability_chain:
            sparse_journey.append(
                match_connection_to_reachability(connections, reachability)
            )

        journey: List[Connection] = []
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


def remove_duplicate_journeys(journeys: List[List[Connection]]):
    journeys = sorted(journeys, key=lambda j: j[0].dp_ts)
    unique_journeys: List[List[Connection]] = []
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


def journey_to_fptf(journey: List[Connection], routes: Dict[int, Routes]):
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
            line = {
                'type': 'line',
                'id': c1.trip_id,
                'name': routes[c1.trip_id].route_short_name,
                'operator': routes[c1.trip_id].agency_id,
                'isRegio': bool(c1.is_regio),
            }
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
                    'line': line,
                }
            )
            dp_ts = c2.dp_ts
            dp_stop_id = c2.dp_stop_id
            dist_traveled = 0
            stopovers = []

    line = {
        'type': 'line',
        'id': journey[-1].trip_id,
        'name': routes[journey[-1].trip_id].route_short_name,
        'operator': routes[journey[-1].trip_id].agency_id,
        'isRegio': bool(journey[-1].is_regio),
    }
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
            'line': line,
        }
    )

    return fptf_journey

