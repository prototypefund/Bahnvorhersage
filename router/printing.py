from router.datatypes import Connection, Reachability
from typing import List
from datetime import datetime
from helpers.StationPhillip import StationPhillip
from itertools import pairwise

def human_readable_reachability(reachability: Reachability, stations: StationPhillip):
    dp_ts = datetime.fromtimestamp(reachability.dp_ts)
    ar_ts = datetime.fromtimestamp(reachability.ar_ts)
    duration_seconds = reachability.ar_ts - reachability.dp_ts
    transfers = reachability.transfers
    dist_traveled = reachability.dist_traveled

    last_dp_ts = datetime.fromtimestamp(reachability.last_dp_ts)

    last_station_name = stations.get_name(eva=reachability.last_stop_id)

    journey_str = f'{dp_ts.strftime("%H:%M")} - {ar_ts.strftime("%H:%M")} ({last_station_name}, {duration_seconds//60:n}min, {transfers}x, last_dp:{last_dp_ts.strftime("%H:%M")}, {dist_traveled // 1000:n}km)'
    return journey_str


def journey_to_str(
    journey: List[Connection], stations: StationPhillip
):
    dp_ts = datetime.fromtimestamp(journey[0].dp_ts)
    ar_ts = datetime.fromtimestamp(journey[-1].ar_ts)
    duration = journey[-1].ar_ts - journey[0].dp_ts
    dist_traveled = sum([c.dist_traveled for c in journey])
    is_regio = all([c.is_regio for c in journey])

    journey_str = f'{dp_ts.strftime("%H:%M")} - '
    journey_str += f'{ar_ts.strftime("%H:%M")} '
    journey_str += f'({duration//60:n}min, {dist_traveled//1000:n}km, r:{is_regio}): '
    journey_str += f'{stations.get_name(eva=journey[0].dp_stop_id)} '

    last_dp_ts = journey[0].dp_ts

    for c1, c2 in pairwise(journey):
        if c1.trip_id != c2.trip_id:
            duration = (c1.ar_ts - last_dp_ts) // 60
            journey_str += f'{datetime.fromtimestamp(last_dp_ts).strftime("%H:%M")} -> '
            if c1.is_regio:
                journey_str += f'🭃{duration:n}min {c1.trip_id}🭎 -> '
            else:
                journey_str += f'🭊🭂{duration:n}min {c1.trip_id}🭍🬿 -> '
            journey_str += f'{datetime.fromtimestamp(c1.ar_ts).strftime("%H:%M")} '
            journey_str += f'{stations.get_name(eva=c1.ar_stop_id)} '

            last_dp_ts = c2.dp_ts

    duration = (journey[-1].ar_ts - last_dp_ts) // 60
    journey_str += f'{datetime.fromtimestamp(last_dp_ts).strftime("%H:%M")} -> '
    if journey[-1].is_regio:
        journey_str += f'🭃{duration:n}min {journey[-1].trip_id}🭎 -> '
    else:
        journey_str += f'🭊🭂{duration:n}min {journey[-1].trip_id}🭍🬿 -> '
    journey_str += f'{datetime.fromtimestamp(journey[-1].ar_ts).strftime("%H:%M")} '
    journey_str += f'{stations.get_name(eva=journey[-1].ar_stop_id)} '

    return journey_str


def print_journeys(
    journeys: List[List[Connection]],
    stations: StationPhillip,
):
    for journey in journeys:
        print(journey_to_str(journey, stations))
