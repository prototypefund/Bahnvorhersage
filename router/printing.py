from datetime import datetime
from itertools import pairwise
from typing import Dict, List

from gtfs.routes import Routes
from gtfs.stops import StopSteffen
from router.constants import WALKING_TRIP_ID
from router.datatypes import Connection, Reachability


def human_readable_reachability(reachability: Reachability, stop_steffen: StopSteffen):
    dp_ts = datetime.fromtimestamp(reachability.dp_ts) # TODO: probably not handling timezones correctly
    ar_ts = datetime.fromtimestamp(reachability.ar_ts)
    duration_seconds = reachability.ar_ts - reachability.dp_ts
    changeovers = reachability.changeovers
    dist_traveled = reachability.dist_traveled

    last_dp_ts = datetime.fromtimestamp(reachability.last_dp_ts)

    last_station_name = stop_steffen.get_name(reachability.last_stop_id)

    journey_str = f'{dp_ts.strftime("%H:%M")} - {ar_ts.strftime("%H:%M")} ({last_station_name}, {duration_seconds//60:n}min, {changeovers}x, last_dp:{last_dp_ts.strftime("%H:%M")}, {dist_traveled // 1000:n}km)'
    return journey_str


def journey_to_str(
    journey: List[Connection], stop_steffen: StopSteffen, routes: Dict[int, Routes]
):
    dp_ts = datetime.fromtimestamp(journey[0].dp_ts) # TODO: probably not handling timezones correctly
    ar_ts = datetime.fromtimestamp(journey[-1].ar_ts)
    duration = journey[-1].ar_ts - journey[0].dp_ts
    dist_traveled = sum([c.dist_traveled for c in journey])
    is_regio = all([c.is_regio for c in journey])

    journey_str = f'{dp_ts.strftime("%H:%M")} - '
    journey_str += f'{ar_ts.strftime("%H:%M")} '
    journey_str += f'({duration//60:n}min, {dist_traveled//1000:n}km, r:{is_regio}): '
    journey_str += f'{stop_steffen.get_name(journey[0].dp_stop_id)} '

    last_dp_ts = journey[0].dp_ts

    for c1, c2 in pairwise(journey):
        if c1.trip_id != c2.trip_id:
            duration = (c1.ar_ts - last_dp_ts) // 60
            journey_str += f'{datetime.fromtimestamp(last_dp_ts).strftime("%H:%M")} -> '
            if c1.trip_id == WALKING_TRIP_ID:
                journey_str += f'🚶{duration:n}min -> '
            elif c1.is_regio:
                journey_str += (
                    f'🭃{duration:n}min {routes[c1.trip_id].route_short_name}🭎 -> '
                )
            else:
                journey_str += (
                    f'🭊🭂{duration:n}min {routes[c1.trip_id].route_short_name}🭍🬿 -> '
                )
            journey_str += f'{datetime.fromtimestamp(c1.ar_ts).strftime("%H:%M")} '
            journey_str += f'{stop_steffen.get_name(c1.ar_stop_id)} '

            last_dp_ts = c2.dp_ts

    duration = (journey[-1].ar_ts - last_dp_ts) // 60
    journey_str += f'{datetime.fromtimestamp(last_dp_ts).strftime("%H:%M")} -> '
    if journey[-1].trip_id == WALKING_TRIP_ID:
        journey_str += f'🚶{duration:n}min -> '
    elif journey[-1].is_regio:
        journey_str += (
            f'🭃{duration:n}min {routes[journey[-1].trip_id].route_short_name}🭎 -> '
        )
    else:
        journey_str += (
            f'🭊🭂{duration:n}min {routes[journey[-1].trip_id].route_short_name}🭍🬿 -> '
        )
    journey_str += f'{datetime.fromtimestamp(journey[-1].ar_ts).strftime("%H:%M")} '
    journey_str += f'{stop_steffen.get_name(journey[-1].ar_stop_id)} '

    return journey_str


def print_journeys(
    journeys: List[List[Connection]],
    stop_steffen: StopSteffen,
    routes: Dict[int, Routes],
):
    for journey in journeys:
        print(journey_to_str(journey, stop_steffen, routes=routes))
