import concurrent.futures
import json
import multiprocessing as mp
import os
from parser.gtfs_upserter import GTFSUpserter
from typing import List, Tuple

import sqlalchemy
from tqdm import tqdm

from api.iris import TimetableStop
from database.base import create_all
from database.engine import sessionfactory
from database.plan_by_id_v2 import PlanByIdV2
from gtfs.agency import Agency
from gtfs.calendar_dates import CalendarDates, ExceptionType
from gtfs.routes import Routes, RouteType
from gtfs.stop_times import StopTimes
from gtfs.stops import LocationType, Stops
from gtfs.trips import Trips
from helpers.StreckennetzSteffi import StreckennetzSteffi
from helpers.hash64 import xxhash64

""" Clear dangeling connections to db
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'tcp'
AND pid <> pg_backend_pid()
AND state in ('idle', 'idle in transaction', 'idle in transaction (aborted)', 'disabled');
"""

streckennetz = StreckennetzSteffi(prefer_cache=False)


def stop_to_gtfs(
    plan: PlanByIdV2,
) -> Tuple[Stops, Stops, Agency, CalendarDates, Routes, StopTimes, Trips]:
    stop = TimetableStop(json.loads(plan.plan))
    line = stop.arrival.line if stop.arrival is not None else stop.departure.line
    if line is None:
        line = stop.trip_label.number
    route_short_name = f"{stop.trip_label.category} {line}"
    route_long_name = f"{stop.trip_label.category} {stop.trip_label.number}"

    trip_id = xxhash64(str(stop.trip_id) + '_' + stop.date_id.isoformat())
    route_id = xxhash64(route_long_name)
    service_id = xxhash64(stop.date_id.date().isoformat())

    station_name = streckennetz.get_name(eva=plan.stop_id)
    platform_code = (
        stop.arrival.planned_platform
        if stop.arrival is not None
        else stop.departure.planned_platform
    )
    stop_id = xxhash64(str(plan.stop_id) + '_' + platform_code)

    station_coords = streckennetz.get_location(eva=plan.stop_id)

    station = Stops(
        stop_id=plan.stop_id,
        stop_name=station_name,
        stop_lat=station_coords[0],
        stop_lon=station_coords[1],
        location_type=LocationType.STATION,
        parent_station=None,
        platform_code=None,
    )

    platform = Stops(
        stop_id=stop_id,
        stop_name=station_name,
        stop_lat=station_coords[0],
        stop_lon=station_coords[1],
        location_type=LocationType.STOP,
        parent_station=plan.stop_id,
        platform_code=platform_code,
    )

    agency = Agency(
        agency_id=stop.trip_label.owner,
        agency_name=stop.trip_label.owner,
        agency_url='https://bahnvorhersage.de',
        agency_timezone='Europe/Berlin',
    )

    calendar_dates = CalendarDates(
        service_id=service_id,
        date=stop.date_id.date(),
        exception_type=ExceptionType.ADDED,
    )

    routes = Routes(
        route_id=route_id,
        agency_id=stop.trip_label.owner,
        route_short_name=route_short_name,
        route_long_name=route_long_name,
        route_type=RouteType.BUS if stop.is_bus() else RouteType.RAIL,
    )

    stop_times = StopTimes(
        trip_id=trip_id,
        stop_id=stop_id,
        stop_sequence=stop.stop_id,
        arrival_time=stop.arrival.planned_time  # get_gtfs_stop_time(stop.date_id, stop.arrival.planned_time)
        if stop.arrival is not None
        else None,
        departure_time=stop.departure.planned_time  # get_gtfs_stop_time(stop.date_id, stop.departure.planned_time)
        if stop.departure is not None
        else None,
        shape_dist_traveled=streckennetz.route_length(
            stop.arrival.planned_path + [station_name], is_bus=stop.is_bus()
        )
        if stop.arrival is not None
        else 0,
    )

    trips = Trips(
        trip_id=trip_id,
        route_id=route_id,
        service_id=service_id,
        # shape_id=...,
    )

    return station, platform, agency, calendar_dates, routes, stop_times, trips


def parse_chunk(chunk_limits: Tuple[int, int] = None, hash_ids: List[int] = None):
    engine, Session = sessionfactory(
        poolclass=sqlalchemy.pool.NullPool,
    )

    with Session() as session:
        if chunk_limits is not None:
            plans = PlanByIdV2.get_stops_from_chunk(session, chunk_limits)
        elif hash_ids is not None:
            plans = PlanByIdV2.get_stops_from_hash_ids(session, hash_ids)
        else:
            raise ValueError('Either chunk_limits or hash_ids must be given')

    stops = {}
    agencies = {}
    calendar_dates = {}
    routes = {}
    stop_times = {}
    trips = {}

    for plan in plans:
        station, platform, agency, calendar_date, route, stop_time, trip = stop_to_gtfs(
            plan
        )

        stops[station.stop_id] = station.as_tuple()
        stops[platform.stop_id] = platform.as_tuple()
        agencies[agency.agency_id] = agency.as_tuple()
        calendar_dates[calendar_date.service_id] = calendar_date.as_tuple()
        routes[route.route_id] = route.as_tuple()
        stop_times[stop_time.trip_id, stop_time.stop_id] = stop_time.as_tuple()
        trips[trip.trip_id] = trip.as_tuple()

    return (
        stops,
        agencies,
        calendar_dates,
        routes,
        stop_times,
        trips,
    )


def parse_all():
    """Parse all raw data there is"""

    engine, Session = sessionfactory(
        poolclass=sqlalchemy.pool.NullPool,
    )

    with Session() as session:
        chunk_limits = PlanByIdV2.get_chunk_limits(session)

    # import pickle

    # pickle.dump(chunk_limits, open('chunk_limits.pickle', 'wb'))
    # chunk_limits = pickle.load(open('chunk_limits.pickle', 'rb'))

    gtfs_upserter = GTFSUpserter()

    # # Non-concurrent code for debugging
    # for chunk in tqdm(chunk_limits, total=len(chunk_limits)):
    #     gtfs_upserter.upsert(*parse_chunk(chunk_limits=chunk))

    n_processes = min(40, os.cpu_count())

    with tqdm(total=len(chunk_limits)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            n_processes, mp_context=mp.get_context('spawn')
        ) as executor:
            parser_tasks = {
                executor.submit(parse_chunk, chunk_limits.pop(0))
                for _ in range(min(500, len(chunk_limits)))
            }

            while parser_tasks:
                done, parser_tasks = concurrent.futures.wait(
                    parser_tasks, return_when='FIRST_COMPLETED'
                )

                for future in done:
                    gtfs_upserter.upsert(*future.result())
                    if chunk_limits:
                        parser_tasks.add(
                            executor.submit(parse_chunk, chunk_limits.pop(0))
                        )
                    pbar.update()

        gtfs_upserter.flush()


def main():
    engine, Session = sessionfactory()
    create_all(engine)
    engine.dispose()

    parse_all()


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    main()
