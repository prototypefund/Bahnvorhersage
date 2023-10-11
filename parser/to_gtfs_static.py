import concurrent.futures
import csv
import io
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import sqlalchemy
from redis import Redis
from tqdm import tqdm

from api.iris import TimetableStop
from config import redis_url
from database import (Change, PlanById, Rtd, sessionfactory, unparsed)
from database.upsert import upsert_with_retry
from gtfs.agency import Agency
from gtfs.calendar_dates import CalendarDates, ExceptionType
from gtfs.routes import Routes, RouteType
from gtfs.stop_times import StopTimes, StopTimesTemp
from gtfs.stops import LocationType, Stops
from gtfs.trips import Trips, TripTemp
from helpers.batcher import batcher
from helpers.StationPhillip import StationPhillip
from helpers.StreckennetzSteffi import StreckennetzSteffi
from rtd_crawler.hash64 import xxhash64
from database.base import create_all

""" Clear dangeling connections to db
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'tcp'
AND pid <> pg_backend_pid()
AND state in ('idle', 'idle in transaction', 'idle in transaction (aborted)', 'disabled');
"""

engine, Session = sessionfactory()
streckennetz = StreckennetzSteffi(prefer_cache=False)


def get_gtfs_stop_time(start_of_trip: datetime, stop_time: datetime) -> str:
    start_of_trip = start_of_trip.date()
    start_of_trip = datetime.combine(start_of_trip, datetime.min.time())

    total_seconds = (stop_time - start_of_trip).total_seconds()

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def all_stations_to_gtfs():
    stations = StationPhillip(prefer_cache=False)

    stops = [
        Stops(
            stop_id=0,
            stop_name='Unknown',
            stop_lat=0,
            stop_lon=0,
            location_type=LocationType.STATION,
            parent_station=None,
        ).as_dict()
    ]
    for eva in tqdm(stations.evas, desc='Adding stations to GTFS'):
        station = stations.stations_by_eva[eva]
        stop = Stops(
            stop_id=eva,
            stop_name=station.name,
            stop_lat=station.lat,
            stop_lon=station.lon,
            location_type=LocationType.STATION,
            parent_station=None,
        )
        stops.append(stop.as_dict())

    with Session() as session:
        upsert_base(session, Stops.__table__, stops)
        session.commit()


def stop_to_gtfs(
    stop_json: dict,
) -> Tuple[Agency, CalendarDates, Routes, StopTimes, Trips]:
    stop = TimetableStop(stop_json)
    line = stop.arrival.line if stop.arrival is not None else stop.depature.line
    route_short_name = f"{stop.trip_label.category} {line}"

    trip_id = xxhash64(str(stop.trip_id) + '_' + stop.date_id.isoformat())
    route_id = xxhash64(route_short_name)
    service_id = xxhash64(stop.date_id.date().isoformat())

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
        route_type=RouteType.BUS if stop.is_bus() else RouteType.RAIL,
    )

    try:
        stop_id = streckennetz.get_eva(name=stop.station_name)
    except KeyError:
        # TODO
        # warnings.warn(f'Could not find {stop.station_name} in station database. Using 0 as stop_id.')
        stop_id = 0

    stop_times = StopTimes(
        trip_id=trip_id,
        stop_id=stop_id,
        stop_sequence=stop.stop_id,
        arrival_time=get_gtfs_stop_time(stop.date_id, stop.arrival.planned_time)
        if stop.arrival is not None
        else None,
        departure_time=get_gtfs_stop_time(stop.date_id, stop.depature.planned_time)
        if stop.depature is not None
        else None,
        shape_dist_traveled=streckennetz.route_length(
            stop.arrival.planned_path, is_bus=stop.is_bus()
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

    return agency, calendar_dates, routes, stop_times, trips


def parse_batch(hash_ids: List[int], plans: Dict[int, Dict] = None):
    with Session() as session:
        if plans is None:
            plans = PlanById.get_stops(session, hash_ids)
        changes = Change.get_changes(session, hash_ids)
    parsed = []
    for hash_id in plans:
        parsed.append(parse_stop(hash_id, plans[hash_id], changes.get(hash_id, {})))

    if parsed:
        parsed = pd.DataFrame(parsed).set_index('hash_id')  # TODO
        Rtd.upsert(parsed, engine)  # TODO


def parse_unparsed(redis_client: Redis, last_stream_id: bytes) -> bytes:
    last_stream_id, unparsed_hash_ids = unparsed.get(redis_client, last_stream_id)
    if unparsed_hash_ids:
        print('parsing', len(unparsed_hash_ids), 'unparsed events')
        parse_batch(unparsed_hash_ids)
    return last_stream_id


def parse_unparsed_continues():
    redis_client = Redis.from_url(redis_url)
    last_stream_id = b'0-0'
    while True:
        try:
            last_stream_id = parse_unparsed(redis_client, last_stream_id)
        except Exception:
            traceback.print_exc(file=sys.stdout)
        time.sleep(60)


class psql_dialect(csv.Dialect):
    """Describe the usual properties of Unix-generated CSV files."""

    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL


def dicts_to_csv(dicts: List[Dict]) -> str:
    """Convert a dict to a csv string

    Parameters
    ----------
    dict_ : dict
        dict to convert to csv

    Returns
    -------
    str
        csv string
    """
    fbuf = io.StringIO()
    writer = csv.DictWriter(fbuf, fieldnames=dicts[0].keys(), dialect=psql_dialect)
    # writer.writeheader()
    writer.writerows(dicts)
    fbuf.seek(0)
    return fbuf.read()


def tuples_to_csv(tuples: List[tuple]) -> str:
    """Convert a list of tuples to a csv string

    Parameters
    ----------
    tuples : List[tuple]
        list of tuples to convert to csv

    Returns
    -------
    str
        csv string
    """
    fbuf = io.StringIO()
    writer = csv.writer(fbuf, dialect=psql_dialect)
    writer.writerows(tuples)
    fbuf.seek(0)
    return fbuf.read()


def clear_temp_tables():
    with Session() as session:
        session.execute(sqlalchemy.text(f"TRUNCATE {StopTimesTemp.__table__.fullname}"))
        session.execute(sqlalchemy.text(f"TRUNCATE {TripTemp.__table__.fullname}"))
        session.commit()


def upsert_copy_from(
    table: sqlalchemy.schema.Table, temp_table: sqlalchemy.schema.Table, csv: str
):
    # Use copy from to insert the date into a temporary table
    sql_cnxn = engine.raw_connection()
    cursor = sql_cnxn.cursor()

    fbuf = io.StringIO(csv)
    cursor.copy_from(fbuf, temp_table.fullname, sep=',', null='')
    sql_cnxn.commit()
    cursor.close()
    sql_cnxn.close()

    # INSTERT INTO table SELECT * FROM temp_table ON CONFLICT DO UPDATE
    # Insert the data from the temporary table into the real table using raw sql
    # update_cols = [c.name for c in table.c if c not in list(table.primary_key.columns)]

    sql_insert = f"""
        INSERT INTO {table.fullname}
        SELECT * FROM {temp_table.fullname}
        ON CONFLICT ({', '.join(table.primary_key.columns.keys())}) DO NOTHING
    """
    # SET {', '.join([f'{col} = excluded.{col}' for col in update_cols])}
    sql_clear = f"TRUNCATE {temp_table.fullname}"

    with Session() as session:
        session.execute(sqlalchemy.text(sql_insert))
        session.execute(sqlalchemy.text(sql_clear))
        session.commit()


def parse_chunk(chunk_limits: Tuple[int, int]):
    """Parse all stops with hash_id within the limits

    Parameters
    ----------
    chunk_limits : Tuple[int, int]
        min and max hash_id to parse in this chunk
    """
    with Session() as session:
        stops = PlanById.get_stops_from_chunk(session, chunk_limits)

    agencies = {}
    calendar_dates = {}
    routes = {}
    stop_times = {}
    trips = {}

    for hash_id in stops:
        agency, calendar_date, route, stop_time, trip = stop_to_gtfs(stops[hash_id])

        agencies[agency.agency_id] = agency.as_tuple()
        calendar_dates[calendar_date.service_id] = calendar_date.as_tuple()
        routes[route.route_id] = route.as_tuple()
        stop_times[stop_time.trip_id, stop_time.stop_id] = stop_time.as_tuple()
        trips[trip.trip_id] = trip.as_tuple()

    return (
        agencies,
        calendar_dates,
        routes,
        stop_times,
        trips,
    )


class GTFSUpserter:
    def __init__(self):
        self.agecies = {}
        self.calendar_dates = {}
        self.routes = {}
        self.stop_times = {}
        self.trips = {}

    def upsert(
        self,
        agencies: Dict[int, Tuple],
        calendar_dates: Dict[int, Tuple],
        routes: Dict[int, Tuple],
        stop_times: Dict[int, Tuple],
        trips: Dict[int, Tuple],
    ):
        self.agecies.update(agencies)
        self.calendar_dates.update(calendar_dates)
        self.routes.update(routes)
        self.stop_times.update(stop_times)
        self.trips.update(trips)

        if len(self.stop_times) > 1_000_000:
            self.flush()

    def flush(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            futures.append(
                executor.submit(
                    upsert_with_retry,
                    Session,
                    Agency.__table__,
                    list(self.agecies.values()),
                )
            )
            futures.append(
                executor.submit(
                    upsert_with_retry,
                    Session,
                    CalendarDates.__table__,
                    list(self.calendar_dates.values()),
                )
            )

            futures.append(
                executor.submit(
                    upsert_with_retry,
                    Session,
                    Routes.__table__,
                    list(self.routes.values()),
                )
            )

            futures.append(
                executor.submit(
                    upsert_copy_from,
                    StopTimes.__table__,
                    StopTimesTemp.__table__,
                    tuples_to_csv(list(self.stop_times.values())),
                )
            )
            futures.append(
                executor.submit(
                    upsert_copy_from,
                    Trips.__table__,
                    TripTemp.__table__,
                    tuples_to_csv(list(self.trips.values())),
                )
            )

            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.agecies = {}
        self.calendar_dates = {}
        self.routes = {}
        self.stop_times = {}
        self.trips = {}


def parse_all():
    """Parse all raw data there is"""

    # with Session() as session:
    #     chunk_limits = PlanById.get_chunk_limits(session)

    import pickle

    # pickle.dump(chunk_limits, open('chunk_limits.pickle', 'wb'))
    chunk_limits = pickle.load(open('chunk_limits.pickle', 'rb'))

    gtfs_upserter = GTFSUpserter()

    # # Non-concurrent code for debugging
    # for chunk in tqdm(chunk_limits, total=len(chunk_limits)):
    #     gtfs_upserter.upsert(*parse_chunk(chunk))

    n_processes = min(64, os.cpu_count())

    with tqdm(total=len(chunk_limits)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            n_processes, mp_context=mp.get_context('spawn')
        ) as executor:
            parser_tasks = {
                executor.submit(parse_chunk, chunk_limits.pop(0))
                for _ in range(min(n_processes * 2, len(chunk_limits)))
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


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    create_all(engine)
    clear_temp_tables()
    # all_stations_to_gtfs()
    parse_all()
