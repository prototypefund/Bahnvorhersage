from datetime import date, timedelta
from itertools import pairwise
from typing import Dict, List, Tuple

import sqlalchemy
from sqlalchemy.orm import Session as SessionType
from tqdm import tqdm

from database.base import create_all
from database.engine import sessionfactory
from database.upsert_copy_from import tuples_to_csv, upsert_copy_from
from gtfs.calendar_dates import CalendarDates
from gtfs.connections import Connections, ConnectionsTemp
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes
from gtfs.stops import StopSteffen
from gtfs.trips import Trips


def date_range(start_date: date, end_date: date):
    for n in range((end_date - start_date).days):
        yield start_date + timedelta(days=n)


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


def to_csa_connections(service_date: date, stop_steffen: StopSteffen, engine, Session):
    with Session() as session:
        stop_times = _get_stop_times(service_date, session)

        trip_ids = set(stop_time.trip_id for stop_time in stop_times)
        routes = get_routes(trip_ids, session)

    connections = [
        Connections.create_tuple(
            dp_ts=dp.departure_time,
            ar_ts=ar.arrival_time,
            planned_dp_ts=dp.departure_time,
            planned_ar_ts=ar.arrival_time,
            dp_stop_id=stop_steffen.get_stop(dp.stop_id).parent_station,
            ar_stop_id=stop_steffen.get_stop(ar.stop_id).parent_station,
            trip_id=dp.trip_id,
            is_regio=routes[dp.trip_id].is_regional(),
            dist_traveled=int(ar.shape_dist_traveled - dp.shape_dist_traveled),
            dp_platform_id=dp.stop_id,
            ar_platform_id=ar.stop_id,
        )
        for dp, ar in pairwise(stop_times)
        if dp.trip_id == ar.trip_id
    ]

    upsert_copy_from(
        table=Connections.__table__,
        temp_table=ConnectionsTemp.__table__,
        csv=tuples_to_csv(connections),
        engine=engine,
    )


def parse_connections():
    engine, Session = sessionfactory()
    stop_steffen = StopSteffen()
    create_all(engine=engine)

    start_date = date(2024, 1, 1)
    end_date = date.today() + timedelta(days=1)

    for service_date in tqdm(
        date_range(start_date, end_date), total=(end_date - start_date).days
    ):
        to_csa_connections(service_date, stop_steffen, engine, Session)


if __name__ == "__main__":
    parse_connections()
