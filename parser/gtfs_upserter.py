from database.engine import sessionfactory
from typing import Dict, Tuple
import concurrent.futures
from database.upsert_copy_from import upsert_copy_from, tuples_to_csv
import sqlalchemy
import sqlalchemy.orm
from database.upsert import upsert_with_retry
from gtfs.stops import Stops
from gtfs.agency import Agency
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes, StopTimesTemp
from gtfs.trips import Trips, TripTemp


def clear_temp_tables(session: sqlalchemy.orm.Session):
    session.execute(sqlalchemy.text(f"TRUNCATE {StopTimesTemp.__table__.fullname}"))
    session.execute(sqlalchemy.text(f"TRUNCATE {TripTemp.__table__.fullname}"))
    session.commit()


class GTFSUpserter:
    def __init__(self):
        self.stops = {}
        self.agecies = {}
        self.calendar_dates = {}
        self.routes = {}
        self.stop_times = {}
        self.trips = {}
        self.sa_engine, Session = sessionfactory()

        with Session() as session:
            clear_temp_tables(session)

    def upsert(
        self,
        stops: Dict[int, Tuple],
        agencies: Dict[int, Tuple],
        calendar_dates: Dict[int, Tuple],
        routes: Dict[int, Tuple],
        stop_times: Dict[int, Tuple],
        trips: Dict[int, Tuple],
    ):
        self.stops.update(stops)
        self.agecies.update(agencies)
        self.calendar_dates.update(calendar_dates)
        self.routes.update(routes)
        self.stop_times.update(stop_times)
        self.trips.update(trips)

        if len(self.stop_times) > 5_000_000:
            self.flush()

    def flush(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []

            futures.append(
                executor.submit(
                    upsert_with_retry,
                    self.sa_engine,
                    Stops.__table__,
                    list(self.stops.values()),
                )
            )
            futures.append(
                executor.submit(
                    upsert_with_retry,
                    self.sa_engine,
                    Agency.__table__,
                    list(self.agecies.values()),
                )
            )
            futures.append(
                executor.submit(
                    upsert_with_retry,
                    self.sa_engine,
                    CalendarDates.__table__,
                    list(self.calendar_dates.values()),
                )
            )

            futures.append(
                executor.submit(
                    upsert_with_retry,
                    self.sa_engine,
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
                    self.sa_engine,
                )
            )
            futures.append(
                executor.submit(
                    upsert_copy_from,
                    Trips.__table__,
                    TripTemp.__table__,
                    tuples_to_csv(list(self.trips.values())),
                    self.sa_engine,
                )
            )

            for future in concurrent.futures.as_completed(futures):
                future.result()
            self.sa_engine.dispose()

        self.stops = {}
        self.agecies = {}
        self.calendar_dates = {}
        self.routes = {}
        self.stop_times = {}
        self.trips = {}
