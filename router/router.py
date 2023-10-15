import networkx as nx
from datetime import datetime
import queue
from dataclasses import dataclass, field
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from gtfs.calendar_dates import CalendarDates
import sqlalchemy as sa
from database.engine import sessionfactory

# Note: Every timestamp must be in UTC.


def get_full_trip(trip_id: int):
    pass


def get_next_departures(station: int, timestamp: datetime, session):
    stops = sa.select(StopTimes, Trips, CalendarDates).join(
        Trips, StopTimes.trip_id == Trips.trip_id
    ).join(CalendarDates, Trips.service_id == CalendarDates.service_id).where(
        StopTimes.stop_id == station,
        # StopTimes.departure_time >= timestamp,
        CalendarDates.date == timestamp.date(),
    ).order_by(
        StopTimes.departure_time
    )

    return session.execute(stops).fetchall()


@dataclass(order=True)
class QueueItem:
    timestamp: datetime
    station: int = field(compare=False)
    trip_id: int = field(compare=False)


def get_routes(
    source: int,
    target: int,
) -> nx.DiGraph():
    """Get possible routes from source to target station.

    Parameters
    ----------
    source : int
        Eva number of the source station.
    target : int
        Eva number of the target station.
    """
    routes = nx.DiGraph()
    to_process = queue.PriorityQueue()


if __name__ == "__main__":
    engine, Session = sessionfactory()
    with Session() as session:
        print(get_next_departures(8000001, datetime(2022, 1, 1, 12, 0, 0), session))
