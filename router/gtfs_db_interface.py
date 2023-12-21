from datetime import date, datetime
from typing import Dict, List

import sqlalchemy
from sortedcontainers import SortedKeyList

from database.engine import sessionfactory
from gtfs.calendar_dates import CalendarDates
from gtfs.routes import Routes
from gtfs.stop_times import StopTimes
from gtfs.trips import Trips
from helpers.StationPhillip import StationPhillip
from router.datatypes import JourneyIdentifier, MockStopTimes, Stop, Trip
from router.exceptions import TimeNotFetchedError
from router.heuristic import compute_heuristic


class GTFSInterface:
    def __init__(self, destination_id: int, stations: StationPhillip) -> None:
        self.destination_id = destination_id
        self.stations = stations

        self.trips_by_stop: Dict[int, SortedKeyList] = {}
        self.trips: Dict[int, Trip] = {}
        self.stops: Dict[int, Stop] = {}
        self.session = sessionfactory()[1]()

    def prefetch_for_date(self, service_date: date):
        # import pickle

        # self.trips_by_stop = pickle.load(open('trips_by_stop_1_9_22.cpickle', 'rb'))
        # self.trips = pickle.load(open('trips_1_9_22.cpickle', 'rb'))
        # return

        trips_today: Dict[int, List[StopTimes]] = {}

        stop_times = self._all_stop_times_for_date(service_date)
        for stop_time in stop_times:
            if stop_time.trip_id in trips_today:
                trips_today[stop_time.trip_id].append(stop_time)
            else:
                trips_today[stop_time.trip_id] = [stop_time]

            if stop_time.departure_time is None:
                continue

            if stop_time.stop_id in self.trips_by_stop:
                self.trips_by_stop[stop_time.stop_id].add(stop_time)
            else:
                self.trips_by_stop[stop_time.stop_id] = SortedKeyList(
                    [stop_time], key=lambda x: x.departure_time
                )

        routes = self._get_routes_from_db(list(trips_today.keys()))

        for route, trip_id in routes:
            self.trips[trip_id] = Trip(
                trip_id=trip_id,
                route=route,
                stop_sequence=list(
                    sorted(trips_today[trip_id], key=lambda x: x.stop_sequence)
                ),
            )

    def _all_stop_times_for_date(self, service_date: date) -> List[StopTimes]:
        stmt = (
            sqlalchemy.select(StopTimes)
            .join(Trips, StopTimes.trip_id == Trips.trip_id)
            .join(CalendarDates, Trips.service_id == CalendarDates.service_id)
            .where(
                CalendarDates.date == service_date,
            )
        )

        return self.session.scalars(stmt).all()

    def _get_routes_from_db(self, trip_ids: List[int]):
        stmt = (
            sqlalchemy.select(Routes, Trips.trip_id)
            .join(Trips, Routes.route_id == Trips.route_id)
            .where(Trips.trip_id.in_(trip_ids))
        )

        return self.session.execute(stmt).all()

    def _next_stop_times_from_cache(
        self, stop_id: int, timestamp: datetime
    ) -> List[StopTimes]:
        if stop_id in self.trips_by_stop:
            index = self.trips_by_stop[stop_id].bisect_left(MockStopTimes(timestamp))
            if index == len(self.trips_by_stop[stop_id]):
                raise TimeNotFetchedError()
            min_ts = self.trips_by_stop[stop_id][index].departure_time
            if min_ts < timestamp:
                raise Exception("Minimal timestamp smaller then query")

            stop_times = []
            for stop_time in self.trips_by_stop[stop_id][index:]:
                if stop_time.departure_time == min_ts:
                    stop_times.append(stop_time)
                else:
                    break
            return stop_times
        raise KeyError(f"No stop time for {stop_id} in cache")

    def _next_stop_times_from_db(
        self, stop_id: int, timestamp: datetime
    ) -> List[StopTimes]:
        stmt = (
            sqlalchemy.select(StopTimes)
            .join(Trips, StopTimes.trip_id == Trips.trip_id)
            .join(CalendarDates, Trips.service_id == CalendarDates.service_id)
            .where(
                StopTimes.stop_id == stop_id,
                StopTimes.departure_time != None,
                StopTimes.departure_time >= timestamp,
                CalendarDates.date == timestamp.date(),
            )
            .order_by(StopTimes.departure_time)
        )

        stop_times = self.session.scalars(stmt).all()
        self.trips_by_stop[stop_id] = SortedKeyList(
            stop_times, key=lambda x: x.departure_time
        )

        return self._next_stop_times_from_cache(stop_id, timestamp)

    def next_trips(self, stop_id: int, timestamp: datetime) -> List[int]:
        try:
            stop_times = self._next_stop_times_from_cache(stop_id, timestamp)
        # except TimeNotFetchedError:
        #     stop_times = self._next_stop_times_from_db(stop_id, timestamp)
        except KeyError:
            stop_times = self._next_stop_times_from_db(stop_id, timestamp)

        return [
            stop_time.trip_id
            for stop_time in sorted(stop_times, key=lambda s: s.trip_id)
        ]

    def _get_trip_from_db(self, trip_id: int) -> Trip:
        stmt = (
            sqlalchemy.select(StopTimes)
            .where(
                StopTimes.trip_id == trip_id,
            )
            .order_by(StopTimes.stop_sequence)
        )
        trip = self.session.scalars(stmt).all()

        route_stmt = sqlalchemy.select(Routes).where(
            Trips.route_id == Routes.route_id,
            Trips.trip_id == trip_id,
        )
        route = self.session.scalars(route_stmt).one()

        self.trips[trip_id] = Trip(trip_id=trip_id, route=route, stop_sequence=trip)

        return self.trips[trip_id]

    def get_trip(self, trip_id: int) -> Trip:
        if trip_id in self.trips:
            return self.trips[trip_id]
        else:
            return self._get_trip_from_db(trip_id)

    def get_stop_sequence(
        self, trip_id: int, start_stop_id: int, include_start: bool = False
    ) -> List[StopTimes]:
        if trip_id in self.trips:
            return self.trips[trip_id].stop_sequence_from(
                start_stop_id, include_start=include_start
            )
        else:
            return self._get_trip_from_db(trip_id).stop_sequence_from(
                start_stop_id, include_start=include_start
            )

    def get_stop_time(self, trip_id: int, stop_id: int) -> StopTimes:
        if trip_id in self.trips:
            return self.trips[trip_id].stop_time_at(stop_id)
        else:
            return self._get_trip_from_db(trip_id).stop_time_at(stop_id)

    def init_stop(self, stop_id: int) -> Stop:
        stop = Stop(
            stop_id=stop_id,
            name=self.stations.get_name(stop_id),
            heuristic=compute_heuristic(
                stations=self.stations,
                stop_id=stop_id,
                destination_id=self.destination_id,
            ),
        )
        self.stops[stop_id] = stop
        return stop

    def get_stop(self, stop_id: int) -> Stop:
        if stop_id in self.stops:
            return self.stops[stop_id]
        else:
            return self.init_stop(stop_id)

    # def add_pareto_connection(self, stop_id: int, journey: JourneyIdentifier) -> None:
    #     stop = self.get_stop(stop_id)
    #     # comparable_connections: List[JourneyIdentifier] = []
    #     # for con in stop.pareto_connections:
    #     #     if journey.fully_overlapped(con):
    #     #         comparable_connections.append(con)

    #     # if journey.is_regional:
    #     #     comparable_connections = list(
    #     #         filter(lambda x: x.is_regional, comparable_connections)
    #     #     )

    #     # # find connections with equal or more transfers and their shortest duration,
    #     # # this will be used as the duration of this journey.
    #     # # This might feel weird, but the pareto set has a little bit of play in it.
    #     # # Not doing this correction would lead to escalate the duration.
    #     # comparable_connections = list(
    #     #     filter(lambda x: x.transfers == journey.transfers, comparable_connections)
    #     # )
    #     # if len(comparable_connections):
    #     #     min_duration = min([con.duration for con in comparable_connections])
    #     #     if min_duration < journey.duration:
    #     #         journey.duration = min_duration

    #     stop.pareto_connections.append(journey)
    #     self.stops[stop_id] = stop
