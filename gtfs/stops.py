import enum
from typing import Dict, Generator, List, Tuple

import geopy.distance
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger
import sqlalchemy

from database.base import Base
from database.engine import sessionfactory


class LocationType(enum.Enum):
    """
    GTFS location_type enum.
    See https://developers.google.com/transit/gtfs/reference/#stopstxt
    """

    STOP = 0
    STATION = 1
    ENTRANCE = 2
    GENERIC_NODE = 3
    BOARDING_AREA = 4


class Stops(Base):
    __tablename__ = 'gtfs_stops'

    stop_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    stop_name: Mapped[str]
    stop_lat: Mapped[float]
    stop_lon: Mapped[float]
    location_type: Mapped[LocationType]
    parent_station: Mapped[int] = mapped_column(BigInteger, nullable=True)
    platform_code: Mapped[str] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return f'<Stop {self.stop_id} {self.stop_name} at {self.stop_lat} lat and {self.stop_lon} lon>'

    def as_dict(self) -> dict:
        return {
            'stop_id': self.stop_id,
            'stop_name': self.stop_name,
            'stop_lat': self.stop_lat,
            'stop_lon': self.stop_lon,
            'location_type': self.location_type.name,
            'parent_station': self.parent_station,
            'platform_code': self.platform_code,
        }

    def as_tuple(self) -> tuple:
        return (
            self.stop_id,
            self.stop_name,
            self.stop_lat,
            self.stop_lon,
            self.location_type.name,
            self.parent_station,
            self.platform_code,
        )


class StopSteffen:
    names_to_ids: Dict[str, List[int]]
    stops: Dict[int, Stops]

    def __init__(self) -> None:
        stops = self._get_stops()

        self.stops = {stop.stop_id: stop for stop in stops}

        self.names_to_ids = {}
        for stop in self.stations():
            if stop.stop_name not in self.names_to_ids:
                self.names_to_ids[stop.stop_name] = []
            self.names_to_ids[stop.stop_name].append(stop.stop_id)

    def _get_stops(self) -> List[Stops]:
        engine, Session = sessionfactory()
        with Session() as session:
            stops = session.scalars(sqlalchemy.select(Stops)).all()
        engine.dispose()
        return stops

    def stations(self) -> Generator[Stops, None, None]:
        for stop in self.stops.values():
            if stop.location_type == LocationType.STATION:
                yield stop

    def name_to_ids(self, name: str) -> List[int]:
        return self.names_to_ids[name]

    def get_stop(self, stop_id: int) -> Stops:
        return self.stops[stop_id]

    def get_name(self, stop_id: int) -> str:
        return self.get_stop(stop_id).stop_name

    def get_location(self, stop_id: int) -> Tuple[float, float]:
        stop = self.get_stop(stop_id)
        return stop.stop_lat, stop.stop_lon

    def get_distance(self, stop_id1: int, stop_id2: int) -> float:
        loc1 = self.get_location(stop_id1)
        loc2 = self.get_location(stop_id2)
        return geopy.distance.great_circle(loc1, loc2).meters
