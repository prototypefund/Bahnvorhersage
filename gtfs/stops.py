from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from gtfs.base import Base

import enum


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
    __tablename__ = 'stops'

    stop_id: Mapped[int] = mapped_column(primary_key=True)
    stop_name: Mapped[str]
    stop_lat: Mapped[float]
    stop_lon: Mapped[float]
    location_type: Mapped[LocationType]
    parent_station: Mapped[int] = mapped_column(ForeignKey('stops.stop_id'))

    def __repr__(self) -> str:
        return f'<Stop {self.stop_id} {self.stop_name} at {self.stop_lat} lat and {self.stop_lon} lon>'
