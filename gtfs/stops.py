import enum

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger

from database.base import Base


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
