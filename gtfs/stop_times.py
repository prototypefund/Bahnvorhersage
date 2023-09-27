from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey

from gtfs.base import Base


class StopTimes(Base):
    __tablename__ = 'gtfs_stop_times'

    trip_id: Mapped[int] = mapped_column(ForeignKey('gtfs_trips.trip_id'), primary_key=True)
    stop_id: Mapped[int] = mapped_column(ForeignKey('gtfs_stops.stop_id'), primary_key=True)
    stop_sequence: Mapped[int]
    arrival_time: Mapped[str]
    departure_time: Mapped[str]
    shape_dist_traveled: Mapped[float]