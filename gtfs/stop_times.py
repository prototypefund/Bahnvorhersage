from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger, String
from datetime import datetime

from database.base import Base


class StopTimes(Base):
    __tablename__ = 'gtfs_stop_times'

    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    stop_id: Mapped[int] = mapped_column(primary_key=True)
    stop_sequence: Mapped[int]
    arrival_time: Mapped[datetime] = mapped_column(nullable=True)
    departure_time: Mapped[datetime] = mapped_column(nullable=True)
    shape_dist_traveled: Mapped[float]

    def __repr__(self):
        return f'<StopTimes {self.trip_id} {self.stop_id} {self.arrival_time} {self.departure_time}>'

    def as_dict(self):
        return {
            'trip_id': self.trip_id,
            'stop_id': self.stop_id,
            'stop_sequence': self.stop_sequence,
            'arrival_time': self.arrival_time,
            'departure_time': self.departure_time,
            'shape_dist_traveled': self.shape_dist_traveled,
        }

    def as_tuple(self):
        return (
            self.trip_id,
            self.stop_id,
            self.stop_sequence,
            self.arrival_time,
            self.departure_time,
            self.shape_dist_traveled,
        )


class StopTimesTemp(Base):
    __tablename__ = 'gtfs_stop_times_temp'

    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    stop_id: Mapped[int] = mapped_column(primary_key=True)
    stop_sequence: Mapped[int]
    arrival_time: Mapped[datetime] = mapped_column(nullable=True)
    departure_time: Mapped[datetime] = mapped_column(nullable=True)
    shape_dist_traveled: Mapped[float]
