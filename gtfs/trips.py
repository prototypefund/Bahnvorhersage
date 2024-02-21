from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger

from database.base import Base


class Trips(Base):
    __tablename__ = 'gtfs_trips'

    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    route_id: Mapped[int] = mapped_column(BigInteger)
    service_id: Mapped[int] = mapped_column(BigInteger)
    # shape_id: Mapped[int] = mapped_column(BigInteger)

    def __repr__(self):
        return f'<Trips {self.trip_id} {self.route_id} {self.service_id}>'

    def as_dict(self):
        return {
            'trip_id': self.trip_id,
            'route_id': self.route_id,
            'service_id': self.service_id,
        }

    def as_tuple(self):
        return (
            self.trip_id,
            self.route_id,
            self.service_id,
        )


class TripTemp(Base):
    __tablename__ = 'gtfs_trips_temp'

    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    route_id: Mapped[int] = mapped_column(BigInteger)
    service_id: Mapped[int] = mapped_column(BigInteger)
    # shape_id: Mapped[int] = mapped_column(BigInteger)
