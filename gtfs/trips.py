from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from gtfs.base import Base

class Trips(Base):
    __tablename__ = 'gtfs_trips'

    trip_id: Mapped[int] = mapped_column(primary_key=True)
    route_id: Mapped[int] = mapped_column(ForeignKey('gtfs_routes.route_id'))
    service_id: Mapped[int] = mapped_column(ForeignKey('gtfs_calendar_dates.service_id'))
    # shape_id: Mapped[int] = mapped_column(ForeignKey('gtfs_shapes.shape_id'))