from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from gtfs.base import Base

import enum


class Trips(Base):
    __tablename__ = 'trips'

    trip_id: Mapped[int] = mapped_column(primary_key=True)
    route_id: Mapped[int] = mapped_column(ForeignKey('routes.route_id'))
    service_id: Mapped[int] = mapped_column(ForeignKey('calendar_dates.service_id'))
    shape_id: Mapped[int] = mapped_column(ForeignKey('shapes.shape_id'))