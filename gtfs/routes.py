from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey

from gtfs.base import Base

import enum


class RouteType(enum.Enum):
    TRAM = 0
    UNDERGROUND = 1
    RAIL = 2
    BUS = 3
    FERRY = 4
    CABLE_CAR = 5
    GONDOLA = 6
    FUNICULAR = 7
    TROLLEYBUS = 11
    MONORAIL = 12


class Routes(Base):
    __tablename__ = 'gtfs_routes'

    route_id: Mapped[int] = mapped_column(primary_key=True)
    agency_id: Mapped[str] = mapped_column(ForeignKey('gtfs_agency.agency_id'))
    route_short_name: Mapped[str]
    route_type: Mapped[RouteType]
