import enum

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger

from database.base import Base


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

    route_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    agency_id: Mapped[str]
    route_short_name: Mapped[str]
    route_long_name: Mapped[str]
    route_type: Mapped[RouteType]

    def __repr__(self):
        return f'<Routes {self.route_id} {self.route_short_name}>'

    def as_dict(self):
        return {
            'route_id': self.route_id,
            'agency_id': self.agency_id,
            'route_short_name': self.route_short_name,
            'route_long_name': self.route_long_name,
            'route_type': self.route_type,
        }

    def as_tuple(self):
        return (
            self.route_id,
            self.agency_id,
            self.route_short_name,
            self.route_long_name,
            self.route_type,
        )

    def is_regional(self):
        train_cat = self.route_short_name.split(' ')[0]
        return train_cat not in {
            'IC',
            'EC',
            'ECE',
            'ICE',
            'EN',
            'RJ',
            'RJX',
            'TGV',
            'FLX',
            'NJ',
        }
