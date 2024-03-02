import enum
from collections import namedtuple
from typing import Dict, List

import geopy.distance
import sqlalchemy
from shapely import STRtree
from shapely.geometry import Point
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger
from tqdm import tqdm

from database.base import Base, create_all
from database.engine import get_engine, sessionfactory
from database.upsert import upsert_with_retry
from gtfs.stops import StopSteffen


class TransferType(enum.Enum):
    """
    GTFS transfer_type enum.
    See https://gtfs.org/schedule/reference/#transferstxt
    """

    RECOMMENDED = 0
    TIMED = 1
    MINIMUM_TIME = 2
    NO_TRANSFER = 3


Transfer = namedtuple('Transfer', ['from_stop', 'to_stop', 'duration', 'distance'])


class Transfers(Base):
    __tablename__ = 'gtfs_transfers'

    from_stop_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    to_stop_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transfer_type: Mapped[TransferType]
    min_transfer_time: Mapped[int]
    distance: Mapped[int]

    def __repr__(self) -> str:  #
        return f'<Transfer from {self.from_stop_id} to {self.to_stop_id} with type {self.transfer_type.name} and min transfer time {self.min_transfer_time}>'

    def as_dict(self) -> dict:
        return {
            'from_stop_id': self.from_stop_id,
            'to_stop_id': self.to_stop_id,
            'transfer_type': self.transfer_type.name,
            'min_transfer_time': self.min_transfer_time,
            'distance': self.distance,
        }

    def as_tuple(self) -> tuple:
        return (
            self.from_stop_id,
            self.to_stop_id,
            self.transfer_type.name,
            self.min_transfer_time,
            self.distance,
        )


def get_transfers() -> Dict[int, List[Transfer]]:
    engine, Session = sessionfactory()

    with Session() as session:
        result = session.scalars(sqlalchemy.select(Transfers)).all()

    transfers: Dict[int, List[Transfer]] = {}
    for row in result:
        row: Transfers
        transfer = Transfer(
            from_stop=row.from_stop_id,
            to_stop=row.to_stop_id,
            duration=row.min_transfer_time,
            distance=row.distance,
        )
        if row.from_stop_id in transfers:
            transfers[row.from_stop_id].append(transfer)
        else:
            transfers[row.from_stop_id] = [transfer]

    return transfers


WALKING_SPEED_M_S = 1.0
MAX_WALKING_TIME_S = 30 * 60
MAX_WALKING_DISTANCE_M = WALKING_SPEED_M_S * MAX_WALKING_TIME_S
MAX_DEGREE_DISTANCE_SEARCH_SPACE = 0.5


def calculate_transfers_from_stops(stop_dicts):
    s_index = STRtree(
        [Point(stop['stop_lon'], stop['stop_lat']) for stop in stop_dicts]
    )
    for stop1 in tqdm(stop_dicts):
        bbox = Point(stop1['stop_lon'], stop1['stop_lat']).buffer(
            MAX_DEGREE_DISTANCE_SEARCH_SPACE
        )
        nearby_stops_indices = s_index.query(bbox)
        for i in nearby_stops_indices:
            stop2 = stop_dicts[i]
            if stop1['stop_id'] == stop2['stop_id']:
                continue
            distance = geopy.distance.great_circle(
                (stop1['stop_lat'], stop1['stop_lon']),
                (stop2['stop_lat'], stop2['stop_lon']),
            ).meters
            if distance < MAX_WALKING_DISTANCE_M:
                yield Transfers(
                    from_stop_id=stop1['stop_id'],
                    to_stop_id=stop2['stop_id'],
                    transfer_type=TransferType.RECOMMENDED,
                    min_transfer_time=distance / WALKING_SPEED_M_S,
                    distance=distance,
                ).as_tuple()


def calculate_transfers():
    engine = get_engine()

    # Drop and recreate
    Transfers.__table__.drop(engine, checkfirst=True)

    create_all(engine)

    stop_steffen = StopSteffen()

    stop_dicts = [stop.as_dict() for stop in stop_steffen.stations()]
    transfers = list(calculate_transfers_from_stops(stop_dicts))

    upsert_with_retry(engine, Transfers.__table__, transfers)


if __name__ == '__main__':
    calculate_transfers()
