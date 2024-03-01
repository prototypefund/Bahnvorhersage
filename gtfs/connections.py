import enum
from typing import List

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger, DateTime
import sqlalchemy
from datetime import datetime, UTC
from helpers.hash64 import xxhash64
from sqlalchemy.orm import Session as SessionType

from database.base import Base
from database.engine import sessionfactory
from router.datatypes import Connection as CSAConnectionTuple


class Connections(Base):
    __tablename__ = 'csa_connections'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dp_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ar_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_dp_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_ar_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dp_stop_id: Mapped[int] = mapped_column(BigInteger)
    ar_stop_id: Mapped[int] = mapped_column(BigInteger)
    trip_id: Mapped[int] = mapped_column(BigInteger)
    is_regio: Mapped[bool]
    dist_traveled: Mapped[int]
    dp_platform_id: Mapped[int] = mapped_column(BigInteger)
    ar_platform_id: Mapped[int] = mapped_column(BigInteger)

    @staticmethod
    def create_id(
        planned_dp_ts: datetime,
        planned_ar_ts: datetime,
        dp_stop_id: int,
        ar_stop_id: int,
        trip_id: int,
    ) -> int:
        return xxhash64(
            planned_dp_ts.isoformat()
            + planned_ar_ts.isoformat()
            + str(dp_stop_id)
            + str(ar_stop_id)
            + str(trip_id)
        )

    @staticmethod
    def create_tuple(
        dp_ts: datetime,
        ar_ts: datetime,
        planned_dp_ts: datetime,
        planned_ar_ts: datetime,
        dp_stop_id: int,
        ar_stop_id: int,
        trip_id: int,
        is_regio: bool,
        dist_traveled: int,
        dp_platform_id: int,
        ar_platform_id: int,
    ) -> tuple:
        return (
            Connections.create_id(
                planned_dp_ts, planned_ar_ts, dp_stop_id, ar_stop_id, trip_id
            ),
            dp_ts,
            ar_ts,
            planned_dp_ts,
            planned_ar_ts,
            dp_stop_id,
            ar_stop_id,
            trip_id,
            is_regio,
            dist_traveled,
            dp_platform_id,
            ar_platform_id,
        )

    def __repr__(self) -> str:
        return f'<Connection {self.dp_stop_id} {self.ar_stop_id} at {self.dp_ts} dp_ts and {self.ar_ts} ar_ts>'

    def as_tuple(self) -> tuple:
        return (
            self.id,
            self.dp_ts,
            self.ar_ts,
            self.planned_dp_ts,
            self.planned_ar_ts,
            self.dp_stop_id,
            self.ar_stop_id,
            self.trip_id,
            self.is_regio,
            self.dist_traveled,
            self.dp_platform_id,
            self.ar_platform_id,
        )

    @staticmethod
    def get_for_routing(
        session: SessionType, from_ts: datetime, to_ts: datetime
    ) -> List['Connections']:
        stmt = (
            sqlalchemy.select(Connections)
            .where(Connections.dp_ts >= from_ts)
            .where(Connections.ar_ts < to_ts)
            .order_by(Connections.dp_ts)
        )
        result = session.scalars(stmt).all()

        'dp_stop_id',
        'ar_stop_id',
        'trip_id',
        'is_regio',
        'dist_traveled',
        'dp_platform_id',
        'ar_platform_id',

        return [
            CSAConnectionTuple(
                dp_ts=int(r.dp_ts.timestamp()),
                ar_ts=int(r.ar_ts.timestamp()),
                dp_stop_id=r.dp_stop_id,
                ar_stop_id=r.ar_stop_id,
                trip_id=r.trip_id,
                is_regio=int(r.is_regio),
                dist_traveled=r.dist_traveled,
                dp_platform_id=r.dp_platform_id,
                ar_platform_id=r.ar_platform_id,
            )
            for r in result
        ]


class ConnectionsTemp(Base):
    __tablename__ = 'csa_connections_temp'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dp_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ar_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_dp_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    planned_ar_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dp_stop_id: Mapped[int] = mapped_column(BigInteger)
    ar_stop_id: Mapped[int] = mapped_column(BigInteger)
    trip_id: Mapped[int] = mapped_column(BigInteger)
    is_regio: Mapped[bool]
    dist_traveled: Mapped[int]
    dp_platform_id: Mapped[int] = mapped_column(BigInteger)
    ar_platform_id: Mapped[int] = mapped_column(BigInteger)

    @staticmethod
    def clear(session: SessionType):
        session.execute(
            sqlalchemy.text(f"TRUNCATE {ConnectionsTemp.__table__.fullname}")
        )
        session.commit()


if __name__ == '__main__':
    engine, Session = sessionfactory()
    with Session() as session:
        from_ts = datetime(2024, 1, 1, 0, 0, 0)
        to_ts = datetime(2024, 1, 2, 0, 0, 0)
        connections = Connections.get(session, from_ts, to_ts)
