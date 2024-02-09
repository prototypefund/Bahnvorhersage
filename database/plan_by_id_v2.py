import json
from typing import Dict, List, Tuple

import numpy as np
import sqlalchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import BIGINT
from sqlalchemy.dialects.postgresql import JSON

from database.base import Base
from rtd_crawler.hash64 import xxhash64


class PlanByIdV2(Base):
    __tablename__ = 'plan_by_id_v2'
    hash_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    stop_id: Mapped[int]
    plan: Mapped[str] = mapped_column(JSON)

    def __init__(self, plan: Dict, stop_id: int) -> None:
        self.hash_id = xxhash64(plan['id'])
        self.stop_id = stop_id
        self.plan = json.dumps(plan, sort_keys=True)

    def as_dict(self):
        return {
            'hash_id': self.hash_id,
            'stop_id': self.stop_id,
            'plan': self.plan,
        }

    @staticmethod
    def get_stops(
        session: sqlalchemy.orm.Session, hash_ids: List[int]
    ) -> Dict[int, dict]:
        """
        Get stops that have a given hash_id

        Parameters
        ----------
        hash_ids: list
            A list of hash_ids to get the corresponding rows from the db.

        Returns
        -------
        Sqlalchemy query with the results.
        """
        stops = session.query(PlanByIdV2).filter(PlanByIdV2.hash_id.in_(hash_ids)).all()
        return {stop.hash_id: json.loads(stop.stop) for stop in stops}

    @staticmethod
    def count_entries(session: sqlalchemy.orm.Session) -> int:
        """
        Get the number of rows in db.

        Returns
        -------
        int
            Number of Rows.
        """
        return session.query(PlanByIdV2).count()

    @staticmethod
    def get_chunk_limits(session: sqlalchemy.orm.Session) -> List[Tuple[int, int]]:
        minimum = session.query(sqlalchemy.func.min(PlanByIdV2.hash_id)).scalar()
        maximum = session.query(sqlalchemy.func.max(PlanByIdV2.hash_id)).scalar()
        count = session.query(PlanByIdV2.hash_id).count()
        n_divisions = count // 20_000
        divisions = np.linspace(minimum, maximum, n_divisions, dtype=int)
        chunk_limits = [
            (int(divisions[i]), int(divisions[i + 1]))
            for i in range(len(divisions) - 1)
        ]
        return chunk_limits

    @staticmethod
    def get_stops_from_chunk(
        session: sqlalchemy.orm.Session, chunk_limits: Tuple[int, int]
    ) -> List['PlanByIdV2']:
        """
        Get stops that have a hash_id within chunk_limits

        Parameters
        ----------
        session: sqlalchemy.orm.Session
            The session to use for the query.

        chunk_limits: tuple
            A tuple with the lower and upper limit of the chunk.

        Returns
        -------
        Dict[int, dict]: A dictionary with the hash_id as key and the stop as value.
        """
        # IMPORTANT:
        # The filter should use ints and not floats. Ints makes the postgres planner
        # do a fast Index Scan compared to a slow Parallel Seq Scan
        stops = (
            session.query(PlanByIdV2)
            .filter(
                PlanByIdV2.hash_id >= int(chunk_limits[0]),
                PlanByIdV2.hash_id < int(chunk_limits[1]),
            )
            .all()
        )

        return stops
    
    @staticmethod
    def get_hash_ids_in_chunk_limits(
        session: sqlalchemy.orm.Session, chunk_limits: Tuple[int, int]
    ) -> List[int]:
        hash_ids = (
            session.query(PlanByIdV2.hash_id)
            .filter(
                PlanByIdV2.hash_id >= chunk_limits[0], PlanByIdV2.hash_id <= chunk_limits[1]
            )
            .all()
        )
        return [hash_id[0] for hash_id in hash_ids]


if __name__ == '__main__':
    from database.engine import sessionfactory

    engine, Session = sessionfactory()

    with Session() as session:
        PlanByIdV2.get_chunk_limits(session)
