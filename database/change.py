import json

import sqlalchemy
from sqlalchemy import BIGINT, Column
from sqlalchemy.dialects.postgresql import JSON

from database.base import Base
from database.engine import get_engine
from database.upsert import upsert_base


class Change(Base):
    __tablename__ = 'change_rtd'
    hash_id = Column(BIGINT, primary_key=True, autoincrement=False)
    change = Column(JSON)

    def __init__(self) -> None:
        try:
            engine = get_engine()
            self.metadata.create_all(engine)
            engine.dispose()
        except sqlalchemy.exc.OperationalError:
            print(f'database.{self.__tablename__} running offline!')

    @staticmethod
    def upsert(session: sqlalchemy.orm.Session, rows: list[dict]):
        return upsert_base(session, Change.__table__, rows)

    @staticmethod
    def add_changes(session: sqlalchemy.orm.Session, changes: dict):
        new_changes = [
            {'hash_id': train_id, 'change': json.dumps(changes[train_id])}
            for train_id in changes
        ]
        Change.upsert(session, new_changes)

    @staticmethod
    def get_changes(
        session: sqlalchemy.orm.Session, hash_ids: list[int]
    ) -> dict[int, dict]:
        """
        Get changes that have a given hash_id

        Parameters
        ----------
        hash_ids: list
            A list of hash_ids to get the corresponding rows from the db.

        Returns
        -------
        Sqlalchemy query with the results.
        """
        changes = session.query(Change).filter(Change.hash_id.in_(hash_ids)).all()
        return {change.hash_id: json.loads(change.change) for change in changes}

    @staticmethod
    def count_entries(session: sqlalchemy.orm.Session) -> int:
        """
        Get the number of rows in db.

        Returns
        -------
        int
            Number of Rows.
        """
        return session.query(Change).count()


if __name__ == '__main__':
    pass
