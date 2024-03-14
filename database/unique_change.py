import json
from datetime import UTC, datetime

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import JSON, BigInteger

from database.base import Base
from helpers.hash64 import xxhash64


class UniqueChange(Base):
    __tablename__ = 'unique_change'
    hash_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    change_hash: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    time_crawled: Mapped[datetime]
    change: Mapped[str] = mapped_column(JSON)

    def __init__(self, change: dict) -> None:
        change_str = json.dumps(change, sort_keys=True)
        self.hash_id = xxhash64(change['id'])
        self.change_hash = xxhash64(change_str)
        self.time_crawled = datetime.now(UTC)
        self.change = change_str

    def as_dict(self):
        return {
            'hash_id': self.hash_id,
            'change_hash': self.change_hash,
            'time_crawled': self.time_crawled,
            'change': self.change,
        }

    @staticmethod
    def count_entries(session: Session) -> int:
        """
        Get the number of rows in db.

        Returns
        -------
        int
            Number of Rows.
        """
        return session.query(UniqueChange).count()
