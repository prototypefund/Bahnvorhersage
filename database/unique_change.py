from database.base import Base
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column, Session
from sqlalchemy.types import BigInteger, JSON
from typing import Dict
import json
from helpers.hash64 import xxhash64


class UniqueChange(Base):
    __tablename__ = 'unique_change'
    hash_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    change_hash: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    time_crawled: Mapped[datetime]
    change: Mapped[str] = mapped_column(JSON)

    def __init__(self, change: Dict) -> None:
        change_str = json.dumps(change, sort_keys=True)
        self.hash_id = xxhash64(change['id'])
        self.change_hash = xxhash64(change_str)
        self.time_crawled = datetime.now(timezone.utc)
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