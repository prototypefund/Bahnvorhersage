from database.base import Base
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column, Session
from sqlalchemy.types import BigInteger, JSON
from typing import List, Dict
import json
from rtd_crawler.hash64 import xxhash64


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
        self.time_crawled = datetime.now(timezone.utc)
        self.change = change_str

    def as_dict(self):
        return {
            'hash_id': self.hash_id,
            'change_hash': self.change_hash,
            'time_crawled': self.time_crawled,
            'change': self.change,
        }
    
    def as_tuple(self):
        return (
            self.hash_id,
            self.change_hash,
            self.time_crawled,
            self.change,
        )
    
    @staticmethod
    def add_changes(session: Session, changes: List[Dict]):
        for change in changes:
            session.merge(UniqueChange(change))
        session.commit()