import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import Column, Integer, DateTime, String, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from database import get_engine
from datetime import datetime
from helpers import ttl_lru_cache


Base = declarative_base()


class MlStat(Base):
    """
    Scheme for ml stats.
    """

    __tablename__ = 'ml_stat'
    id = Column(Integer, primary_key=True)
    minute = Column(Integer)
    ar_or_dp = Column(String(length=2))
    date = Column(DateTime)
    baseline = Column(Float)
    accuracy = Column(Float)
    improvement = Column(Float)

    def asdict(self) -> dict[str, str | int | float | datetime]:
        return {
            'minute': self.minute,
            'ar_or_dp': self.ar_or_dp,
            'date': self.date,
            'baseline': self.baseline,
            'accuracy': self.accuracy,
            'improvement': self.improvement,
        }

    @staticmethod
    def add_stats(stats: list[dict[str, str | int | float]]):
        date = datetime.now()
        with Session(get_engine()) as session:
            for stat in stats:
                session.add(
                    MlStat(
                        minute=stat['minute'],
                        ar_or_dp=stat['ar_or_dp'],
                        date=date,
                        baseline=stat['baseline'],
                        accuracy=stat['accuracy'],
                        improvement=stat['improvement'],
                    )
                )
            session.commit()

    @staticmethod
    def get_most_recent_stats() -> list[dict[str, str | int | float | datetime]]:
        """Get the most recent stats from the database. These should
        describe the ml models from yesterday.

        Returns
        -------
        list[dict[str, str | int | float | datetime]]
            The stats as list of dicts.
        """
        with Session(get_engine()) as session:
            result = (
                session.query(MlStat)
                .filter_by(date=session.query(func.max(MlStat.date)).scalar_subquery())
                .all()
            )
        return [stat.asdict() for stat in result]

    @ttl_lru_cache(seconds_to_live=60 * 60)
    @staticmethod
    def load_stats() -> list[dict[str, str | int | float | datetime]]:
        """Same as `get_most_recent_stats` but with time sensitive cache.

        See Also
        --------
        get_most_recent_stats
        """
        return MlStat.get_most_recent_stats()


if __name__ == '__main__':
    engine = get_engine()
    Base.metadata.create_all(engine)
    engine.dispose()

    MlStat()
