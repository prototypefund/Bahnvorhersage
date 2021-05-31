import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlalchemy
from sqlalchemy import Column, Text, DateTime
from sqlalchemy.dialects.postgresql import JSON, insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database import get_engine
import datetime

Base = declarative_base()


class Plan(Base):
    __tablename__ = 'plan_rtd'
    date = Column(DateTime, primary_key=True)
    bhf = Column(Text, primary_key=True)
    plan = Column(JSON)

    def __init__(self) -> None:
        try:
            engine = get_engine()
            Base.metadata.create_all(engine)
            engine.dispose()
        except sqlalchemy.exc.OperationalError:
            print('database.plan running offline!')
            
        self.engine = None
        self.session = None

        self.queue = []

    def __enter__(self):
        self.engine = get_engine()
        self.session = sessionmaker(bind=self.engine)()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        self.session.close()
        self.engine.dispose()

        self.engine = None
        self.session = None


    def upsert(self, rows, no_update_cols=[]):
        if self.session is None:
            raise ValueError('upsert only works within a with')
        table = Plan.__table__

        stmt = insert(table).values(rows)

        update_cols = [c.name for c in table.c
                       if c not in list(table.primary_key.columns)
                       and c.name not in no_update_cols]

        on_conflict_stmt = stmt.on_conflict_do_update(
            index_elements=table.primary_key.columns,
            set_={k: getattr(stmt.excluded, k) for k in update_cols}
        )

        self.session.execute(on_conflict_stmt)

    def add_plan(self, plan, bhf, date, hour):
        if self.session is None:
            raise ValueError('add only works within a with')
        date = datetime.datetime.combine(date, datetime.time(hour, 0))
        self.queue.append({'date': date, 'bhf': bhf, 'plan': plan})
        if len(self.queue) > 1000:
            self.commit()

    def commit(self):
        if self.session is None:
            raise ValueError('commit only works within a with')
        self.upsert(self.queue)
        self.queue = []
        self.session.commit()

    def plan_of_station(self, bhf: str, date1: datetime.datetime, date2: datetime.datetime):
        if self.session is None:
            raise ValueError('plan_of_station only works within a with')

        if date1 is None:
            return self.session.query(Plan).filter((Plan.bhf == bhf)).all()
        if date2 is None:
            return self.session.query(Plan).filter(
                (Plan.bhf == bhf) & (Plan.date == date1)).first()
        return self.session.query(Plan).filter((Plan.bhf == bhf)
                                               & (Plan.date >= date1)
                                               & (Plan.date < date2)).all()

    def count_entries_at_date(self, date: datetime.datetime) -> int:
        if self.session is None:
            raise ValueError('count_entries_at_date only works within a with')
        return self.session.query(Plan).filter(Plan.date == date).count()


if __name__ == '__main__':
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        engine.dispose()
    except sqlalchemy.exc.OperationalError:
        print('database.plan running offline!')