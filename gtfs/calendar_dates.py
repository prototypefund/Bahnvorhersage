import enum
from datetime import date

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import BigInteger

from database.base import Base


class ExceptionType(enum.Enum):
    ADDED = 1
    REMOVED = 2


class CalendarDates(Base):
    __tablename__ = 'gtfs_calendar_dates'

    service_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    date: Mapped[date]
    exception_type: Mapped[ExceptionType]

    def __repr__(self):
        return f'<CalendarDates {self.service_id} {self.date} {self.exception_type}>'

    def as_dict(self):
        return {
            'service_id': self.service_id,
            'date': self.date,
            'exception_type': self.exception_type,
        }

    def as_tuple(self):
        return (
            self.service_id,
            self.date,
            self.exception_type,
        )
