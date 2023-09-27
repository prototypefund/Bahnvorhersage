from sqlalchemy.orm import Mapped, mapped_column

from gtfs.base import Base

import enum

from datetime import date


class ExceptionType(enum.Enum):
    ADDED = 1
    REMOVED = 2


class CalendarDates(Base):
    __tablename__ = 'gtfs_calendar_dates'

    service_id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date]
    exception_type: Mapped[ExceptionType]
