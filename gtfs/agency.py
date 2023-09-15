from sqlalchemy.orm import Mapped, mapped_column

from gtfs.base import Base

class Agency(Base):
    __tablename__ = 'agency'

    agency_id: Mapped[str] = mapped_column(primary_key=True)
    agency_name: Mapped[str]
    agency_url: Mapped[str]
    agency_timezone: Mapped[str]

    def __repr__(self):
        return f'<Agency {self.agency_id} {self.agency_name}>'
