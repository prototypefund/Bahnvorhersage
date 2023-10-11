from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class Agency(Base):
    __tablename__ = 'gtfs_agency'

    agency_id: Mapped[str] = mapped_column(primary_key=True)
    agency_name: Mapped[str]
    agency_url: Mapped[str]
    agency_timezone: Mapped[str]

    def __repr__(self):
        return f'<Agency {self.agency_id} {self.agency_name}>'

    def as_dict(self):
        return {
            'agency_id': self.agency_id,
            'agency_name': self.agency_name,
            'agency_url': self.agency_url,
            'agency_timezone': self.agency_timezone,
        }

    def as_tuple(self):
        return (
            self.agency_id,
            self.agency_name,
            self.agency_url,
            self.agency_timezone,
        )
