from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def create_all(engine):
    Base.metadata.create_all(engine)
