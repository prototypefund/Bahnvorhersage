import sqlalchemy
from sqlalchemy.orm import sessionmaker

from config import db_database, db_password, db_server, db_username

DB_CONNECT_STRING = (
    'postgresql+psycopg2://'
    + db_username
    + ':'
    + db_password
    + '@'
    + db_server
    + '/'
    + db_database
    + '?sslmode=require'
)


def get_engine(**kwargs):
    return sqlalchemy.create_engine(
        DB_CONNECT_STRING, pool_pre_ping=True, pool_recycle=3600, **kwargs
    )


def sessionfactory(**kwargs):
    engine = get_engine(**kwargs)
    Session = sessionmaker(bind=engine)
    return engine, Session
