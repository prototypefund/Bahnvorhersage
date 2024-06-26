from database.engine import DB_CONNECT_STRING


class Config:
    # Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///tcp.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    pass


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = DB_CONNECT_STRING
