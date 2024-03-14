import os

# Do not use GUI for matplotlib
import matplotlib
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from data_analysis.per_station import PerStationOverTime
from helpers.logger import logging
from helpers.StreckennetzSteffi import StreckennetzSteffi
from ml_models.predictor import Predictor
from router.router_csa import RouterCSA
from webserver.db_logger import db

matplotlib.use('Agg')

logging.info('Initialising streckennetz')
streckennetz = StreckennetzSteffi(prefer_cache=True)
logging.info('Done!')

logging.info('Initialising per_station_time')
per_station_time = PerStationOverTime(prefer_cache=True)
logging.info('Done!')

logging.info('Initialising predictor')
predictor = Predictor(n_models=15)
logging.info('Done!')

logging.info('Initialising router')
router = RouterCSA()
logging.info('Done!')


def create_app():
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    # Create app with changed paths https://stackoverflow.com/a/42791810
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder='website/dist',
        static_folder='website/dist',
        static_url_path='',
    )

    from webserverconfig import DevelopmentConfig, ProductionConfig

    if app.config['DEBUG']:
        app.config.from_object(DevelopmentConfig)
    else:
        app.config.from_object(ProductionConfig)

    app.logger.info('DB init...')
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.logger.info('Done')

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.logger.info('Initializing the api...')
    from webserver import api

    app.register_blueprint(api.bp)
    app.register_blueprint(api.bp_limited)
    app.logger.info('Done')

    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    limiter.limit('2 per minute;60 per day')(api.bp_limited)

    app.logger.info(
        '\nSetup done, webserver is up and running!\
        \n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'
    )

    return app
