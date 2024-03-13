from datetime import datetime

from flask import Blueprint, abort, current_app, jsonify, make_response, request
from flask.helpers import send_file

from data_analysis import data_stats
from router.exceptions import NoRouteFound, NoTimetableFound
from webserver import per_station_time, router, streckennetz
from webserver.connection import get_and_rate_journeys
from webserver.db_logger import db, log_activity

bp = Blueprint('api', __name__, url_prefix='/api')
bp_limited = Blueprint('api_rate_limited', __name__, url_prefix='/api')

CUSTOM_500_ERROR = 505


@bp.errorhandler(CUSTOM_500_ERROR)
def custom_error(e):
    return jsonify({'error': e.description}), 500


@bp.route('/station_list.json', methods=['GET'])
@log_activity
def get_station_list():
    """
    Gets called when the website is loaded
    And gets some data from and about the user
    It returns the trainstations for the autofill forms

    Returns
    -------
    flask generated json
        list: a list of strings with all the known train stations
    """
    resp = jsonify({'stations': streckennetz.sta_list})
    # Cache for 1 week
    resp.cache_control.max_age = 60 * 60 * 24 * 7
    return resp


@bp_limited.route('/stations.json')
@log_activity
def station_dump():
    """Return all station data as json

    Returns
    -------
    json
        All station data
    """
    # We don't use jsonify here, because we have to do the
    # conversion to str within pandas. Converting the stations
    # to a dict first results in a Datetime out of bounds error.
    r = make_response(
        streckennetz.stations.reset_index(drop=True).to_json(
            orient='records', indent=None, force_ascii=False
        )
    )
    r.mimetype = 'application/json'
    # Cache for 1 week
    r.cache_control.max_age = 60 * 60 * 24 * 7
    return r


@bp.route('/trip', methods=['POST'])
@log_activity
def trip():
    """
    Searches and rates connections from `start` to
    `destination` at a given `date`.

    Parameters
    ----------
    start : str
        (from request) the station name where to start
    destination : str
        (from request) the station name of the destination
    date  : str
        (from request) the date and time at which the trip
        should take place in format `%d.%m.%Y %H:%M`

    Returns
    -------
    flask generated json
        All the possible connections
    """
    start = request.json['start']
    destination = request.json['destination']
    date = datetime.strptime(request.json['date'], '%d.%m.%Y %H:%M')

    # optional:
    search_for_arrival = (
        request.json['search_for_arrival']
        if 'search_for_arrival' in request.json
        else False
    )
    only_regional = (
        request.json['only_regional'] if 'only_regional' in request.json else False
    )
    bike = request.json['bike'] if 'bike' in request.json else False

    current_app.logger.info(
        'Getting connections from ' + start + ' to ' + destination + ', ' + str(date)
    )

    journeys = get_and_rate_journeys(
        start, destination, date, search_for_arrival, only_regional, bike
    )
    resp = jsonify(journeys)
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp


@bp_limited.route('/journeys', methods=['POST'])
@log_activity
def journeys():
    origin = request.json['origin']
    destination = request.json['destination']
    departure = datetime.fromisoformat(request.json['departure'])

    current_app.logger.info(f'Routing from {origin} to {destination} at {departure}')

    try:
        journeys = router.do_routing(origin, destination, departure, db.session)
    except NoTimetableFound as e:
        current_app.logger.error(f'No timetable found: {e}')
        abort(CUSTOM_500_ERROR, str(e))
    except NoRouteFound as e:
        current_app.logger.error(f'No route found: {e}')
        abort(CUSTOM_500_ERROR, str(e))

    resp = jsonify(journeys)
    # resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


@bp.route('/stats')
@log_activity
def stats():
    """
    Returns the stats stored in `{cache}/stats.json`
    Usually the stats of our train data

    Parameters
    ----------

    Returns
    -------
    flask generated json
        The stats
    """
    return data_stats.load_stats()


@bp.route('/stationplot/<string:date_range>.webp')
@log_activity
def station_plot(date_range):
    """
    Generates a plot that visualizes all the delays
    between the two dates specified in the url.

    Parameters
    ----------
    date_range : string
        The date range to generate the plot of in format `%d.%m.%Y, %H:%M-%d.%m.%Y, %H:%M`

    Returns
    -------
    image/webp
        The generated plot
    """

    if date_range in per_station_time.DEFAULT_PLOTS:
        path_to_plot = per_station_time.generate_default(plot_title=date_range)
    else:
        date_range = date_range.split('-')
        path_to_plot = per_station_time.generate_plot(
            datetime.strptime(date_range[0], '%d.%m.%Y'),
            datetime.strptime(date_range[1], '%d.%m.%Y'),
            use_cached_images=True,
        )

    current_app.logger.info(f'Returning plot: {path_to_plot}')
    # For some fucking reason flask searches the file from inside webserver
    # so we have to go back a bit even though
    # os.path.isfile('cache/plots/'+ plot_name + '.png') works
    return send_file(path_to_plot, mimetype='image/webp')


@bp.route('/stationplot/limits')
@log_activity
def limits():
    """
    Returns the current datetime limits between which we can generate plots.

    Returns
    -------
    {
        "min": <min_date>,
        "max": <max_date>,
        "freq": <frequency in hours>
    }
    """
    per_station_time.data_loader()
    limits = {
        'max': per_station_time.limits.max.date().isoformat(),
        'min': per_station_time.limits.min.date().isoformat(),
        'freq': per_station_time.limits.freq_hours,
    }
    return jsonify(limits)
