import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import request, jsonify, current_app, Blueprint, make_response
from flask.helpers import send_file

from datetime import datetime
import numpy as np

from webserver.connection import (
    get_journeys,
    from_utc
)
from webserver import predictor, streckennetz, per_station_time
from webserver.db_logger import log_activity
from data_analysis import data_stats
from config import CACHE_PATH

bp = Blueprint("api", __name__, url_prefix="/api")
bp_limited = Blueprint("api_rate_limited", __name__, url_prefix='/api')


def analysis(connection: dict):
    """
    Analyses/evaluates/rates a given connection using machine learning

    Parameters
    ----------
    connection : dict
        The connection to analyze

    Returns
    -------
    dict
        The connection with the evaluation/rating
    """
    prediction = {}

    ar_data, dp_data = predictor.get_pred_data(connection)
    ar_prediction = predictor.predict_ar(ar_data)
    dp_prediction = predictor.predict_dp(dp_data)
    transfer_time = np.array(
        [segment['transfer_time'] for segment in connection[:-1]]
    )
    con_scores = predictor.predict_con(
        ar_prediction[:-1], dp_prediction[1:], transfer_time
    )

    prediction['connection_score'] = int(round(con_scores.prod() * 100))
    prediction['transfer_score'] = list(map(lambda s: int(round(s)), con_scores * 100))
    prediction['ar_predictions'] = ar_prediction[:, 0].tolist()
    prediction['dp_predictions'] = dp_prediction[:, 1].tolist()
    prediction['transfer_times'] = transfer_time.tolist()

    return prediction


@bp.route("/connect", methods=["GET"])
@log_activity
def connect():
    """
    Gets called when the website is loaded
    And gets some data from and about the user
    It returns the trainstations for the autofill forms

    Returns
    -------
    flask generated json
        list: a list of strings with all the known train stations
    """
    resp = jsonify({"stations": streckennetz.sta_list})
    return resp


@bp_limited.route("/stations.json")
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
    return r


@bp.route("/trip", methods=["POST"])
@log_activity
def trip():
    """
    Gets a connection from `startbhf` to `zielbhf` at a given date `date`
    using marudors HAFAS api. And rates the connection

    Parameters
    ----------
    start : str
        (from request) the station name where to start
    destination : str
        (from request) the station name of the destination
    date  : str
        (from request) the date and time at which the trip should take place in format `%d.%m.%Y %H:%M`

    Returns
    -------
    flask generated json
        All the possible connections
    """
    start = request.json['start']
    destination = request.json['destination']
    date = datetime.strptime(request.json['date'], "%d.%m.%Y %H:%M")

    # optional:
    search_for_departure = request.json['search_for_departure'] if 'search_for_departure' in request.json else True
    only_regional = request.json['only_regional'] if 'only_regional' in request.json else False
    bike = request.json['bike'] if 'bike' in request.json else False

    current_app.logger.info(
        "Getting connections from " + start + " to " + destination + ", " + str(date)
    )
    journeys, prediction_data = get_journeys(
        start=start,
        destination=destination,
        date=date,
        search_for_departure=search_for_departure,
        only_regional=only_regional,
        bike=bike,
    )

    for i, prediction in enumerate(map(analysis, prediction_data)):
        journeys[i]['connectionScore'] = prediction['connection_score']

        # This id is used for vue.js to render the list of connections
        journeys[i]['id'] = i
        journeys[i]['departure'] = journeys[i]['legs'][0]['departure']
        journeys[i]['plannedDeparture'] = journeys[i]['legs'][0]['plannedDeparture']
        journeys[i]['arrival'] = journeys[i]['legs'][-1]['arrival']
        journeys[i]['plannedArrival'] = journeys[i]['legs'][-1]['plannedArrival']
        journeys[i]['duration'] = (
            from_utc(journeys[i]['arrival'])
            - from_utc(journeys[i]['departure'])
        ).total_seconds()
        journeys[i]['plannedDuration'] = (
            from_utc(journeys[i]['plannedArrival'])
            - from_utc(journeys[i]['plannedDeparture'])
        ).total_seconds()
        journeys[i]['price'] = journeys[i]['price']['amount'] if journeys[i]['price'] is not None else -1


        walking_legs = 0
        train_categories = set()
        for leg_index, leg in enumerate(journeys[i]['legs']):
            if 'walking' in leg and leg['walking'] == True:
                walking_legs += 1
                train_categories.add('Fußweg')
                continue
            # The last leg has no transfer and thus no transferScore
            if leg_index != len(journeys[i]['legs']) - 1:
                journeys[i]['legs'][leg_index]['transferScore'] = prediction['transfer_score'][leg_index - walking_legs]
                journeys[i]['legs'][leg_index]['transferTime'] = prediction['transfer_times'][leg_index - walking_legs]
            journeys[i]['legs'][leg_index]['arrivalPrediction'] = prediction['ar_predictions'][leg_index - walking_legs]
            journeys[i]['legs'][leg_index]['departurePrediction'] = prediction['dp_predictions'][leg_index - walking_legs]
            train_categories.add(leg['line']['productName'])

        journeys[i]['trainCategories'] = sorted(list(train_categories))
        journeys[i]['transfers'] = len(journeys[i]['legs']) - 1 - walking_legs

    resp = jsonify(journeys)
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp


@bp.route("/stats")
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


@bp.route("/stationplot/<string:date_range>.webp")
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
        path_to_plot = per_station_time.generate_default(title=date_range)
    else:
        date_range = date_range.split("-")
        path_to_plot = per_station_time.generate_plot(
            datetime.strptime(date_range[0], "%d.%m.%Y"),
            datetime.strptime(date_range[1], "%d.%m.%Y"),
            use_cached_images=True,
        )

    current_app.logger.info(f"Returning plot: {path_to_plot}")
    # For some fucking reason flask searches the file from inside webserver
    # so we have to go back a bit even though
    # os.path.isfile('cache/plot_cache/'+ plot_name + '.png') works
    return send_file(path_to_plot, mimetype="image/webp")


@bp.route("/stationplot/limits")
@log_activity
def limits():
    """
    Returns the current datetime limits between which we can generate plots.

    Returns
    -------
    {
        "min": <min_date>,
        "max": <max_date>
    }
    """
    limits = per_station_time.limits()
    limits['min'] = limits['min'].date().isoformat()
    limits['max'] = limits['max'].date().isoformat()
    return limits


@bp.route("/obstacleplot/<string:date_range>.png")
@log_activity
def obstacle_plot(date_range):
    """
    Generates a plot that visualizes all the delays
    between the two dates specified in the url.

    Parameters
    ----------
    date_range : string
        The date range to generate the plot of in format `%d.%m.%Y, %H:%M-%d.%m.%Y, %H:%M`

    Returns
    -------
    flask generated image/png
        The generated plot
    """

    if date_range in per_station_time.DEFAULT_PLOTS:
        plot_name = date_range
    else:
        date_range = date_range.split("-")
        plot_name = per_station_time.generate_plot(
            datetime.strptime(date_range[0], "%d.%m.%Y %H:%M"),
            datetime.strptime(date_range[1], "%d.%m.%Y %H:%M"),
            use_cached_images=True,
        )

    current_app.logger.info(f"Returning plot: cache/plot_cache/{plot_name}.png")
    # For some fucking reason flask searches the file from inside webserver
    # so we have to go back a bit even though
    # os.path.isfile('cache/plot_cache/'+ plot_name + '.png') works
    return send_file(f"{CACHE_PATH}/plot_cache/{plot_name}.png", mimetype="image/png")