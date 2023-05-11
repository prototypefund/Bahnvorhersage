import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import requests
from pytz import timezone

from database.ris_transfer_time import Connection
from helpers import ttl_lru_cache
from webserver import predictor, streckennetz
from webserver.transfer_times import (
    get_needed_transfer_times,
    shift_predictions_by_transfer_time,
)


@dataclass
class Prediction:
    connection_score: int
    transfer_score: list[int]
    ar_predictions: list[float]
    dp_predictions: list[float]
    transfer_times: list[int]
    needed_transfer_times: list[Connection]


def rate_journey(iris_journey: list[dict], fptf_journey: list[dict]) -> Prediction:
    """
    Analyses/evaluates/rates a given journey using machine learning

    Parameters
    ----------
    ...

    Returns
    -------
    Prediction
        The journey with the evaluation/rating
    """
    needed_transfer_times = list(get_needed_transfer_times(fptf_journey['legs']))

    ar_data, dp_data = predictor.get_pred_data(iris_journey, streckennetz)
    ar_prediction = predictor.predict_ar(ar_data)
    dp_prediction = predictor.predict_dp(dp_data)

    transfer_time = np.array(
        [segment['transfer_time'] for segment in iris_journey[:-1]]
    )

    ar_con_prediction, dp_con_prediction = shift_predictions_by_transfer_time(
        ar_prediction[:-1].copy(),
        dp_prediction[1:].copy(),
        transfer_time,
        needed_transfer_times,
    )

    con_scores = predictor.predict_con(ar_con_prediction, dp_con_prediction)

    return Prediction(
        connection_score=int(round(con_scores.prod() * 100)),
        transfer_score=list(map(lambda s: int(round(s)), con_scores * 100)),
        ar_predictions=ar_prediction[:, 0].tolist(),
        dp_predictions=dp_prediction[:, 1].tolist(),
        transfer_times=transfer_time.tolist(),
        needed_transfer_times=needed_transfer_times,
    )


def from_utc(utc_time: str) -> datetime.datetime:
    return (
        datetime.datetime.fromisoformat(utc_time)
        .astimezone(timezone("Europe/Berlin"))
        .replace(tzinfo=None)
    )


def get_journeys(
    start: str,
    destination: str,
    date: datetime.datetime,
    max_changes: int = -1,
    transfer_time: int = 0,
    search_for_departure: bool = True,
    only_regional: bool = False,
    bike: bool = False,
) -> tuple[list[dict], list[list[dict]]]:
    """Search for journeys using db-rest api and augment
    them with trip details for each leg of each journey

    Parameters
    ----------
    start : str
        start station name
    destination : str
        destination station name
    date : datetime.datetime
        date and time of departure
    max_changes : int, optional
        Maximum number of allowed changes, by default -1
    transfer_time : int, optional
        Minimal transfer time, by default 0
    search_for_departure : bool, optional
        False = time == arrival time, by default True
    only_regional : bool, optional
        True = only search for regional connections, by default False
    bike : bool, optional
        True = only search for bike connections, by default False

    Returns
    -------
        list : Parsed connections
    """
    request_data = {
        'from': streckennetz.get_eva(name=start, date=date),
        'to': streckennetz.get_eva(name=destination, date=date),
        'results': 6,
        'transfers': max_changes,
        'transferTime': transfer_time,
        'bike': bike,
        'tickets': True,
        'nationalExpress': False if only_regional else True,
        'national': False if only_regional else True,
    }

    if search_for_departure:
        request_data['departure'] = date.replace(tzinfo=timezone("CET")).isoformat()
    else:
        request_data['arrival'] = date.replace(tzinfo=timezone("CET")).isoformat()

    journeys: list[dict] = requests.get(
        'https://db-rest.bahnvorhersage.de/journeys', params=request_data
    ).json()['journeys']

    trip_ids = extract_trip_ids(journeys)
    train_trips = get_trips_of_trains(trip_ids)

    prediction_data = [extract_iris_like(journey, train_trips) for journey in journeys]

    ridable = {i for i in range(len(prediction_data)) if prediction_data[i] is not None}
    prediction_data = [prediction_data[i] for i in ridable]
    journeys = [journeys[i] for i in ridable]

    return journeys, prediction_data


@ttl_lru_cache(maxsize=100, seconds_to_live=60 * 3)
def get_and_rate_journeys(
    start: str,
    destination: str,
    date: datetime.datetime,
    search_for_departure: bool = True,
    only_regional: bool = False,
    bike: bool = False,
) -> list[dict]:
    journeys, prediction_data = get_journeys(
        start=start,
        destination=destination,
        date=date,
        search_for_departure=search_for_departure,
        only_regional=only_regional,
        bike=bike,
    )

    for i, prediction in enumerate(map(rate_journey, prediction_data, journeys)):
        prediction: Prediction
        journeys[i]['connectionScore'] = prediction.connection_score

        # This id is used for vue.js to render the list of connections
        journeys[i]['id'] = i
        journeys[i]['departure'] = journeys[i]['legs'][0]['departure']
        journeys[i]['plannedDeparture'] = journeys[i]['legs'][0]['plannedDeparture']
        journeys[i]['arrival'] = journeys[i]['legs'][-1]['arrival']
        journeys[i]['plannedArrival'] = journeys[i]['legs'][-1]['plannedArrival']
        journeys[i]['duration'] = (
            from_utc(journeys[i]['arrival']) - from_utc(journeys[i]['departure'])
        ).total_seconds()
        journeys[i]['plannedDuration'] = (
            from_utc(journeys[i]['plannedArrival'])
            - from_utc(journeys[i]['plannedDeparture'])
        ).total_seconds()
        journeys[i]['price'] = (
            journeys[i]['price']['amount'] if journeys[i]['price'] is not None else -1
        )

        walking_legs = 0
        train_categories = set()
        for leg_index, leg in enumerate(journeys[i]['legs']):
            if 'walking' in leg and leg['walking'] == True:
                walking_legs += 1
                train_categories.add('Fußweg')
                continue
            # The last leg has no transfer and thus no transferScore
            if leg_index != len(journeys[i]['legs']) - 1:
                journeys[i]['legs'][leg_index][
                    'transferScore'
                ] = prediction.transfer_score[leg_index - walking_legs]
                journeys[i]['legs'][leg_index][
                    'neededTransferTime'
                ] = prediction.needed_transfer_times[leg_index - walking_legs].to_dict()
                journeys[i]['legs'][leg_index][
                    'transferTime'
                ] = prediction.transfer_times[leg_index - walking_legs]
            journeys[i]['legs'][leg_index][
                'arrivalPrediction'
            ] = prediction.ar_predictions[leg_index - walking_legs]
            journeys[i]['legs'][leg_index][
                'departurePrediction'
            ] = prediction.dp_predictions[leg_index - walking_legs]
            train_categories.add(leg['line']['productName'])

        journeys[i]['trainCategories'] = sorted(list(train_categories))
        journeys[i]['transfers'] = len(journeys[i]['legs']) - 1 - walking_legs

    return journeys


def extract_trip_ids(journeys: list[dict]) -> set:
    trip_ids = set()
    for journey in journeys:
        for leg in journey['legs']:
            if not leg.get('walking'):
                trip_ids.add(leg['tripId'])
    return trip_ids


def extract_iris_like(journeys: list[dict], train_trips: dict) -> list[dict]:
    segments = []

    for leg in journeys['legs']:
        if 'walking' in leg and leg['walking'] == True:
            continue

        if 'cancelled' in leg and leg['cancelled'] == True:
            return None

        parsed_segment = {
            'dp_lat': leg['origin']['location']['latitude'],
            'dp_lon': leg['origin']['location']['longitude'],
            'dp_pt': from_utc(leg['plannedDeparture']),
            'dp_ct': from_utc(leg['departure']),
            'dp_pp': leg['plannedDeparturePlatform']
            if 'plannedDeparturePlatform' in leg
            else None,
            'dp_cp': leg['departurePlatform'] if 'departurePlatform' in leg else None,
            'dp_station': leg['origin']['name'],
            'ar_lat': leg['destination']['location']['latitude'],
            'ar_lon': leg['destination']['location']['longitude'],
            'ar_pt': from_utc(leg['plannedArrival']),
            'ar_ct': from_utc(leg['arrival']),
            'ar_pp': leg['plannedArrivalPlatform']
            if 'plannedArrivalPlatform' in leg
            else None,
            'ar_cp': leg['arrivalPlatform'] if 'arrivalPlatform' in leg else None,
            'ar_station': leg['destination']['name'],
            'train_name': leg['line']['name'],
            'ar_c': leg['line']['productName'],
            'ar_n': leg['line']['fahrtNr'],
            'ar_o': leg['line']['adminCode'].replace('_', ''),
            'dp_c': leg['line']['productName'],
            'dp_n': leg['line']['fahrtNr'],
            'dp_o': leg['line']['adminCode'].replace('_', ''),
            'walk': 0,
            'train_destination': leg['direction'],
        }

        parsed_segment['full_trip'], parsed_segment['stay_times'] = train_trips[
            leg['tripId']
        ]
        try:
            parsed_segment['dp_stop_id'] = parsed_segment['full_trip'].index(
                parsed_segment['dp_station']
            )
        except ValueError:
            try:
                parsed_segment['dp_stop_id'] = parsed_segment['full_trip'].index(
                    parsed_segment['dp_station_display_name']
                )
            except ValueError:
                # Sometimes, none of the stations is in the trip and we have no clue why
                print('Did not find station in trip')
                print('trip', parsed_segment['full_trip'])
                print('station', parsed_segment['dp_station_display_name'])
                parsed_segment['dp_stop_id'] = -1
        try:
            parsed_segment['ar_stop_id'] = parsed_segment['full_trip'].index(
                parsed_segment['ar_station']
            )
        except ValueError:
            parsed_segment['ar_stop_id'] = parsed_segment['full_trip'].index(
                parsed_segment['ar_station_display_name']
            )
        parsed_segment['duration'] = str(
            parsed_segment['ar_ct'] - parsed_segment['dp_ct']
        )[:-3]
        segments.append(parsed_segment)

    # Add transfer times
    for leg in range(len(segments) - 1):
        if segments[leg + 1]['dp_ct'] < segments[leg]['ar_ct']:
            # Negative transfer time. This should not happen (opinion of the tcp dev team).
            # Hafas however sometimes returns connections with negative transfer times.
            segments[leg]['transfer_time'] = -(
                (segments[leg + 1]['ar_ct'] - segments[leg]['dp_ct']).seconds // 60
            )
        else:
            segments[leg]['transfer_time'] = (
                segments[leg + 1]['dp_ct'] - segments[leg]['ar_ct']
            ).seconds // 60
    return segments

def get_trip(trip_id: str) -> dict:
    for _ in range(3):
        r = requests.get(
            'https://db-rest.bahnvorhersage.de/trips/{}'.format(trip_id),
        )
        if r.ok:
            return r.json()['trip']
    raise requests.RequestException(r.text)


# This information does change over time, so a permanent cache would give
# wrong results. Thus, we only cache the result for 3 minutes.
@ttl_lru_cache(seconds_to_live=180, maxsize=500)
def get_trip_of_train(trip_id: str):
    trip = get_trip(trip_id)
    waypoints = [stopover['stop']['name'] for stopover in trip['stopovers']]
    stay_times = [
        (
            datetime.datetime.fromisoformat(stopover['departure'])
            - datetime.datetime.fromisoformat(stopover['arrival'])
        ).seconds
        // 60
        if stopover['departure'] is not None and stopover['arrival'] is not None
        else None
        for stopover in trip['stopovers']
    ]
    return trip_id, waypoints, stay_times


def get_trips_of_trains(trip_ids: set[str]):
    with ThreadPoolExecutor(max_workers=len(trip_ids)) as executor:
        return {
            trip_id: (waypoints, stay_times)
            for trip_id, waypoints, stay_times in executor.map(
                get_trip_of_train, trip_ids
            )
        }
