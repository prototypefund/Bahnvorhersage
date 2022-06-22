import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from pytz import timezone
from webserver import streckennetz
from concurrent.futures import ThreadPoolExecutor
import datetime
from helpers import ttl_lru_cache
from . import client


def from_utc(utc_time: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(
        utc_time
    ).astimezone(timezone("Europe/Berlin")).replace(tzinfo=None)


def get_journeys(
    start: str,
    destination: str,
    date: datetime.datetime,
    max_changes: int=-1,
    transfer_time: int=0,
    search_for_departure: bool=True,
    only_regional: bool=False,
    bike: bool=False,
) -> list:
    """[summary]

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

    journeys = requests.get(
        'https://db-rest.bahnvorhersage.de/journeys', params=request_data
    ).json()['journeys']

    trip_ids = extract_trip_ids(journeys)
    train_trips = get_trips_of_trains(trip_ids)

    prediction_data = [extract_iris_like(journey, train_trips) for journey in journeys]
    
    ridable = { i for i in range(len(prediction_data)) if prediction_data[i] is not None }
    prediction_data = [prediction_data[i] for i in ridable]
    journeys = [journeys[i] for i in ridable]

    return journeys, prediction_data


def extract_trip_ids(journeys) -> set:
    trip_ids = set()
    for journey in journeys:
        for leg in journey['legs']:
            if not leg.get('walking'):
                trip_ids.add(leg['tripId'])
    return trip_ids


def extract_iris_like(journeys, train_trips):
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
            'dp_pp': leg['plannedDeparturePlatform'] if 'plannedDeparturePlatform' in leg else None,
            'dp_cp': leg['departurePlatform'] if 'departurePlatform' in leg else None,
            'dp_station': leg['origin']['name'],

            'ar_lat': leg['destination']['location']['latitude'],
            'ar_lon': leg['destination']['location']['longitude'],
            'ar_pt': from_utc(leg['plannedArrival']),
            'ar_ct': from_utc(leg['arrival']),
            'ar_pp': leg['plannedArrivalPlatform'] if 'plannedArrivalPlatform' in leg else None,
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
        
        parsed_segment['full_trip'], parsed_segment['stay_times'] = train_trips[leg['tripId']]
        try:
            parsed_segment['dp_stop_id'] = parsed_segment['full_trip'].index(parsed_segment['dp_station'])
        except ValueError:
            try:
                parsed_segment['dp_stop_id'] = parsed_segment['full_trip'].index(parsed_segment['dp_station_display_name'])
            except ValueError:
                # Sometimes, none of the stations is in the trip and we have no clue why
                print('Did not find station in trip')
                print('trip', parsed_segment['full_trip'])
                print('station', parsed_segment['dp_station_display_name'])
                parsed_segment['dp_stop_id'] = -1
        try:
            parsed_segment['ar_stop_id'] = parsed_segment['full_trip'].index(parsed_segment['ar_station'])
        except ValueError:
            parsed_segment['ar_stop_id'] = parsed_segment['full_trip'].index(parsed_segment['ar_station_display_name'])
        parsed_segment['duration'] = str(parsed_segment['ar_ct'] - parsed_segment['dp_ct'])[:-3]
        segments.append(parsed_segment)

    # Add transfer times
    for leg in range(len(segments) - 1):
        if segments[leg + 1]['dp_ct'] < segments[leg]['ar_ct']:
            # Negative transfer time. This should not happen (opinion of the tcp dev team).
            # Hafas however sometimes returns connections with negative transfer times.
            segments[leg]['transfer_time'] = -(
                (segments[leg + 1]['ar_ct']
                - segments[leg]['dp_ct']).seconds
                // 60
            )
        else:
            segments[leg]['transfer_time'] = (
                (segments[leg + 1]['dp_ct']
                - segments[leg]['ar_ct']).seconds
                // 60
            )
    return segments


# This information does change over time, so a permanent cache would give
# wrong results. Thus, we only cache the result for 3 minutes.
@ttl_lru_cache(seconds_to_live=180, maxsize=500)
def get_trip_of_train(trip_id):
    trip = client.trip(trip_id)
    waypoints = [stopover.stop.name for stopover in trip.stopovers]
    stay_times = [
        (stopover.departure - stopover.arrival).seconds // 60
        if stopover.departure is not None and stopover.arrival is not None
        else None
        for stopover
        in trip.stopovers
    ]
    return trip_id, waypoints, stay_times


def get_trips_of_trains(trip_ids: set) -> dict:
    with ThreadPoolExecutor(max_workers=len(trip_ids)) as executor:
        return {
            trip_id: (waypoints, stay_times)
            for trip_id, waypoints, stay_times
            in executor.map(get_trip_of_train, trip_ids)
        }