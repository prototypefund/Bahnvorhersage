import time
from typing import Optional, Tuple, Union

import pandas as pd
import requests


def get_station_from_db_stations(eva: int) -> Union[dict, None]:
    r = requests.get(f'https://v5.db.transport.rest/stations/{eva}')
    station = r.json()
    if 'error' in station:
        return None
    elif eva == int(station['id']):
        return {
            'eva': int(station['id']),
            'name': station['name'],
            'ds100': station['ril100'],
            'lat': station['location']['latitude'],
            'lon': station['location']['longitude'],
        }
    elif str(eva) in station['additionalIds']:
        return {
            'eva': eva,
            'name': station['name'],
            'ds100': station['ril100'],
            'lat': station['location']['latitude'],
            'lon': station['location']['longitude'],
        }
    else:
        return None


def get_station_from_hafas_stop(eva: int) -> Union[dict, None]:
    r = requests.get(f'https://db-rest.bahnvorhersage.de/stops/{eva}')
    stop = r.json()
    if 'error' in stop:
        return None
    elif eva == int(stop['id']):
        return {
            'eva': int(stop['id']),
            'name': stop['name'],
            # 'ds100': stop['ril100'],
            'lat': stop['location']['latitude'],
            'lon': stop['location']['longitude'],
        }
    elif 'additionalIds' in stop and str(eva) in stop['additionalIds']:
        return {
            'eva': eva,
            'name': stop['name'],
            # 'ds100': stop['ril100'],
            'lat': stop['location']['latitude'],
            'lon': stop['location']['longitude'],
        }
    else:
        return None


def get_pyhafas_location(client, eva: int) -> Optional[Tuple[float, float]]:
    import pyhafas.types.exceptions

    for _ in range(10):
        try:
            locations = client.locations(eva)
            for location in locations:
                if int(location.id) == eva:
                    return location.latitude, location.longitude
            else:
                return None

        except pyhafas.types.exceptions.GeneralHafasError:
            time.sleep(20)
    else:
        raise ValueError(f'Max retries exceeded for eva: {eva}')


def read_stations_from_derf_Travel_Status_DE_IRIS():
    derf_stations = requests.get(
        'https://raw.githubusercontent.com/derf/Travel-Status-DE-IRIS/master/share/stations.json'
    ).json()
    parsed_derf_stations = {
        'name': [],
        'eva': [],
        'ds100': [],
        'lat': [],
        'lon': [],
    }
    for station in derf_stations:
        parsed_derf_stations['name'].append(station['name'])
        parsed_derf_stations['eva'].append(station['eva'])
        parsed_derf_stations['ds100'].append(station['ds100'])
        parsed_derf_stations['lat'].append(station['latlong'][0])
        parsed_derf_stations['lon'].append(station['latlong'][1])
    return pd.DataFrame(parsed_derf_stations)
