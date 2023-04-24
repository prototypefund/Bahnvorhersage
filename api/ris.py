import urllib.parse
from dataclasses import dataclass

import requests

from config import ris_headers


@dataclass
class RisStopPlace:
    name: str
    eva: int
    # station_id does not exist in NL (probably in other non DE countries neither)
    # might also be missing on all not DB stations
    station_id: int | None
    lat: float
    lon: float
    country: str
    # state does not exist in NL (probably in other non DE countries neither)
    state: str | None
    # metropolis is missing sometimes
    metropolis: str | None
    available_transports: list[str]
    # transport_associations does not exist in NL (probably in other non DE countries neither)
    transport_associations: list[str] | None

    def __init__(self, stop_place: dict) -> None:
        self.name = stop_place['names']['DE']['nameLong']
        self.eva = int(stop_place['evaNumber'])
        self.station_id = None
        if 'stationID' in stop_place:
            self.station_id = int(stop_place['stationID'])
        self.lat = float(stop_place['position']['latitude'])
        self.lon = float(stop_place['position']['longitude'])
        self.country = stop_place['countryCode']
        self.state = None
        if 'state' in stop_place:
            self.state = stop_place['state']
        self.metropolis = None
        if 'metropolis' in stop_place:
            self.metropolis = stop_place['metropolis']['DE']
        self.available_transports = stop_place['availableTransports']
        self.transport_associations = None
        if 'transportAssociations' in stop_place:
            self.transport_associations = stop_place['transportAssociations']


def stop_place_by_name(name: str) -> RisStopPlace | None:
    """Get a stop place by its name. See https://developers.deutschebahn.com/db-api-marketplace/apis/product/ris-stations/api/ris-stations#/RISStations_1132/operation/%2Fstop-places%2Fby-name%2F{query}/get

    Parameters
    ----------
    name : str
        The name of the stop place to get

    Returns
    -------
    RisStopPlace | None
        Either the stop place or None if it could not be found
    """
    r = requests.get(
        f'https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/stop-places/by-name/{urllib.parse.quote(name)}',
        headers=ris_headers,
    )
    for stop_place in r.json().get('stopPlaces', []):
        if stop_place['names']['DE']['nameLong'] == name:
            return RisStopPlace(stop_place)
    else:
        return None


def stop_place_by_eva(eva: int) -> RisStopPlace | None:
    """Get a RIS stop place by its EVA number. See https://developers.deutschebahn.com/db-api-marketplace/apis/product/ris-stations/api/ris-stations#/RISStations_1132/operation/%2Fstop-places%2Fby-key/get

    Parameters
    ----------
    eva : int
        Eva number of the stop place to get

    Returns
    -------
    RisStopPlace | None
        Either the stop place or None if it could not be found
    """
    params = {'key': eva, 'keyType': 'EVA'}
    r = requests.get(
        'https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/stop-places/by-key',
        headers=ris_headers,
        params=params,
    )
    for stop_place in r.json().get('stopPlaces', []):
        if int(stop_place['evaNumber']) == eva:
            return RisStopPlace(stop_place)
    else:
        return None


if __name__ == '__main__':
    print(stop_place_by_eva(8000141))
    print(stop_place_by_eva(8000001))
    print(stop_place_by_eva(8000002))
    print(stop_place_by_eva(8000003))  # Does not exist
