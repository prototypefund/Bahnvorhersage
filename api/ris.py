import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import isodate
import requests

from config import ris_headers


def credentials_factory() -> dict:
    """
    Get credential headers where used the fewest times
    in order to not exceed the limit
    """
    headers_used = [0 for _ in ris_headers]

    def _get_credentials_header() -> dict:
        min_index = headers_used.index(min(headers_used))
        headers_used[min_index] += 1
        return ris_headers[min_index]

    return _get_credentials_header


get_credentials_header = credentials_factory()


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
        headers=get_credentials_header(),
    )
    if r.ok:
        for stop_place in r.json().get('stopPlaces', []):
            if stop_place['names']['DE']['nameLong'] == name:
                return RisStopPlace(stop_place)
        else:
            return None
    elif r.status_code == 429:
        raise requests.HTTPError('Too many requests, rate limit exceeded')
    else:
        raise requests.RequestException(r.text)


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
        headers=get_credentials_header(),
        params=params,
    )
    if r.ok:
        for stop_place in r.json().get('stopPlaces', []):
            if int(stop_place['evaNumber']) == eva:
                return RisStopPlace(stop_place)
        else:
            return None
    elif r.status_code == 429:
        raise requests.HTTPError('Too many requests, rate limit exceeded')
    else:
        raise requests.RequestException(r.text)


@dataclass
class RisTransferDuration:
    duration: timedelta
    distance: float | None

    def __init__(
        self,
        connection_duration: dict | None,
        duration: timedelta = None,
        distance: float = None,
    ) -> None:
        if connection_duration is None:
            self.duration = duration
            self.distance = distance
            return
        self.duration = isodate.parse_duration(connection_duration['duration'])
        # Distance does only exist if duration source is INDOOR_ROUTING
        self.distance = connection_duration.get('distance', None)


@dataclass
class RisTransfer:
    from_eva: int
    to_eva: int
    # platform might be missing on inter-station connections
    from_platform: str | None
    to_platform: str | None
    identical_physical_platform: bool
    frequent_traveller: RisTransferDuration
    mobility_impaired: RisTransferDuration
    occasional_traveller: RisTransferDuration
    source: Literal['RIL420', 'INDOOR_ROUTING', 'EFZ']

    def __init__(self, connection_time: dict) -> None:
        self.from_eva = int(connection_time['fromEvaNumber'])
        self.to_eva = int(connection_time['toEvaNumber'])
        self.from_platform = connection_time.get('fromPlatform', None)
        self.to_platform = connection_time.get('toPlatform', None)
        self.identical_physical_platform = connection_time['identicalPhysicalPlatform']
        self.frequent_traveller = RisTransferDuration(None)
        self.mobility_impaired = RisTransferDuration(None)
        self.occasional_traveller = RisTransferDuration(None)
        for time in connection_time['times']:
            if time['persona'] == 'FREQUENT_TRAVELLER':
                self.frequent_traveller = RisTransferDuration(time)
            elif time['persona'] == 'HANDICAPPED':
                self.mobility_impaired = RisTransferDuration(time)
            elif time['persona'] == 'OCCASIONAL_TRAVELLER':
                self.occasional_traveller = RisTransferDuration(time)
        self.source = connection_time['source']


def transfer_times_by_eva(eva: int):
    r = requests.get(
        f'https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/connecting-times/{eva}',
        headers=get_credentials_header(),
    )
    if r.ok:
        return [
            RisTransfer(connection_time)
            for connection_time in r.json().get('connectingTimesList', [])
        ]
    elif r.status_code == 429:
        raise requests.HTTPError('Too many requests, rate limit exceeded')
    else:
        raise requests.RequestException(r.text)


if __name__ == '__main__':
    transfer_times_by_eva(8000096)
    print(transfer_times_by_eva(8000001))
    print(transfer_times_by_eva(8000002))
    print(transfer_times_by_eva(8000003))  # Does not exist
