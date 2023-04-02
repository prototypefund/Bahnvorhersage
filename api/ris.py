import requests
import urllib.parse
from config import ris_headers


def stop_place_by_name(name: str) -> dict | None:
    r = requests.get(
        f'https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/stop-places/by-name/{urllib.parse.quote(name)}',
        headers=ris_headers,
    )
    for stop_place in r.json().get('stopPlaces', []):
        if stop_place['names']['DE']['nameLong'] == name:
            return stop_place


def stop_place_by_eva(eva: int) -> dict | None:
    params = {'key': eva, 'keyType': 'EVA'}
    r = requests.get(
        'https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/stop-places/by-key',
        headers=ris_headers,
        params=params,
    )
    for stop_place in r.json().get('stopPlaces', []):
        if int(stop_place['evaNumber']) == eva:
            return stop_place


def parse_stop_place(stop_place: dict) -> dict


def location():
    pass


if __name__ == '__main__':
    print(stop_place_by_name('Tübingen Hbf'))
    print(stop_place_by_eva(8000141))
