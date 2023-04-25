import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Union

import lxml.etree as etree
import pandas as pd
import requests

from rtd_crawler.xml_parser import xml_to_json


@dataclass
class IrisStation:
    name: str
    eva: int
    ds100: str
    db: bool
    valid_from: datetime
    valid_to: datetime
    creationts: datetime
    meta: List[int] = field(default_factory=list)

    def __init__(self, iris_station: dict) -> None:
        self.name = iris_station['name']
        self.eva = int(iris_station['eva'])
        self.ds100 = iris_station['ds100']
        self.db = iris_station['db'] == 'true'
        self.creationts = datetime.strptime(
            iris_station['creationts'], '%d-%m-%y %H:%M:%S.%f'
        )
        self.valid_from = datetime.strptime(
            iris_station['creationts'], '%d-%m-%y %H:%M:%S.%f'
        )
        self.valid_to = pd.Timestamp.max.to_pydatetime(warn=False)

        self.meta = parse_meta(iris_station.get('meta', ''))


def stations_equal(iris_station: IrisStation, station: pd.Series) -> bool:
    """
    Check weather two stations are equal (same name, same eva, same ds100)
    Args:
        iris_station: A station from the IRIS API
        station: A station from StationPhillip's database

    Returns: True if the stations are equal, False otherwise
    """
    return (
        iris_station.name == station['name']
        and iris_station.eva == station['eva']
        and iris_station.ds100 == station['ds100']
    )


def parse_meta(meta: str) -> List[int]:
    """Parse meta string from IRIS

    Parameters
    ----------
    meta : str
        Metas separated by | (e.g. 80001|80002|80004)

    Returns
    -------
    List[int]
        List of metas. Empty list if meta is empty
    """
    if not meta:
        return []
    meta = meta.split('|')
    return list(map(int, meta))


def get_stations_from_iris(search_term: Union[str, int]) -> IrisStation | None:
    """
    Search IRIS for a station either by name or eva number

    Parameters
    ----------
    search_term : Union[str, int]
        Name or eva number of the station

    Returns
    -------
    IrisStation | None
        found station or None if no station was found

    Raises
    ------
    ValueError
        Invalid response from IRIS
    ValueError
        More than 1 match found for search_term
    """
    if isinstance(search_term, int):
        search_term = str(search_term)
    search_term = urllib.parse.quote(search_term)

    matches = requests.get(
        f'http://iris.noncd.db.de/iris-tts/timetable/station/{search_term}'
    )
    if matches.ok:
        matches = matches.text
    else:
        raise ValueError(
            'Did not get a valid response. Http Code: ' + str(matches.status_code)
        )

    matches = etree.fromstring(matches.encode())
    matches = list(xml_to_json(match) for match in matches)

    if len(matches) == 0:
        return None
    elif len(matches) != 1:
        raise ValueError(
            f'More than 1 match found for {search_term} which is unexpected'
        )
    match = matches[0]
    return IrisStation(match)


def search_iris_multiple(
    search_terms: Set[Union[str, int]]
) -> Iterator[Union[IrisStation, None]]:
    """Search IRIS for multiple stations

    Parameters
    ----------
    search_terms : Set[Union[str, int]]
        Iterable of search terms (name or eva number)

    Yields
    ------
    Iterator[Union[IrisStation, None]]
        An Iterator over the results of the search. If a station
        was not found, None is yielded instead of an IrisStation
    """
    for search_term in search_terms:
        yield get_stations_from_iris(search_term)


if __name__ == '__main__':
    # get_stations_from_iris(8000001)
    get_stations_from_iris('Tübingen')
