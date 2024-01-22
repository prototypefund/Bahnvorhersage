import enum
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Literal, Set, Union, Dict

import lxml.etree as etree
import pandas as pd
import requests

from rtd_crawler.hash64 import xxhash64
from rtd_crawler.parser_helpers import db_to_utc, parse_id, parse_path
from rtd_crawler.xml_parser import xml_to_json
from helpers.retry import retry


PLAN_URL = 'https://iris.noncd.db.de/iris-tts/timetable/plan/'
CHANGES_URL = 'https://iris.noncd.db.de/iris-tts/timetable/fchg/'
RECENT_CHANGES_URL = 'https://iris.noncd.db.de/iris-tts/timetable/rchg/'

IRIS_TIMEOUT = 90


class EventStatus(enum.Enum):
    PLANNED = 'p'
    ADDED = 'a'
    CANCELLED = 'c'


class MessagePriority(enum.Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DONE = 4


class MessageType(enum.Enum):
    HAFAS_INFORMATION_MANAGER = 'h'
    QUALITY_CHANGE = 'q'
    FREE = 'f'
    CAUSE_OF_DELAY = 'd'
    IBIS = 'i'
    UNASSIGNED_IBIS_MESSAGE = 'u'
    DISRUPTION = 'r'
    CONNECTION = 'c'


class DistributorType(enum.Enum):
    CITY = 's'
    REGION = 'r'
    LONG_DISTANCE = 'f'
    OTHER = 'x'


class Filter(enum.Enum):
    NON_DB = 'D'
    LONG_DISTANCE = 'F'
    REGIONAL = 'N'
    SBAHN = 'S'


class ConnectionStatus(enum.Enum):
    WAITING = 'w'
    TRANSITION = 'n'
    ALTERNATIVE = 'a'


class DelaySource(enum.Enum):
    LEIBIT = 'L'
    AUTOMATIC_IRIS_NE = 'NA'
    MANUAL_IRIS_NE = 'NM'
    VDV = 'V'
    AUTOMATIC_ISTP = 'IA'
    MANUAL_ISTP = 'IM'
    AUTOMATIC_PROGNOSIS = 'A'


class ReferenceTripRelationToStop(enum.Enum):
    BEFORE = 'b'
    END = 'e'
    BETWEEN = 'c'
    START = 's'
    AFTER = 'a'


@dataclass
class TripLabel:
    """Compound data type that contains common data items that characterize a Trip"""

    category: str
    filter: Filter
    number: str
    owner: str
    type: Literal['p', 'e', 'z', 's', 'h', 'n']
    # I did not find any documentation on what these letters mean

    def __init__(self, trip_label: dict) -> None:
        self.category = trip_label['c']
        self.filter = Filter(trip_label['f']) if 'f' in trip_label else None
        self.number = trip_label['n']
        self.owner = trip_label['o']
        self.type = trip_label['t']


@dataclass
class DistributorMessage:
    """An additional message to a given station-based disruption by a specific distributor."""

    internal_text: str
    distributor_name: str
    distributor_type: DistributorType
    timestamp: datetime

    def __init__(self, distributor_message: dict) -> None:
        self.internal_text = distributor_message['int']
        self.distributor_name = distributor_message['n']
        self.distributor_type = DistributorType(distributor_message['t'])
        self.timestamp = db_to_utc(distributor_message['ts'])


@dataclass
class Message:
    """A message that is associated with an event, a stop or a trip."""

    code: int
    category: str
    deleted: bool
    distributor_message: DistributorMessage
    external_category: str
    external_link: str
    external_text: str
    valid_from: datetime
    message_id: str
    internal_text: str
    owner: str
    priority: MessagePriority
    message_type: MessageType
    trip_label: TripLabel
    valid_to: datetime
    timestamp: datetime

    def __init__(self, message: dict) -> None:
        self.code = message['c']
        self.category = message['cat']
        self.deleted = bool(message['del'])
        self.distributor_message = DistributorMessage(message['dm'])
        self.external_category = message['ec']
        self.external_link = message['elnk']
        self.external_text = message['ext']
        self.valid_from = db_to_utc(message['from'])
        self.message_id = message['id']
        self.internal_text = message['int']
        self.owner = message['o']
        self.priority = MessagePriority(message['pr'])
        self.message_type = MessageType(message['t'])
        self.trip_label = TripLabel(message['tl'])
        self.valid_to = db_to_utc(message['to'])
        self.timestamp = db_to_utc(message['ts'])


@dataclass
class Event:
    """An event (arrival or departure) that is part of a stop."""

    changed_distant_endpoint: str
    cancellation_time: datetime
    changed_platform: str
    changed_path: List[str]
    changed_status: EventStatus
    changed_time: datetime
    distant_change: int
    hidden: bool
    line: str
    messages: Message
    planned_distant_endpoint: str
    planned_platform: str
    planned_path: List[str]
    planned_status: EventStatus
    planned_time: datetime
    transition: str
    wings: List[str]

    def __init__(self, event: dict) -> None:
        if 'dc' in event:
            raise NotImplementedError(
                f'Found wieird event. Please report this to the developers: {event}'
            )

        self.changed_distant_endpoint = event['cde'] if 'cde' in event else None
        self.cancellation_time = db_to_utc(event['clt']) if 'clt' in event else None
        self.changed_platform = event['cp'] if 'cp' in event else None
        self.changed_path = parse_path(event['cpth']) if 'cpth' in event else None
        self.changed_status = EventStatus(event['cs']) if 'cs' in event else None
        self.changed_time = db_to_utc(event['ct']) if 'ct' in event else None
        self.distant_change = int(event['dc']) if 'dc' in event else None
        self.hidden = bool(int(event['hi'])) if 'hi' in event else None
        self.line = event['l'] if 'l' in event else None
        self.messages = (
            Message(event['m']) if 'm' in event else None if 'm' in event else None
        )
        self.planned_distant_endpoint = event['pde'] if 'pde' in event else None
        self.planned_platform = event['pp']
        self.planned_path = parse_path(event['ppth'])
        self.planned_status = EventStatus(event['ps']) if 'ps' in event else None
        self.planned_time = db_to_utc(event['pt'])
        self.transition = event['tra'] if 'tra' in event else None
        self.wings = event['wings'].split('|') if 'wings' in event else None


@dataclass
class Connection:
    """
    It's information about a connected train at a
    particular stop.
    """

    connection_status: ConnectionStatus
    eva: int
    connection_id: str
    ref: ...  # not in documentation
    s: ...  # not in documentation
    timestamp: datetime


@dataclass
class HistoricDelay:
    """
    It's the history of all delay-messages for a stop.
    This element extends HistoricChange.
    """

    arrival: datetime
    delay_cause: str
    departure: datetime
    delay_source: DelaySource
    timestamp: datetime


@dataclass
class HistoricPlatformChange:
    """
    It's the history of all platform-changes for a stop.
    This element extends HistoricChange.
    """

    arrival: datetime
    cause_of_track_change: str
    departure: datetime
    timestamp: datetime


@dataclass
class TripReference:
    """
    It's a reference to another trip, which holds its
    label and reference trips, if available.
    """

    refered_trips: List[TripLabel]
    trip_label: TripLabel


@dataclass
class ReferenceTripStopLabel:
    """
    It's a compound data type that contains common data
    items that characterize a reference trip stop. The
    contents is represented as a compact 4-tuple in XML.
    """

    eva: int
    index: int
    name: str
    planned_time: datetime


@dataclass
class ReferenceTripLabel:
    category: str
    number: str


@dataclass
class ReferenceTrip:
    """
    A reference trip is another real trip, but it
    doesn't have its own stops and events. It refers
    only to its ref-erenced regular trip. The reference
    trip collects mainly all different attributes of the
    referenced regular trip.
    """

    cancelled: bool
    ea: ReferenceTripStopLabel  # ? what ea means
    trip_id: int
    date_id: datetime
    reference_trip_label: ReferenceTripLabel
    sd: ReferenceTripStopLabel  # ? what sd means


@dataclass
class ReferenceTripRelation:
    reference_trip: ReferenceTrip
    reference_trip_relation_to_stop: ReferenceTripRelationToStop


@dataclass
class TimetableStop:
    """A stop is a part of a Timetable."""

    arrival: Event | None
    depature: Event | None
    connection: Connection
    eva: int
    station_name: str
    historic_delay: HistoricDelay
    historic_platform_change: HistoricPlatformChange
    raw_id: str
    hash_id: int
    trip_id: int
    date_id: datetime
    stop_id: int
    message: Message
    reference: TripReference
    rtr: ReferenceTripRelation
    trip_label: TripLabel

    def __init__(self, stop: dict) -> None:
        if (
            'conn' in stop
            or 'eva' in stop
            or 'hd' in stop
            or 'hpc' in stop
            or 'm' in stop
            or 'ref' in stop
            or 'rtr' in stop
        ):
            raise NotImplementedError(
                f'Found weird stop. Please report this to the developers: {stop}'
            )

        self.arrival = Event(stop['ar'][0]) if 'ar' in stop else None
        self.depature = Event(stop['dp'][0]) if 'dp' in stop else None
        self.connection = Connection(stop['conn']) if 'conn' in stop else None
        self.eva = int(stop['eva']) if 'eva' in stop else None
        self.station_name = stop['station']
        self.historic_delay = HistoricDelay(stop['hd']) if 'hd' in stop else None
        self.historic_platform_change = (
            HistoricPlatformChange(stop['hpc']) if 'hpc' in stop else None
        )
        self.raw_id = stop['id']
        self.trip_id, self.date_id, self.stop_id = parse_id(self.raw_id)
        self.hash_id = xxhash64(self.raw_id)
        self.message = Message(stop['m']) if 'm' in stop else None
        self.reference = TripReference(stop['ref']) if 'ref' in stop else None
        self.rtr = ReferenceTripRelation(stop['rtr']) if 'rtr' in stop else None
        self.trip_label = TripLabel(stop['tl'][0])

    def is_bus(self) -> bool:
        return (
            self.trip_label.category == 'Bus' or self.arrival.line == 'SEV'
            if self.arrival is not None
            else False or self.depature.line == 'SEV'
            if self.depature is not None
            else False
        )


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


def xml_str_to_json(xml_response: str) -> List[Dict]:
    xml_response = etree.fromstring(xml_response.encode())
    return list(xml_to_json(single) for single in xml_response)


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


@retry(max_retries=3)
def _make_iris_request(url: str, session: requests.Session = None) -> List[Dict]:
    if session is None:
        r = requests.get(url, timeout=IRIS_TIMEOUT)
    else:
        r = session.get(url, timeout=IRIS_TIMEOUT)
    r.raise_for_status()
    return xml_str_to_json(r.text)


def get_plan(eva: int, date: date, hour: int, session: requests.Session = None) -> List[Dict]:
    return _make_iris_request(
        PLAN_URL + f"{eva}/{date.strftime('%y%m%d')}/{hour:02d}", session=session
    )


def get_all_changes(eva: int, session: requests.Session = None) -> List[Dict]:
    return _make_iris_request(CHANGES_URL + f"{eva}", session=session)


def get_recent_changes(eva: int, session: requests.Session = None) -> List[Dict]:
    return _make_iris_request(RECENT_CHANGES_URL + f"{eva}", session=session)


if __name__ == '__main__':
    # get_stations_from_iris(8000001)
    get_stations_from_iris('Tübingen')
