from typing import Set, Tuple

import sqlalchemy
from redis import Redis
from tqdm import tqdm

from api.iris import parse_path
from config import redis_url
from database import PlanById, sessionfactory


def extract_station_names(
    session: sqlalchemy.orm.Session, chunk_limits: Tuple[int, int]
) -> Set[str]:
    """
    For each historic delay datapoint, there is a path of stations that
    the train passed through. If a new station is build, it probably
    lies on one of these paths. This function extracts all station names
    from the paths of a chunk of historic delay datapoints.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Session to connect to the database
    chunk_limits : Tuple[int, int]
        Upper and lower limit of the hash_id of the chunk to process

    Returns
    -------
    Set[str]
        Set of station names
    """
    station_names = set()

    stops = PlanById.get_stops_from_chunk(session, chunk_limits)
    for stop in stops.values():
        if 'ar' in stop:
            stations = parse_path(stop['ar'][0].get('ppth'))
            if stations is not None:
                station_names.update(stations)

        if 'dp' in stop:
            stations = parse_path(stop['dp'][0].get('ppth'))
            if stations is not None:
                station_names.update(stations)

    return station_names


def find_stations() -> Set[str]:
    """
    Find all station names that exist in the historic delay data.

    Returns
    -------
    Set[str]
        Set of all station names
    """
    engine, Session = sessionfactory()

    station_names = set()

    with Session() as session:
        print('Get processable chunk limits from table...')
        db_chunk_limits = PlanById.get_chunk_limits(session)

        for chunk_limits in tqdm(db_chunk_limits, desc='Finding stations'):
            station_names.update(extract_station_names(session, chunk_limits))
            print(f'Found {len(station_names)} stations')

    return station_names


def main():
    redis_client = Redis.from_url(redis_url)

    stations = find_stations()
    redis_client.sadd('station_names', *stations)


if __name__ == '__main__':
    main()
