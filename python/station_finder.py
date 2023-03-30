import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import sessionfactory, PlanById
from typing import Tuple, Set
import sqlalchemy
from rtd_crawler.parser_helpers import parse_path
from tqdm import tqdm
from redis import Redis
from config import redis_url


def extract_station_names(
    session: sqlalchemy.orm.Session, chunk_limits: Tuple[int, int]
) -> Set[str]:
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
    engine, Session = sessionfactory()

    station_names = set()


    with Session() as session:
        print('Get processable chunk limits from table...')
        db_chunk_limits = PlanById.get_chunk_limits(session)

        for chunk_limits in tqdm(db_chunk_limits, desc='Finding stations'):
            station_names.update(extract_station_names(session, chunk_limits))
            print('Found {} stations'.format(len(station_names)))

    return station_names


def main():
    redis_client = Redis.from_url(redis_url)

    stations = find_stations()
    redis_client.sadd('station_names', *stations)



if __name__ == '__main__':
    main()
