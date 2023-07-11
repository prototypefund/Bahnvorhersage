import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Tuple

import pandas as pd
from redis import Redis
from tqdm import tqdm

from config import redis_url
from database import Change, PlanById, Rtd, session_scope, sessionfactory, unparsed
from helpers.StreckennetzSteffi import StreckennetzSteffi
from rtd_crawler.parser_helpers import db_to_datetime, parse_path

engine, Session = sessionfactory()

parser = argparse.ArgumentParser(description='Parse train delay data')
parser.add_argument(
    '--parse_continues',
    help='Check for unparsed data every 60 seconds and parse it',
    action="store_true",
)
parser.add_argument(
    '--parse_all',
    help='Parse all raw data that is in the database',
    action="store_true",
)
Rtd()
streckennetz = StreckennetzSteffi(prefer_cache=False)


def parse_stop_plan(hash_id: int, stop: dict) -> dict:
    # Split id into the three id parts: the id unique on the date, the date, the stop number
    id_parts = stop['id'].rsplit('-', 2)

    parsed = {
        'hash_id': hash_id,
        'dayly_id': int(id_parts[0]),
        'date_id': db_to_datetime(id_parts[1]),
        'stop_id': int(id_parts[2]),
        'station': stop['station'],
    }

    if 'tl' in stop:
        parsed['f'] = stop['tl'][0].get('f')
        parsed['t'] = stop['tl'][0].get('t')
        parsed['o'] = stop['tl'][0].get('o')
        parsed['c'] = stop['tl'][0].get('c')
        parsed['n'] = stop['tl'][0].get('n')
    else:
        parsed['f'] = None
        parsed['t'] = None
        parsed['n'] = None
        parsed['o'] = None
        parsed['c'] = None

    if 'ar' in stop:
        parsed['ar_pt'] = db_to_datetime(stop['ar'][0].get('pt'))
        parsed['ar_ppth'] = parse_path(stop['ar'][0].get('ppth'))
        parsed['ar_pp'] = stop['ar'][0].get('pp')  # TODO
        parsed['ar_ps'] = stop['ar'][0].get('ps')  # TODO
        parsed['ar_hi'] = bool(stop['ar'][0].get('hi', 0))  # TODO
        parsed['ar_pde'] = stop['ar'][0].get('pde')  # TODO
        parsed['ar_dc'] = bool(stop['ar'][0].get('dc', 0))  # TODO
        parsed['ar_l'] = stop['ar'][0].get('l')
    else:  # TODO
        parsed['ar_pt'] = None
        parsed['ar_ppth'] = None
        parsed['ar_pp'] = None
        parsed['ar_ps'] = None
        parsed['ar_hi'] = False
        parsed['ar_pde'] = None
        parsed['ar_dc'] = False
        parsed['ar_l'] = None

    if 'dp' in stop:
        parsed['dp_pt'] = db_to_datetime(stop['dp'][0].get('pt'))
        parsed['dp_ppth'] = parse_path(stop['dp'][0].get('ppth'))
        parsed['dp_pp'] = stop['dp'][0].get('pp')  # TODO
        parsed['dp_ps'] = stop['dp'][0].get('ps')  # TODO
        parsed['dp_hi'] = bool(stop['dp'][0].get('hi', 0))  # TODO
        parsed['dp_pde'] = stop['dp'][0].get('pde')  # TODO
        parsed['dp_dc'] = bool(stop['dp'][0].get('dc', 0))  # TODO
        parsed['dp_l'] = stop['dp'][0].get('l')
    else:  # TODO
        parsed['dp_pt'] = None
        parsed['dp_ppth'] = None
        parsed['dp_pp'] = None
        parsed['dp_ps'] = None
        parsed['dp_hi'] = False
        parsed['dp_pde'] = None
        parsed['dp_dc'] = False
        parsed['dp_l'] = None

    return parsed


def add_change(stop: dict, change: dict) -> dict:
    if 'ar' in change:
        stop['ar_ct'] = db_to_datetime(change['ar'][0].get('ct')) or stop['ar_pt']
        stop['ar_clt'] = db_to_datetime(change['ar'][0].get('clt'))
        stop['ar_cpth'] = parse_path(change['ar'][0].get('cpth')) or stop['ar_ppth']
        stop['ar_cs'] = change['ar'][0].get('cs', stop['ar_ps'])
        stop['ar_cp'] = change['ar'][0].get('cp', stop['ar_pp'])
    else:
        stop['ar_ct'] = stop['ar_pt']
        stop['ar_clt'] = None
        stop['ar_cpth'] = stop['ar_ppth']
        stop['ar_cs'] = stop['ar_ps']
        stop['ar_cp'] = stop['ar_pp']

    if 'dp' in change:
        stop['dp_ct'] = db_to_datetime(change['dp'][0].get('ct')) or stop['dp_pt']
        stop['dp_clt'] = db_to_datetime(change['dp'][0].get('clt'))
        stop['dp_cpth'] = parse_path(change['dp'][0].get('cpth')) or stop['dp_ppth']
        stop['dp_cs'] = change['dp'][0].get('cs', stop['dp_ps'])
        stop['dp_cp'] = change['dp'][0].get('cp', stop['dp_pp'])
    else:
        stop['dp_ct'] = stop['dp_pt']
        stop['dp_clt'] = None
        stop['dp_cpth'] = stop['dp_ppth']
        stop['dp_cs'] = stop['dp_ps']
        stop['dp_cp'] = stop['dp_pp']
    return stop


def add_route_info(stop: dict) -> dict:
    if stop['ar_cpth'] is not None:
        stop['distance_to_last'] = streckennetz.route_length(
            waypoints=[stop['ar_cpth'][-1]] + [stop['station']],
            date='latest',
            is_bus=stop['c']=='bus',
        )
        stop['distance_to_start'] = streckennetz.route_length(
            stop['ar_cpth'] + [stop['station']],
            date='latest',
            is_bus=stop['c']=='bus',
        )

        # path_obstacles = streckennetz.obstacles_of_path(stop['ar_cpth'] + [stop['station']], stop['ar_pt'])
        # TODO
        stop['obstacles_priority_24'] = 0
        stop['obstacles_priority_37'] = 0
        stop['obstacles_priority_63'] = 0
        stop['obstacles_priority_65'] = 0
        stop['obstacles_priority_70'] = 0
        stop['obstacles_priority_80'] = 0
    else:
        stop['distance_to_last'] = 0
        stop['distance_to_start'] = 0

        # TODO
        stop['obstacles_priority_24'] = 0
        stop['obstacles_priority_37'] = 0
        stop['obstacles_priority_63'] = 0
        stop['obstacles_priority_65'] = 0
        stop['obstacles_priority_70'] = 0
        stop['obstacles_priority_80'] = 0

    if stop['dp_cpth'] is not None:
        stop['distance_to_next'] = streckennetz.route_length(
            [stop['station']] + [stop['dp_cpth'][0]],
            date='latest',
            is_bus=stop['c']=='bus',
        )
        stop['distance_to_end'] = streckennetz.route_length(
            [stop['station']] + stop['dp_cpth'],
            date='latest',
            is_bus=stop['c']=='bus',
        )
    else:
        stop['distance_to_next'] = 0
        stop['distance_to_end'] = 0

    # These columns are only used during parsing and are no longer needed
    del stop['ar_ppth']
    del stop['ar_cpth']
    del stop['dp_ppth']
    del stop['dp_cpth']

    return stop


def parse_stop(hash_id: int, plan: dict, change: dict) -> dict:
    stop = parse_stop_plan(hash_id, plan)
    stop = add_change(stop, change)
    stop = add_route_info(stop)
    return stop


def parse_batch(hash_ids: List[int], plans: Dict[int, Dict] = None):
    with session_scope(Session) as session:
        if plans is None:
            plans = PlanById.get_stops(session, hash_ids)
        changes = Change.get_changes(session, hash_ids)
    parsed = []
    for hash_id in plans:
        parsed.append(parse_stop(hash_id, plans[hash_id], changes.get(hash_id, {})))

    if parsed:
        parsed = pd.DataFrame(parsed).set_index('hash_id')  # TODO
        Rtd.upsert(parsed, engine)  # TODO


def parse_unparsed(redis_client: Redis, last_stream_id: bytes) -> bytes:
    last_stream_id, unparsed_hash_ids = unparsed.get(redis_client, last_stream_id)
    if unparsed_hash_ids:
        print('parsing', len(unparsed_hash_ids), 'unparsed events')
        parse_batch(unparsed_hash_ids)
    return last_stream_id


def parse_unparsed_continues():
    redis_client = Redis.from_url(redis_url)
    last_stream_id = b'0-0'
    while True:
        try:
            last_stream_id = parse_unparsed(redis_client, last_stream_id)
        except Exception:
            traceback.print_exc(file=sys.stdout)
        time.sleep(60)


def parse_chunk(chunk_limits: Tuple[int, int]):
    """Parse all stops with hash_id within the limits

    Parameters
    ----------
    chunk_limits : Tuple[int, int]
        min and max hash_id to parse in this chunk
    """
    with session_scope(Session) as session:
        stops = PlanById.get_stops_from_chunk(session, chunk_limits)
    parse_batch(stops.keys(), stops)


def parse_all():
    """Parse all raw data there is"""
    # with session_scope(Session) as session:
    #     chunk_limits = PlanById.get_chunk_limits(session)
    
    import pickle
    chunk_limits = pickle.load(open('chunk_limits.pickle', 'rb'))

    # Non-concurrent code for debugging
    for chunk in tqdm(chunk_limits, total=len(chunk_limits)):
        parse_chunk(chunk)

    with concurrent.futures.ProcessPoolExecutor(
        min(16, os.cpu_count()), mp_context=mp.get_context('spawn')
    ) as executor:
        parser_tasks = {
            executor.submit(parse_chunk, chunk): chunk for chunk in chunk_limits
        }
        for future in tqdm(
            concurrent.futures.as_completed(parser_tasks), total=len(chunk_limits)
        ):
            future.result()


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    args = parser.parse_args()
    args.parse_all = True
    if args.parse_all:
        print('Parsing all the data')
        Rtd()
        parse_all()
    if args.parse_continues:
        print('Starting continues parser')
        parse_unparsed_continues()
