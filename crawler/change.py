import concurrent.futures
import datetime
import time
from itertools import chain

from redis import Redis
from tqdm import tqdm
import requests

from config import redis_url, station_to_monitor_per_thread
from database import sessionfactory, unparsed
from helpers.StationPhillip import StationPhillip
from helpers.batcher import batcher
from database.unique_change import UniqueChange
from api.iris import get_all_changes, get_recent_changes
from typing import List, Callable
from database.upsert import upsert_base
from database.base import create_all


def crawler_worker(evas: List[int], download_function: Callable):
    changes = []
    for eva in evas:
        try:
            result = download_function(eva)
            print(f'downloaded {eva}')
            # for change in result:
            #     changes.append(UniqueChange(change).as_dict())
        except requests.exceptions.HTTPError as exc:
            pass
    return changes


def get_and_process_changes(
    evas: List[int], download_function: Callable, Session, redis_client
):
    eva_batches = [
        list(batch) for batch in batcher(evas, station_to_monitor_per_thread)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eva_batches)) as executor:
        futures = list(
            executor.submit(crawler_worker, eva_batch, download_function)
            for eva_batch in eva_batches
        )

        changes = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f'got {len(result)} changes from finished worker')
            changes.extend(result)

        with Session() as session:
            upsert_base(session, UniqueChange.__table__, changes)
            session.commit()

        unparsed.add_change(redis_client, [change['hash_id'] for change in changes])


def main():
    engine, Session = sessionfactory()
    create_all(engine)

    stations = StationPhillip()
    redis_client = Redis.from_url(redis_url)

    get_and_process_changes(
            stations.stations['eva'].unique().tolist(),
            get_all_changes,
            Session,
            redis_client,
        )

    while True:
        start_time = time.time()
        get_and_process_changes(
            stations.stations['eva'].unique().tolist(),
            get_recent_changes,
            Session,
            redis_client,
        )
        print(f'Finished in {time.time() - start_time} seconds')
        time.sleep(max(0, 120 - (time.time() - start_time)))


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    main()