import concurrent.futures
import time
import traceback
from typing import Callable, List

import requests
from redis import Redis

import database.unparsed as unparsed
from api.iris import get_all_changes, get_recent_changes
from config import redis_url, station_to_monitor_per_thread
from database.base import create_all
from database.engine import sessionfactory
from database.unique_change import UniqueChange
from database.upsert import upsert_with_retry
from helpers.batcher import batcher
from helpers.StationPhillip import StationPhillip


def crawler_worker(evas: List[int], download_function: Callable):
    changes = {}
    with requests.Session() as session:
        for eva in evas:
            try:
                result = download_function(eva, session=session)
                for change in result:
                    change = UniqueChange(change).as_dict()
                    changes[change['change_hash']] = change
            except requests.exceptions.HTTPError:
                pass
        return changes


def get_and_process_changes(
    evas: List[int], download_function: Callable, engine, redis_client
):
    eva_batches = [
        list(batch) for batch in batcher(evas, station_to_monitor_per_thread)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(eva_batches)
    ) as executor:
        futures = list(
            executor.submit(crawler_worker, eva_batch, download_function)
            for eva_batch in eva_batches
        )

        changes = {}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            changes.update(result)

        upsert_with_retry(
            engine=engine, table=UniqueChange.__table__, rows=list(changes.values())
        )
        unparsed.add_change(redis_client, [changes[key]['hash_id'] for key in changes])


def main():
    engine, Session = sessionfactory()
    create_all(engine)

    stations = StationPhillip()
    redis_client = Redis.from_url(redis_url)

    get_and_process_changes(
        stations.stations['eva'].unique().tolist(),
        get_all_changes,
        engine,
        redis_client,
    )

    while True:
        try:
            start_time = time.time()
            get_and_process_changes(
                stations.stations['eva'].unique().tolist(),
                get_recent_changes,
                engine,
                redis_client,
            )
            print(f'Finished in {time.time() - start_time} seconds')
            time.sleep(max(0, 120 - (time.time() - start_time)))
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    main()
