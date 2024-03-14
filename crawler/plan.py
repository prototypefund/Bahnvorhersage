import concurrent.futures
import time
import traceback
from datetime import date, datetime, timedelta

import requests
from redis import Redis

import database.unparsed as unparsed
from api.iris import get_plan
from config import redis_url, station_to_monitor_per_thread
from database.base import create_all
from database.engine import get_engine
from database.plan_by_id_v2 import PlanByIdV2
from database.upsert import upsert_with_retry
from helpers.batcher import batcher
from helpers.StationPhillip import StationPhillip


def hour_in_n_hours(hours) -> int:
    return (datetime.now() + timedelta(hours=hours)).time().hour


def date_in_n_hours(hours) -> int:
    return (datetime.now() + timedelta(hours=hours)).date()


def crawler_worker(evas: list[int], date: date, hour: int):
    plans = {}
    with requests.Session() as session:
        for eva in evas:
            try:
                result = get_plan(eva=eva, date=date, hour=hour, session=session)
                for plan in result:
                    plan = PlanByIdV2(plan, stop_id=eva).as_dict()
                    plans[plan['hash_id']] = plan
            except requests.exceptions.HTTPError:
                pass
        return plans


def get_and_process_plan(evas: list[int], date: date, hour: int, engine, redis_client):
    eva_batches = [
        list(batch) for batch in batcher(evas, station_to_monitor_per_thread)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(eva_batches)
    ) as executor:
        futures = list(
            executor.submit(crawler_worker, eva_batch, date, hour)
            for eva_batch in eva_batches
        )

        plans = {}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            plans.update(result)

        upsert_with_retry(
            engine=engine, table=PlanByIdV2.__table__, rows=list(plans.values())
        )
        unparsed.add_plan(redis_client, [hash_id for hash_id in plans])


def main():
    engine = get_engine()
    create_all(engine)

    stations = StationPhillip()
    redis_client = Redis.from_url(redis_url)

    hour = hour_in_n_hours(hours=11)
    date = date_in_n_hours(hours=11)

    while True:
        if hour == hour_in_n_hours(hours=12):
            time.sleep(20)
        else:
            hour = hour_in_n_hours(hours=12)
            date = date_in_n_hours(hours=12)
            try:
                start_time = time.time()
                get_and_process_plan(
                    evas=stations.stations['eva'].unique().tolist(),
                    date=date,
                    hour=hour,
                    engine=engine,
                    redis_client=redis_client,
                )
                print(
                    f'{datetime.now()}: Finished in {time.time() - start_time} seconds'
                )
            except Exception:
                traceback.print_exc()


if __name__ == '__main__':
    from helpers.bahn_vorhersage import COLORFUL_ART

    print(COLORFUL_ART)

    main()
