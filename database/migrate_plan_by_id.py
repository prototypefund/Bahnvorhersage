from database.plan_by_id import PlanById
from database.plan_by_id_v2 import PlanByIdV2
import sqlalchemy
import sqlalchemy.orm
from database.engine import sessionfactory
from typing import Tuple, Dict
from helpers.StationPhillip import StationPhillip
from database.upsert import upsert_with_retry
from tqdm import tqdm
import datetime
from rtd_crawler.parser_helpers import db_to_utc


def migrate_chunk(
    session: sqlalchemy.orm.Session,
    engine: sqlalchemy.engine.Engine,
    chunk: Tuple[int, int],
    stations: StationPhillip,
):
    old_stops = PlanById.get_stops_from_chunk(session, chunk)
    mew_stops: Dict[int, Dict] = {}
    for old_hash in old_stops:
        # Skip stops before 2021-09-01, because there was a bug that mismatched
        # the stations. So data points might have the wrong station.
        ar_pt = old_stops[old_hash].get('ar', [{'pt': None}])[0].get('pt', None)
        dp_pt = old_stops[old_hash].get('dp', [{'pt': None}])[0].get('pt', None)
        if ar_pt is not None and db_to_utc(ar_pt) < datetime.datetime(
            2021, 9, 1, tzinfo=datetime.timezone.utc
        ):
            continue
        if dp_pt is not None and db_to_utc(dp_pt) < datetime.datetime(
            2021, 9, 1, tzinfo=datetime.timezone.utc
        ):
            continue
        stations.get_eva(name=old_stops[old_hash]['station'])

        new_stop = PlanByIdV2(
            old_stops[old_hash],
            stop_id=stations.get_eva(name=old_stops[old_hash]['station']),
        )
        mew_stops[new_stop.hash_id] = new_stop.as_dict()

    upsert_with_retry(
        engine=engine, table=PlanByIdV2.__table__, rows=list(mew_stops.values())
    )


def migrate():
    stations = StationPhillip()

    engine, Session = sessionfactory()

    with Session() as session:
        chunk_limits = PlanById.get_chunk_limits(session)

        for chunk in tqdm(chunk_limits):
            migrate_chunk(
                session,
                engine,
                chunk,
                stations,
            )


if __name__ == '__main__':
    migrate()
