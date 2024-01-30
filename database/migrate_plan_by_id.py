from database.plan_by_id import PlanById
from database.plan_by_id_v2 import PlanByIdV2
import sqlalchemy
import sqlalchemy.orm
from database.engine import sessionfactory
from typing import Tuple, Dict
from helpers.StationPhillip import StationPhillip
from database.upsert import upsert_with_retry
from tqdm import tqdm


def migrate_chunk(
    session: sqlalchemy.orm.Session,
    engine: sqlalchemy.engine.Engine,
    chunk: Tuple[int, int],
    stations: StationPhillip,
    unknown_stations: set
):
    old_stops = PlanById.get_stops_from_chunk(session, chunk)
    mew_stops: Dict[int, Dict] = {}
    for old_hash in old_stops:
        try:
            stations.get_eva(name=old_stops[old_hash]['station'])
        except KeyError:
            if old_stops[old_hash]['station'] not in unknown_stations:
                unknown_stations.add(old_stops[old_hash]['station'])
                print('Unknown:', unknown_stations)
        # new_stop = PlanByIdV2(
        #     old_stops[old_hash],
        #     stop_id=stations.get_eva(name=old_stops[old_hash]['station']),
        # )
        # mew_stops[new_stop.hash_id] = new_stop.as_dict()

    # upsert_with_retry(
    #     engine=engine, table=PlanByIdV2.__table__, rows=list(mew_stops.values())
    # )
    return unknown_stations


def migrate():
    stations = StationPhillip()

    engine, Session = sessionfactory()

    with Session() as session:
        chunk_limits = PlanById.get_chunk_limits(session)

        unknown_stations = set()
        for chunk in tqdm(chunk_limits):
            unknown_stations = migrate_chunk(session, engine, chunk, stations, unknown_stations)


if __name__ == '__main__':
    migrate()
