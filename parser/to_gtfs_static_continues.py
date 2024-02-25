import sys
import time
import traceback
from parser.gtfs_upserter import GTFSUpserter
from parser.to_gtfs_static import parse_chunk

from redis import Redis

from config import redis_url
from database import unparsed


def parse_unparsed(
    redis_client: Redis, last_stream_id: bytes, upserter: GTFSUpserter
) -> bytes:
    last_stream_id, unparsed_hash_ids = unparsed.get_plan(redis_client, last_stream_id)
    if unparsed_hash_ids:
        print('parsing', len(unparsed_hash_ids), 'unparsed events')
        upserter.upsert(*parse_chunk(hash_ids=unparsed_hash_ids))
        upserter.flush()
    return last_stream_id


def parse_unparsed_continues():
    upserter = GTFSUpserter()
    redis_client = Redis.from_url(redis_url)
    last_stream_id = b'0-0'
    while True:
        try:
            last_stream_id = parse_unparsed(redis_client, last_stream_id, upserter)
        except Exception:
            traceback.print_exc(file=sys.stdout)
        time.sleep(60)


if __name__ == '__main__':
    parse_unparsed_continues()
