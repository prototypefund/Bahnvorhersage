from typing import List, Tuple

from redis import Redis

def add_change(redis_client: Redis, hash_ids: List[int]) -> None:
    if hash_ids:
        pipe = redis_client.pipeline()
        for hash_id in hash_ids:
            pipe.xadd(
                'unparsed_change',
                {'hash_id': hash_id.to_bytes(8, 'big', signed=True)},
                maxlen=1_000_000,
                approximate=True,
            )
        pipe.execute()

def add_plan(redis_client: Redis, hash_ids: List[int]) -> None:
    if hash_ids:
        pipe = redis_client.pipeline()
        for hash_id in hash_ids:
            pipe.xadd(
                'unparsed_plan',
                {'hash_id': hash_id.to_bytes(8, 'big', signed=True)},
                maxlen=1_000_000,
                approximate=True,
            )
        pipe.execute()

def add(redis_client: Redis, hash_ids: List[int]) -> None:
    if hash_ids:
        for hash_id in hash_ids:
            redis_client.xadd(
                'unparsed',
                {'hash_id': hash_id.to_bytes(8, 'big', signed=True)},
                maxlen=1_000_000,
                approximate=True,
            )


def get(redis_client: Redis, from_id: bytes) -> Tuple[bytes, List[int]]:
    resp = redis_client.xread({'unparsed': from_id})
    if resp:
        hash_ids = set()
        for last_id, data in resp[0][1]:
            hash_ids.add(int.from_bytes(data[b'hash_id'], 'big', signed=True))
        return last_id, list(hash_ids)
    else:
        return from_id, []


if __name__ == '__main__':
    from config import redis_url
    from rtd_crawler.hash64 import hash64

    redis_connection = Redis.from_url(redis_url)

    hash_ids = sorted([hash64(str(i)) for i in range(10)])
    print(hash_ids)
    add(redis_connection, hash_ids)

    from_id = b'0-0'
    last_id, hash_ids_redis = get(redis_connection, from_id)
    hash_ids_redis = sorted(hash_ids_redis)
    print(last_id)
    print(hash_ids_redis)
