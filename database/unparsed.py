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


def get_change(redis_client: Redis, from_id: bytes) -> Tuple[bytes, List[int]]:
    resp = redis_client.xread({'unparsed_change': from_id})
    if resp:
        hash_ids = set()
        for last_id, data in resp[0][1]:
            hash_ids.add(int.from_bytes(data[b'hash_id'], 'big', signed=True))
        return last_id, list(hash_ids)
    else:
        return from_id, []


def get_plan(redis_client: Redis, from_id: bytes) -> Tuple[bytes, List[int]]:
    resp = redis_client.xread({'unparsed_plan': from_id})
    if resp:
        hash_ids = set()
        for last_id, data in resp[0][1]:
            hash_ids.add(int.from_bytes(data[b'hash_id'], 'big', signed=True))
        return last_id, list(hash_ids)
    else:
        return from_id, []


def get(redis_client: Redis, from_id: bytes) -> Tuple[bytes, List[int]]:
    resp = redis_client.xread({'unparsed': from_id})
    if resp:
        hash_ids = set()
        for last_id, data in resp[0][1]:
            hash_ids.add(int.from_bytes(data[b'hash_id'], 'big', signed=True))
        return last_id, list(hash_ids)
    else:
        return from_id, []
