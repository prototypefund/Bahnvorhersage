import itertools
import json
import warnings
from functools import lru_cache
from typing import List
from datetime import datetime

import igraph
from cityhash import CityHash64
from redis import Redis

from config import redis_url
from database import cached_sql_fetch
from helpers import pairwise
from helpers.StationPhillip import DateSelector, StationPhillip


def redis_lru_cache_str_float(name: str, maxsize: int = 128):
    redis_client = Redis.from_url(redis_url)

    def wrapper(func):
        @lru_cache(maxsize)
        def inner(self, **kwargs):
            hashable_kwargs = {}
            for key in kwargs:
                if key != 'date':
                    hashable_kwargs[key] = kwargs[key]
            cache_key = hex(CityHash64(json.dumps(hashable_kwargs, sort_keys=True)))
            result = redis_client.hget(name, cache_key)
            if result is not None:
                return float(result)
            else:
                result = func(self, **kwargs)
                redis_client.hset(name, cache_key, result)
                return result

        return inner

    return wrapper


class StreckennetzSteffi(StationPhillip):
    streckennetz: igraph.Graph

    def __init__(self, **kwargs):
        if 'generate' in kwargs:
            kwargs['generate'] = False
            print('StreckennetzSteffi does not support generate')

        super().__init__(**kwargs)

        self._get_streckennetz(**kwargs)

    def _get_streckennetz(self, **kwargs):
        edges = cached_sql_fetch(
            sql='SELECT source, target, length FROM streckennetz_zwei_null_simplified_edges',
            **kwargs,
        )
        nodes = cached_sql_fetch(
            sql='SELECT name, railway, x, y FROM streckennetz_zwei_null_simplified_nodes ORDER BY level_0',
            **kwargs,
        )
        self.streckennetz = igraph.Graph.DataFrame(
            edges=edges, vertices=nodes, directed=False
        )

    def network_distance(self, source: str, target: str) -> float:
        source_nodes = self.streckennetz.vs.select(name_eq=source)
        target_nodes = self.streckennetz.vs.select(name_eq=target)

        if not source_nodes or not target_nodes:
            raise KeyError(f'Could not find {source} or {target} in network')

        path_lengths = []
        for source, target in itertools.product(source_nodes, target_nodes):
            path = self.streckennetz.get_shortest_paths(
                source, target, 'length', output='epath'
            )[0]

            path_lengths.append(sum(self.streckennetz.es[path]['length']))

        return min(path_lengths)

    @redis_lru_cache_str_float(name='best_distance_cache', maxsize=None)
    def best_distance(
        self, source: str, target: str, is_bus: bool
    ) -> float:
        try:
            geo_distance = self.geographic_distance(source, target)

            # For bus routes, use geographic distance, as the railway-network
            # distance would not make sense.
            if is_bus:
                return geo_distance
        except KeyError:
            return 0
        try:
            network_distance = self.network_distance(source, target)
        except KeyError:
            # warnings.warn(
            #     f'Could not find {source} or {target} in network. Using geographical distance.',
            #     RuntimeWarning,
            # )
            return geo_distance

        if geo_distance > network_distance:
            # warnings.warn(
            #     f'network distance ({network_distance:.0f} m) smaller than geographical '
            #     f'distance ({geo_distance:.0f} m) for {source} to {target}',
            #     RuntimeWarning,
            # )
            return geo_distance
        elif network_distance > geo_distance * 3:
            warnings.warn(
                f'network distance ({network_distance:.0f} m) is more than 3 times longer '
                f'than geographical distance ({geo_distance:.0f} m) for {source} to {target}',
                RuntimeWarning,
            )
            return geo_distance
        else:
            return network_distance

    def route_length(
        self, waypoints: List[str], is_bus: bool
    ) -> float:
        """
        Calculate approximate length of a route, e.g. the sum of the distances
        between the waypoints.

        Parameters
        ----------
        waypoints: list
            List of station names that describe the route.
        date: DateSelector
            Date of the route.

        Returns
        -------
        float:
            Length of route.

        """
        length = 0
        for source, target in pairwise(waypoints):
            length += self.best_distance(
                source=source, target=target, is_bus=is_bus
            )
        return length


if __name__ == '__main__':
    import helpers.bahn_vorhersage

    streckennetz_steffi = StreckennetzSteffi(prefer_cache=False)

    print(
        'Tübingen Hbf - Altingen(Württ):',
        streckennetz_steffi.route_length(
            ['Tübingen Hbf', 'Altingen(Württ)'],
            is_bus=False,
        ),
    )
    print(
        'Tübingen Hbf - Reutlingen Hbf:',
        streckennetz_steffi.route_length(
            ['Tübingen Hbf', 'Reutlingen Hbf'],
            date='latest',
            is_bus=False,
        ),
    )
    print(
        'Tübingen Hbf - Stuttgart Hbf:',
        streckennetz_steffi.route_length(
            ['Tübingen Hbf', 'Stuttgart Hbf'],
            date='latest',
            is_bus=False,
        ),
    )
    print(
        'Tübingen Hbf - Ulm Hbf:',
        streckennetz_steffi.route_length(
            ['Tübingen Hbf', 'Ulm Hbf'],
            date='latest',
            is_bus=False,
        ),
    )

    # An der Bahnbrücke, Chemnitz
