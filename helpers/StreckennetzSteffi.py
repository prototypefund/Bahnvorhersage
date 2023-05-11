import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopy.distance
from helpers import StationPhillip
from database import cached_table_fetch #, cached_table_push, get_engine
import igraph
# import pandas as pd
# import sqlalchemy
# from sqlalchemy import Column, String, INT
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.dialects import postgresql
# import random
# import time
import datetime

from functools import lru_cache
from redis import Redis
from config import redis_url


def redis_lru_cache_str_float(name: str, maxsize: int = 128):
    """
    """
    redis_client = Redis.from_url(redis_url)
    def wrapper(func):

        @lru_cache(maxsize)
        def inner(sefl, cache_key: str, *args, **kwargs):
            result = redis_client.hget(name, cache_key)
            if result is not None:
                return float(result)
            else:
                result = func(sefl, *args, **kwargs)
                redis_client.hset(name, cache_key, result)
                return result

        return inner
    return wrapper


class StreckennetzSteffi(StationPhillip):
    def __init__(self, **kwargs):
        if "generate" in kwargs:
            kwargs["generate"] = False
            print("StreckennetzSteffi does not support generate")

        self.kwargs = kwargs

        super().__init__(**kwargs)

        streckennetz_df = cached_table_fetch("minimal_streckennetz", **kwargs)

        # Lazy loading of persistent cache, as many programs will not use it
        self.persistent_path_cache = None
        self.original_persistent_path_cache_len = None

        nodes = list(
            set(streckennetz_df['u'].to_list() + streckennetz_df['v'].to_list())
        )
        node_ids = dict(zip(nodes, range(len(nodes))))
        edges = list(
            zip(
                streckennetz_df["u"].map(node_ids.get).to_list(),
                streckennetz_df["v"].map(node_ids.get).to_list(),
            )
        )
        self.streckennetz_igraph = igraph.Graph(
            n=len(nodes),
            edges=edges,
            directed=False,
            vertex_attrs={'name': nodes},
            edge_attrs={'length': streckennetz_df['length'].to_list()},
        )

        self.get_length = lambda edge: self.streckennetz_igraph.es[edge]["length"]

    def route_length(self, waypoints, date: datetime.datetime) -> float:
        """
        Calculate approximate length of a route, e.g. the sum of the distances between the waypoints.

        Parameters
        ----------
        waypoints: list
            List of station names that describe the route.
        date: datetime.datetime
            Date of the route.

        Returns
        -------
        float:
            Length of route.

        """
        length = 0
        for i in range(len(waypoints) - 1):
            try:
                length += self.distance(
                    waypoints[i] + '__' + waypoints[i + 1],
                    waypoints[i],
                    waypoints[i + 1],
                    date,
                )
            except KeyError:
                pass
        return length

    def eva_route_length(self, waypoints, date: datetime.datetime) -> float:
        """
        Calculate approximate length of a route, e.g. the sum of the distances between the waypoints.

        Parameters
        ----------
        waypoints: list
            List of station evas that describe the route.
        date: datetime.datetime
            Date of the route.

        Returns
        -------
        float:
            Length of route.

        """
        length = 0
        for i in range(len(waypoints) - 1):
            try:
                length += self.distance(
                    *sorted(
                        self.get_name(eva=waypoints[i], date=date),
                        self.get_name(eva=waypoints[i + 1], date=date),
                    ),
                    date=date,
                )
            except KeyError:
                pass
        return length

    def get_edge_path(self, source, target):
        try:
            return self.streckennetz_igraph.get_shortest_paths(
                source, target, weights="length", output="epath"
            )[0]
        except ValueError:
            return None

    @redis_lru_cache_str_float(name='station_distance', maxsize=None)
    def distance(self, u: str, v: str, date: datetime.datetime) -> float:
        """
        Calculate approx distance between two stations. Uses the Streckennetz if u and v are part of it,
        otherwise it usese geopy.distance.distance.

        Parameters
        ----------
        u: str
            Station name
        v: str
            Station name
        date: datetime.datetime
            Date to use for the calculation

        Returns
        -------
        float:
            Distance in meters between u and v.
        """
        path = self.get_edge_path(u, v)
        if path is not None:
            return sum(map(self.get_length, path))
        else:
            try:
                u_coords = self.get_location(
                    name=u, date=date, allow_duplicates='first'
                )
                v_coords = self.get_location(
                    name=v, date=date, allow_duplicates='first'
                )
                return geopy.distance.distance(u_coords, v_coords).meters
            except KeyError:
                return 0


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    streckennetz_steffi = StreckennetzSteffi(prefer_cache=True)

    print(
        "Tübingen Hbf - Altingen(Württ):",
        streckennetz_steffi.route_length(
            ["Tübingen Hbf", "Altingen(Württ)"], date=datetime.datetime.now()
        ),
    )
    print(
        "Tübingen Hbf - Reutlingen Hbf:",
        streckennetz_steffi.route_length(
            ["Tübingen Hbf", "Reutlingen Hbf"], date=datetime.datetime.now()
        ),
    )
    print(
        "Tübingen Hbf - Stuttgart Hbf:",
        streckennetz_steffi.route_length(
            ["Tübingen Hbf", "Stuttgart Hbf"], date=datetime.datetime.now()
        ),
    )
    print(
        "Tübingen Hbf - Ulm Hbf:",
        streckennetz_steffi.route_length(
            ["Tübingen Hbf", "Ulm Hbf"], date=datetime.datetime.now()
        ),
    )
