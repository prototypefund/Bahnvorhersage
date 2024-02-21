import math
from datetime import timedelta
from itertools import pairwise
from typing import Dict, Iterable, List, Tuple

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_AUTH, NEO4J_URI
from database.ris_transfer_time import (Platform, TransferInfo,
                                        get_transfer_time)


def remove_walking_segments(fptf_journey_legs: List[Dict]) -> List[Dict]:
    return [leg for leg in fptf_journey_legs if leg.get('walking', False) != True]


def get_needed_transfer_times(fptf_journey_legs: List[Dict]) -> Iterable[TransferInfo]:
    fptf_journey_legs = remove_walking_segments(fptf_journey_legs)

    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        with driver.session() as session:
            for arriving, departing in pairwise(fptf_journey_legs):
                start = Platform(
                    eva=int(arriving['destination']['id']),
                    platform=arriving.get('arrivalPlatform', None),
                )

                destination = Platform(
                    eva=int(departing['origin']['id']),
                    platform=departing.get('departurePlatform', None),
                )

                yield get_transfer_time(session, start, destination)
