import math
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_AUTH, NEO4J_URI
from database.ris_transfer_time import Connection, Platform, get_transfer_time
from helpers import pairwise


def remove_walking_segments(fptf_journey_legs: List[Dict]) -> List[Dict]:
    return [leg for leg in fptf_journey_legs if leg.get('walking', False) != True]


def get_needed_transfer_times(fptf_journey_legs: List[Dict]) -> Iterable[Connection]:
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


def shift_predictions_by_transfer_time(
    ar_predictions: np.ndarray,
    dp_predictions: np.ndarray,
    transfer_times: np.ndarray,
    needed_transfer_times: Iterable[Connection],
) -> Tuple[np.ndarray, np.ndarray]:
    for i, (minutes, needed_transfer_time) in enumerate(
        zip(transfer_times, needed_transfer_times)
    ):
        transfer_time = timedelta(minutes=minutes.item())
        transfer_puffer = (
            transfer_time - needed_transfer_time.frequent_traveller.duration
        )

        total_minutes = math.ceil(transfer_puffer.total_seconds() / 60)

        if transfer_puffer == timedelta():
            continue
        elif transfer_puffer < timedelta():
            dp_predictions[i] = np.roll(dp_predictions[i], -total_minutes)
            dp_predictions[i, -total_minutes:] = 0
        elif transfer_puffer > timedelta(minutes=14):
            ar_predictions[i] = np.ones(ar_predictions[i].shape)
            dp_predictions[i] = np.ones(dp_predictions[i].shape)
        else:
            ar_predictions[i] = np.roll(ar_predictions[i], -total_minutes)
            ar_predictions[i, -total_minutes:] = 0

    return ar_predictions, dp_predictions
