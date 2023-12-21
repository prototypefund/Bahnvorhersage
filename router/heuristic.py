from datetime import timedelta

from helpers.StationPhillip import StationPhillip
from router.constants import TRAIN_SPEED_MS


def compute_heuristic(
    stations: StationPhillip, stop_id: int, destination_id: int
) -> timedelta:
    distance_m = stations.geographic_distance_by_eva(
        eva_1=stop_id, eva_2=destination_id
    )
    return timedelta(seconds=distance_m / TRAIN_SPEED_MS)
