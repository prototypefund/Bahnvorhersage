from router.constants import (EXTRA_DURATION_FOR_SHORTER_ROUTE,
                              MINIMAL_DISTANCE_DIFFERENCE, NO_TRIP_ID)
from router.datatypes import Reachability

DOMINANCE_WORSE = 1
DOMINANCE_EQUAL = 2
DOMINANCE_BETTER = 3


def dp_ts_dominance(reachability: Reachability, other: Reachability):
    if reachability.dp_ts > other.dp_ts:
        return DOMINANCE_BETTER
    elif reachability.dp_ts < other.dp_ts:
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def ar_ts_dominance(reachability: Reachability, other: Reachability):
    if reachability.ar_ts < other.ar_ts:
        return DOMINANCE_BETTER
    elif reachability.ar_ts > other.ar_ts:
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def changeovers_dominance(reachability: Reachability, other: Reachability):
    if reachability.changeovers < other.changeovers:
        return DOMINANCE_BETTER
    elif reachability.changeovers > other.changeovers:
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def dist_traveled_dominance(reachability: Reachability, other: Reachability):
    distance_difference = reachability.dist_traveled - other.dist_traveled
    if abs(distance_difference) < MINIMAL_DISTANCE_DIFFERENCE:
        return DOMINANCE_EQUAL
    elif distance_difference < 0:
        reachability_duration = reachability.ar_ts - reachability.dp_ts
        other_duration = other.ar_ts - other.dp_ts
        if reachability_duration > other_duration * EXTRA_DURATION_FOR_SHORTER_ROUTE:
            return DOMINANCE_EQUAL
        else:
            return DOMINANCE_BETTER
    else:
        return DOMINANCE_WORSE


def is_regio_dominance(reachability: Reachability, other: Reachability):
    if reachability.is_regio > other.is_regio:
        return DOMINANCE_BETTER
    elif reachability.is_regio < other.is_regio:
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def is_transfer_time_from_delayed_trip_dominance(
    reachability: Reachability, other: Reachability
):
    if (
        reachability.transfer_time_from_delayed_trip
        > other.transfer_time_from_delayed_trip
    ):
        return DOMINANCE_BETTER
    elif (
        reachability.transfer_time_from_delayed_trip
        < other.transfer_time_from_delayed_trip
    ):
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def is_from_failed_transfer_stop_id_dominance(
    reachability: Reachability, other: Reachability
):
    if reachability.from_failed_transfer_stop_id > other.from_failed_transfer_stop_id:
        return DOMINANCE_BETTER
    elif reachability.from_failed_transfer_stop_id < other.from_failed_transfer_stop_id:
        return DOMINANCE_WORSE
    else:
        return DOMINANCE_EQUAL


def relaxed_pareto_dominated(reachability: Reachability, other: Reachability) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    dominations = set(
        (
            dp_ts_dominance(reachability, other),
            ar_ts_dominance(reachability, other),
            changeovers_dominance(reachability, other),
            dist_traveled_dominance(reachability, other),
            is_regio_dominance(reachability, other),
        )
    )

    if DOMINANCE_WORSE in dominations and DOMINANCE_BETTER not in dominations:
        return True

    return False


def relaxed_alternative_pareto_dominated(
    reachability: Reachability, other: Reachability
) -> bool:
    # Starting points always dominate
    if other.current_trip_id == NO_TRIP_ID:
        return True

    dominations = set(
        (
            ar_ts_dominance(reachability, other),
            changeovers_dominance(reachability, other),
            dist_traveled_dominance(reachability, other),
            is_regio_dominance(reachability, other),
            is_transfer_time_from_delayed_trip_dominance(reachability, other),
            is_from_failed_transfer_stop_id_dominance(reachability, other),
        )
    )

    if DOMINANCE_WORSE in dominations and DOMINANCE_BETTER not in dominations:
        return True

    return False
