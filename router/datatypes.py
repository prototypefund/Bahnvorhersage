from collections import namedtuple


# Connection from one stop to the next,
# no stopovers in between
Connection = namedtuple(
    'Connection',
    [
        'dp_ts',
        'ar_ts',
        'dp_stop_id',
        'ar_stop_id',
        'trip_id',
        'is_regio',
        'dist_traveled',
    ],
)

Reachability = namedtuple(
    'Reachability',
    [
        'dp_ts',            # departure time at the origin stop
        'ar_ts',            # arrival time at this stop
        'transfers',        # number of transfers on the way to this stop
        'dist_traveled',    # distance traveled on the way to this stop
        'is_regio',         # way to this stop is only regio
        'current_trip_id',  # trip id used to reach this stop
        'min_heuristic',    # minimum heuristic on the way to this stop
        'r_ident_id',       # unique id for this reachability
        'last_r_ident_id',  # id of the reachability that was used to reach this one
        'last_stop_id',
        'last_dp_ts',
    ],
)

AlternativeReachability = namedtuple(
    'AlternativeReachability',
    [
        'dp_ts',
        'ar_ts',
        'transfers',
        'dist_traveled',
        'is_regio',
        'transfer_time_from_delayed_trip',
        'from_failed_transfer_stop_id',
        'current_trip_id',
        'min_heuristic',
        'r_ident_id',
        'last_r_ident_id',
        'last_stop_id',
        'last_dp_ts',
    ],
)

Transfer = namedtuple(
    'Transfer',
    [
        'ar_ts',
        'dp_ts',
        'stop_id',
        'is_regio',
        'ar_trip_id',
        'previous_transfer_stop_id',
        'transfers',
        'dist_traveled',
    ],
)