from datetime import timedelta

TRAIN_SPEED_MS = 70  # ~ 250 km/h
MAX_SECONDS_DRIVING_AWAY = int(30_000 / TRAIN_SPEED_MS)  # driving away for 30 km
MAX_TIME_DRIVING_AWAY = timedelta(
    seconds=MAX_SECONDS_DRIVING_AWAY
)
MINIMAL_TRANSFER_TIME = timedelta(seconds=60 * 2)
MINIMAL_TIME_FOR_NEXT_SEARCH = timedelta(seconds=60)
N_MINIMAL_ROUTES_TO_DESTINATION = 8
MAX_TRANSFERS = 4
NO_TRIP_ID = 0
MAX_EXPECTED_DELAY_SECONDS = 60 * 30

MINIMAL_DISTANCE_DIFFERENCE = 1000  # 1 km


def EXTRA_TIME(duration: timedelta) -> timedelta:
    # return duration
    return duration + min(
        max(
            timedelta(seconds=60 * 3),
            timedelta(seconds=duration.total_seconds() * 0.05),
        ),
        timedelta(seconds=60 * 15),
    )


def EXTRA_TIME_UNIX(duration: int) -> int:
    return duration + min(
        max(
            60 * 3,
            int(duration * 0.05),
        ),
        60 * 15,
    )