from datetime import datetime
from typing import List, Union


def db_to_datetime(dt: Union[str, None]) -> Union[datetime, None]:
    """
    Convert bahn time in format: '%y%m%d%H%M' to datetime.
    As it is fastest to directly construct a datetime object from this, no strptime is used.

    Args:
        dt (str): bahn time format

    Returns:
        datetime.datetime: converted bahn time
    """
    if dt is None:
        return None
    return datetime(
        int('20' + dt[0:2]), int(dt[2:4]), int(dt[4:6]), int(dt[6:8]), int(dt[8:10])
    )


def parse_path(path: Union[str, None]) -> Union[List[str], None]:
    if path is None or not path:
        return None
    return path.split('|')


def parse_id(id: str) -> tuple[int, datetime, int]:
    """
    Parse a stop_id into its components

    Parameters
    ----------
    id : str
        A stop_id

    Returns
    -------
    tuple[int, datetime, int]
        trip_id, date_id, stop_id
    """
    trip_id, date_id, stop_id = id.rsplit('-', 2)
    return int(trip_id), db_to_datetime(date_id), int(stop_id)
