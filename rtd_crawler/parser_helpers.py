from typing import Union, List
import datetime

def db_to_datetime(dt: Union[str, None]) -> Union[datetime.datetime, None]:
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
    return datetime.datetime(int('20' + dt[0:2]), int(dt[2:4]), int(dt[4:6]), int(dt[6:8]), int(dt[8:10]))


def parse_path(path: Union[str, None]) -> Union[List[str], None]:
    if path is None or not path:
        return None
    return path.split('|')