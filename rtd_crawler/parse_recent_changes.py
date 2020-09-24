import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import datetime
import progressbar
from helpers.StationPhillip import StationPhillip
from rtd_crawler.hash64 import hash64
from database.plan import PlanManager
from database.change import ChangeManager
from database.rtd import RtdManager, sql_types
import json

empty_rtd = {key: None for key in sql_types.keys()}

plan_db = PlanManager()
change_db = ChangeManager()


# These are the names of columns that contain time information and should be parsed into a datetime
time_names = ('pt', 'ct', 'clt', 'ts')
message_parts_to_parse = ('id', 't', 'c', 'ts')


def db_to_datetime(dt) -> datetime.datetime:
    """
    Convert bahn time in format: '%y%m%d%H%M' to datetime.
    As it it fastest to directly construct a datetime object from this, no strptime is used.

    Args:
        dt (str): bahn time format

    Returns:
        datetime.datetime: converted bahn time
    """
    return datetime.datetime(int('20' + dt[0:2]), int(dt[2:4]), int(dt[4:6]), int(dt[6:8]), int(dt[8:10]))


def parse_stop_plan(stop: dict) -> dict:
    """
    Parse a planned stop: Add index and flatten the arrival and departure events.
    Parameters
    ----------
    stop : dict
        Stop from the Timetables API

    Returns
    -------
    dict
        Parsed Stop

    """
    # Create a int64 hash to be used as index.
    stop['hash_id'] = hash64(stop['id'])
    if 'tl' in stop:
        for key in stop['tl'][0]:
            stop[key] = stop['tl'][0][key]
        stop.pop('tl')
    if 'ar' in stop:
        for key in stop['ar'][0]:
            if key in time_names:
                stop['ar_' + key] = db_to_datetime(stop['ar'][0][key])
            else:
                stop['ar_' + key] = stop['ar'][0][key]
        stop.pop('ar')
    if 'dp' in stop:
        for key in stop['dp'][0]:
            if key in time_names:
                stop['dp_' + key] = db_to_datetime(stop['dp'][0][key])
            else:
                stop['dp_' + key] = stop['dp'][0][key]
        stop.pop('dp')
    return stop


def add_change_to_stop(stop: dict, change: dict) -> dict:
    """
    Add realtime changes to a stop.

    Parameters
    ----------
    stop : dict
        A parsed stop from parse_plan_stop().
    change : dict
        A stop from the timetables API with realtime changes.

    Returns
    -------
    dict
        The stop with realtime changes added to it.

    """
    # Add arrival changes
    if 'ar' in change:
        for key in change['ar'][0]:
            if key == 'm':
                for msg in change['ar'][0][key]:
                    for msg_part in msg:
                        if msg_part in message_parts_to_parse:
                            if 'ar_m_' + msg_part not in stop:
                                stop['ar_m_' + msg_part] = []
                            if msg_part in time_names:
                                stop['ar_m_' + msg_part].append(db_to_datetime(msg[msg_part]))
                            elif msg_part == 'c':
                                stop['ar_m_c'].append(int(msg[msg_part]))
                            else:
                                stop['ar_m_' + msg_part].append(msg[msg_part])
            else:
                if key in time_names:
                    stop['ar_' + key] = db_to_datetime(change['ar'][0][key])
                else:
                    stop['ar_' + key] = change['ar'][0][key]

    # Add departure changes
    if 'dp' in change:
        for key in change['dp'][0]:
            if key == 'm':
                for msg in change['dp'][0][key]:
                    for msg_part in msg:
                        if msg_part in message_parts_to_parse:
                            if 'dp_m_' + msg_part not in stop:
                                stop['dp_m_' + msg_part] = []
                            if msg_part in time_names:
                                stop['dp_m_' + msg_part].append(db_to_datetime(msg[msg_part]))
                            elif msg_part == 'c':
                                stop['dp_m_c'].append(int(msg[msg_part]))
                            else:
                                stop['dp_m_' + msg_part].append(msg[msg_part])
            else:
                if key in time_names:
                    stop['dp_' + key] = db_to_datetime(change['dp'][0][key])
                else:
                    stop['dp_' + key] = change['dp'][0][key]
    # Add message that is associated with the hole stop.
    if 'm' in change:
        for msg in change['m']:
            for msg_part in msg:
                if msg_part in message_parts_to_parse:
                    if 'm_' + msg_part not in stop:
                        stop['m_' + msg_part] = []
                    if msg_part in time_names:
                        stop['m_' + msg_part].append(db_to_datetime(msg[msg_part]))
                    elif msg_part == 'c':
                        stop['m_c'].append(int(msg[msg_part]))
                    else:
                        stop['m_' + msg_part].append(msg[msg_part])
    return stop


def parse_timetable(timetables):
    parsed = []
    timetables = [timetable.plan for timetable in timetables]
    train_ids_to_get = []
    for timetable in timetables:
        train_ids_to_get.extend(timetable.keys())
    changes = change_db.get_changes(train_ids_to_get)
    changes = {change.hash_id: json.loads(change.change) for change in changes}
    for timetable in timetables:
        if timetable is None:
            continue
        for stop in timetable.values():
            stop = parse_stop_plan(stop)

            if stop['hash_id'] in changes:
                stop = add_change_to_stop(stop, changes[stop['hash_id']])
            if stop:
                parsed.append(stop)
    return parsed


if __name__ == "__main__":
    import fancy_print_tcp
    stations = StationPhillip()
    rtd = RtdManager()

    if input('Do you wish to only parse new data? ([y]/n)') == 'n':
        start_date = datetime.datetime(2020, 1, 1, 0, 0)
    else:
        start_date = rtd.max_date() - datetime.timedelta(days=2)

    end_date = datetime.datetime.now()
    with progressbar.ProgressBar(max_value=len(stations)) as bar:
        buffer = []
        buffer_len = 0
        for i, station in enumerate(stations):
            stations_timetables = plan_db.plan_of_station(station, date1=start_date, date2=end_date)

            parsed = parse_timetable(stations_timetables)

            if parsed:
                parsed = pd.DataFrame(parsed)
                parsed = parsed.set_index('hash_id')
                # Remove duplicates. Duplicates may happen if a stop is shifted to the next hour due to delays.
                # It than reappears in the planned timetable of the next hour.
                parsed = parsed.loc[~parsed.index.duplicated(keep='last')]
                parsed['station'] = station
                buffer.append(parsed)
                buffer_len += len(parsed)

                if buffer_len > 10000:
                    rtd.upsert(pd.concat(buffer, ignore_index=False))
                    buffer = []
                    buffer_len = 0

            bar.update(i)
        if buffer:
            rtd.upsert(pd.concat(buffer, ignore_index=False))